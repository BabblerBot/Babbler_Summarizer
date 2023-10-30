from fastapi import FastAPI
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import uvicorn
from typing import List
import sqlite3
import re

MODEL = "pszemraj/led-base-book-summary"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
USE_CACHE = True

app = FastAPI()


@app.get("/get_summary/{book_id}")
async def get_summary(book_id: str):
    # create database if not exist
    create_database()

    # check cache for summaries
    conn = sqlite3.connect('summary_cache.db')
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM summaries WHERE book_id=?", (book_id,))
    result = cursor.fetchone()

    print(f'Use Cache : {USE_CACHE}')
    if result and USE_CACHE:
        summary= result[0]
    else:
        print("Summary is not available in cache")
        print("GPU: ", torch.cuda.is_available())
        url = get_url(book_id)
        text = load_book(url)
        docs = split_doc(text)
        summary = get_final_summary(docs)
        # add summary to database if not exist
        if result == None:
            cursor.execute("INSERT INTO summaries (book_id, summary) VALUES(?,?)", (book_id, summary))
            conn.commit()
    conn.close()

    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)


def create_database():
    conn = sqlite3.connect('summary_cache.db')
    cursor = conn.cursor()
    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS summaries(
                        book_id TEXT PRIMARY KEY,
                        summary TEXT
                    )
                """)
    conn.commit()
    conn.close()


def get_token_size(text):
    return len(TOKENIZER.encode(text))

def get_url(book_id):
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def load_book(url):
    loader = GutenbergLoader(url)
    text = loader.load()[0].page_content

    # remove PROJECT GUTENBERG header and footer sections
    start_marker = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK (.*?)\*\*\*", text)
    end_marker = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
    if (start_marker != None) and (end_marker != None) :
        start_index = start_marker.end()
        end_index = end_marker.start()
        text = text[start_index : end_index]
    return text

# try to split document into chapters. if not split randomly
def split_doc(text):
    chapter_headings =  r"(?i)(?:chapter|section|part|episode|book)\s+[IVXLCDM]+\b.|\b(?:chapter|section|part|episode|book)\s+\d+\b."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\r\n\n\n\r\n\n\n','. '],
        chunk_size = 40000,
        length_function = len,
    )
    chapters = []
    matches = list(re.finditer(chapter_headings, text))
    
    # if chapter headings found split in to chapters
    if matches:
        print("chapter headings found.")
        heading_ids = []
        matched_headings = []
        for match in matches:
            # print(match)
            if match.group(0).lower() != "book i":
                matched_headings.append(match.group(0))
                heading_ids.append((match.start(),match.end()))
        heading_ids.append((-1,-1))
        
        removed = []
        for i in range(len(matched_headings)):
            chapter = text[heading_ids[i][1]:heading_ids[i+1][0]]
            chapter = re.sub(r'\s+', ' ', chapter).strip() # remove unnecessary white spaces
            if len(chapter) < 400:
                removed.append(chapter)
                continue
            if get_token_size(chapter) > 16000:
                print("chapter tokens exceed 16000. Spliting chapter again.")
                chapter = text_splitter.create_documents([chapter])
                chapter = [chap.page_content.replace('\r\n\n\n',' ') for chap in chapter]
                chapters.extend(chapter)
                continue
            chapters.append(chapter.replace('\r\n\n\n',' '))
    # if no chapter heading found split the text randomly.
    else:
        chapter = text_splitter.create_documents([text])
        chapter = [chap.page_content.replace('\r\n\n\n',' ') for chap in chapter]
        chapters.extend(chapter)
    print(f"number of splits: {len(chapters)}")   
    return chapters

# generate a summary for list of text inputs
def generate_summary(docs, max_summary_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        # load_in_8bit=True,
        # low_cpu_mem_usage=True,
    ).to(device)
    
    summarizer = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        # temperature = 0.2,
        device=device,
    )

    max_chunk_sum_len = max_summary_len // len(docs)
    if max_chunk_sum_len <= 150:
        max_chunk_sum_len = 200
        
    chunk_sums = []
    summary = ""
    
    print("Starting chunk summarization...")
    
    for i in range(len(docs)):
        if len(tokenizer.encode(docs[i])) < max_chunk_sum_len:
            print("chunk %d summerization skiped." % i)
            chunk_summary = docs[i]

        else:
            chunk_summary = summarizer(
                docs[i],
                min_length=150,
                max_length=max_chunk_sum_len,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,
                num_beams=4,
                # early_stopping=True,
            )[0]["summary_text"]
            print("chunk %d summerized." % i)

            
        chunk_sums.append(chunk_summary)
        summary += (chunk_summary + ". ")

    print("Chunk summarization completed.")
    print(f"Length:\t\t{len(summary)}\nToken size:\t\t{len(tokenizer.encode(summary))}")

    return summary, chunk_sums


def get_final_summary(docs):
    # iteration 01
    initial_summary, initial_chunk_sums =generate_summary(docs,20000)

    # iteration 02
    to_summarize = []
    for i in range(0,len(initial_chunk_sums)-1,2):
        to_summarize.append(initial_chunk_sums[i]+" "+initial_chunk_sums[i+1])
    if len(initial_chunk_sums)%2:
        to_summarize.append(initial_chunk_sums[-1])
    second_summary, _ =generate_summary(to_summarize,6000)

    # iteration 03
    third_summary, _ =generate_summary([second_summary],1200)

    # iteration 03
    final_summary, _ =generate_summary([third_summary],400)

    return final_summary
