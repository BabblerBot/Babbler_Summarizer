from fastapi import FastAPI
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import uvicorn
from typing import List
import sqlite3

MODEL = "pszemraj/led-base-book-summary"
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
        docs = get_chunks(url)
        summary = generate_summary(docs)
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


def get_url(book_id):
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


# fetch ebook and split into chunks (docs)
def get_chunks(url):
    loader = GutenbergLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=300,
        separators=["\r\n\n\n\r\n\n\n", "\r\n\n\n", "."],
    )

    text = loader.load()[0].page_content

    # remove PROJECT GUTENBERG header and footer sections
    start_marker = "***\r\n\n\n\r\n\n\n\r\n\n\n\r\n\n\n\r\n\n\n"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    text = text[start_index + len(start_marker) : end_index]

    # splitting
    docs = text_splitter.create_documents([text])
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.replace("\r\n\n\n", " ")
    print("created %d chunks." % len(docs))
    return docs


def generate_summary(docs):
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
        max_length=300,
        # temperature = 0.2,
        device=device,
    )

    print("Starting chunk summarization...")
    # chunk_summaries = {}
    initial_summary = ""
    for i in range(len(docs)):
        chunk_summary = summarizer(
            docs[i].page_content,
            min_length=100,
            max_length=300,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            repetition_penalty=3.5,
            num_beams=4,
            early_stopping=True,
        )[0]["summary_text"]
        print("chunk %d summerized." % i)
        # chunk_summaries[i]= chunk_summary
        initial_summary += chunk_summary + "\n"
        # for now just send the first summary
        # return initial_summary
    print("Chunk summarization completed.")

    print("Generating final summary")
    summary = summarizer(
        initial_summary,
        min_length=100,
        max_length=300,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=3.5,
        num_beams=4,
        early_stopping=True,
    )[0]["summary_text"]
    print("Final summary generated.")
    # summary = summarizer(
    #     docs[1].page_content,
    #     # final_summary,
    #     min_length=200,
    #     max_length=400,
    #     no_repeat_ngram_size=3,
    #     encoder_no_repeat_ngram_size=3,
    #     repetition_penalty=3.5,
    #     num_beams=4,
    #     early_stopping=True,
    # )[0]['summary_text']
    return summary
