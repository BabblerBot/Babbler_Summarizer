from fastapi import FastAPI
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = FastAPI()

@app.get("/get_summary/{url:path}")
async def get_summary(url: str):
    docs = get_chunks(url)
    summary = get_summary(docs)
    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run("main:app")

# fetch ebook and split into chunks (docs)
def get_chunks(url):
    loader = GutenbergLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=300, 
        separators=["\r\n\n\n\r\n\n\n", "\r\n\n\n", "."]
    )
    
    text = loader.load()[0].page_content

    # remove PROJECT GUTENBERG header and footer sections
    start_marker = "***\r\n\n\n\r\n\n\n\r\n\n\n\r\n\n\n\r\n\n\n"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    text = text[start_index + len(start_marker):end_index]

    # splitting
    docs = text_splitter.create_documents([text])
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.replace("\r\n\n\n", " ")
    return docs

def get_summary(docs):
    tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-large-book-summary")
    model = AutoModelForSeq2SeqLM.from_pretrained(
                "pszemraj/led-large-book-summary", 
                # load_in_8bit=True,
                #low_cpu_mem_usage=True,
    )
    summarizer = pipeline(
        task = "summarization",
        model = model,
        tokenizer = tokenizer,
        pad_token_id = tokenizer.eos_token_id,
        max_length = 300,
        temperature = 0.2,
        device=0 if torch.cuda.is_available() else -1,
    )

    # chunk_summaries = {}
    # text = ""
    # for i in range(len(docs)):
    #     chunk_summary = summarizer(
    #             docs[i].page_content,
    #             min_length=400,
    #             max_length=1000,
    #             no_repeat_ngram_size=3,
    #             encoder_no_repeat_ngram_size=3,
    #             repetition_penalty=3.5,
    #             num_beams=4,
    #             early_stopping=True,
    #         )[0]['summary_text']
    #     # chunk_summaries[i]= chunk_summary
    #     text += (chunk_summary + "\n")
    #
    # final_summary = summarizer(
    #             text,
    #             min_length=500,
    #             max_length=1200,
    #             no_repeat_ngram_size=3,
    #             encoder_no_repeat_ngram_size=3,
    #             repetition_penalty=3.5,
    #             num_beams=4,
    #             early_stopping=True,
    #         )[0]['summary_text']

    summary = summarizer(
        docs[1].page_content,
        # final_summary,
        min_length=200,
        max_length=400,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=3.5,
        num_beams=4,
        early_stopping=True,
    )[0]['summary_text']
    return summary
