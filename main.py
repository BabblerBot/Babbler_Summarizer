from fastapi import FastAPI
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nest_asyncio
from pyngrok import ngrok
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
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run("main:app", reload=False, port=8000)
