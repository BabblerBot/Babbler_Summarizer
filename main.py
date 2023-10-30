from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from typing import List
import os
from dotenv import load_dotenv
from utils import get_url, get_chunks
from database import get_summary_from_db, store_summary
from summarize import generate_summary

load_dotenv()

MODEL = "pszemraj/led-large-book-summary"
USE_CACHE = True

ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_summary/{book_id}")
async def get_summary(book_id: str):
    result = get_summary_from_db(book_id)

    print(f"Use Cache : {USE_CACHE}")
    if result and USE_CACHE:
        print("Summary is available in cache")
        summary = result[0]
    else:
        print("Summary is not available in cache")
        print("GPU: ", torch.cuda.is_available())
        url = get_url(book_id)
        docs = get_chunks(url)
        summary = generate_summary(
            docs,
            model_string=MODEL,
            use_llm=True,
            book_id=book_id,
        )
        # add summary to database if not exist
        if result == None:
            print("Storing summary in cache")
            store_summary(book_id, summary)

    return {"summary": summary}


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000, domain=os.getenv("NGROK_DOMAIN"))
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run("main:app", reload=False, port=8000)
