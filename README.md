# BabblerBot: Summarizer

Welcome to the BabblerBot Summarizer. This repository contains the book summarization component of the BablerBot project.
The book summarization component is responsible for taking a book (book ID) as input and generating a concise, informative summary of 200 to 300 words. 
The books will be fetched from the [Project Gutenberg site.](https://www.gutenberg.org/)

## General Architecture

The following image shows the abstract architecture of the Summarizer.
![architecture](https://drive.google.com/uc?id=/1s-GbolPtbKfKU8Y-WY4jXWj6gebfZ1qn)

Given the extensive length of books, it becomes essential to split them into smaller chunks to meet the token limitations imposed by available models.
We are currently exploring various techniques to optimize the summarization process and attain the highest-quality summaries.

## Components

- **Language Model:** [pszemraj/led-base-book-summary](https://huggingface.co/pszemraj/led-base-book-summary)
- **Backend Framework:** FastAPI 
