import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import Replicate
from dotenv import load_dotenv
from utils import get_title

load_dotenv()


def half_summarize(curr_sum, summarizer, tokenizer, target_size):
    curr_len = len(tokenizer.encode(curr_sum))
    print("Target Token Size", target_size)
    while curr_len > target_size:
        print("Current Token Size", curr_len)
        curr_sum = summarizer(
            curr_sum,
            min_length=min(target_size, 100),
            max_length=min(target_size, curr_len // 2),
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            repetition_penalty=3.5,
            num_beams=4,
            early_stopping=True,
        )[0]["summary_text"]
        curr_len = len(tokenizer.encode(curr_sum))

    return curr_sum


def generate_summary(docs, model_string: str, use_llm=False, book_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_string,
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
    init_sum_len = len(tokenizer.encode(initial_summary))
    print("Tokensize of Chunk Summaries", init_sum_len)
    if use_llm:
        if init_sum_len > 4000:
            print("Chunk Summaries too long, truncating...")
            initial_summary = half_summarize(
                initial_summary, summarizer, tokenizer, 4000
            )
        print("Using LLM to generate final summary...")
        llm = Replicate(
            model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "temperature": 0.75,
                "max_new_tokens": 500,
                "top_p": 0.95,
                "repetition_penalty": 1.15,
            },
        )
        title = get_title(book_id)
        prompt = f"""Given the following book title and chunk summaries, generate a 200-400 word summary of the book using 
                     your knowledge of the book title and if needed, the chunk summaries. 
                    
                    Book Title: {title}
                     
                    Chunk Summaries: \n {initial_summary}

                    Your Summary:                    
                    """
        summary = llm(prompt)

    else:
        if init_sum_len > 16000:
            print("Chunk Summaries too long, truncating...")
            initial_summary = half_summarize(
                initial_summary, summarizer, tokenizer, 8000
            )
        print("Generating final summary...")
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
    return summary
