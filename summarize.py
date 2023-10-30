import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def generate_summary(docs, model_string: str):
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
