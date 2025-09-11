import json
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
# from json_summarizer.summarizer.metrics_llm_eval import score_summary
from summarizer.metrics_llm_eval import score_summary

# pipeline = pipeline(
#     task="fill-mask",
#     model="facebook/bart-large",
#     dtype=torch.float16,
#     device=0
# )
# pipeline("Plants create <mask> through a process known as photosynthesis.")


def parse_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    history = data.get("history", [])

    summary_text = []
    for i, step in enumerate(history, start=1):
        model_output = step.get("model_output", {})
        thinking = model_output.get("thinking", "")
        next_goal = model_output.get("next_goal", "")
        actions = model_output.get("action", [])
        result = step.get("result", [])

        actions_str = "; ".join([str(a) for a in actions]) if actions else "No action"
        errors = [r.get("error", "") for r in result if r.get("error")]
        error_str = "; ".join(errors) if errors else "No errors"

        step_summary = (
            f"Step {i}: Thought: {thinking}. "
            f"Goal: {next_goal}. "
            f"Actions: {actions_str}. "
            f"Errors: {error_str}. "
        )
        summary_text.append(step_summary)

    final_text = " ".join(summary_text)
    return final_text

# print(final_text)

def summarize(original_text):
    # load model 1024 maximum sequence length
    model_name = "facebook/bart-large-cnn"
    # model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer([original_text], max_length=1024, return_tensors="pt", truncation=True) #encode text
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=500,          #max tokens in summary
        min_length=30,           #min tokens in summary
        early_stopping=True
    )

    # Decode and print summary
    #
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # print("\n\n-------------------Summary by bart-base model---------------- \n\n",summary)
    #

    #pipeline
    summarizer = pipeline("summarization", model = model_name)

    # print("/////////////////////////////////")
    summarized_text = summarizer(original_text)
    summarized_text = summarized_text[0]["summary_text"]
    # print(f"------------------Original Text:------------------- \n\n {original_text}")
    # print("Summarized_text: \n\n", summarized_text)
    return summarized_text

def summarize_with_variants(original_text, num_variants=4):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    prompts = [
        "Summarize the following text concisely:",
        "Write a detailed summary of the following text:",
        "Summarize in simple and clear terms:",
        "Write a high-level overview of the following text:"
    ]

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    summaries = []
    for prompt in prompts[:num_variants]:
        text_with_prompt = f"{prompt} {original_text}"
        summary = summarizer(
            text_with_prompt,
            max_length=500,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            truncation=True
        )[0]["summary_text"]
        summaries.append(summary)

    final_summary = ""
    score = -1
    for i in summaries:
        metrics = score_summary(original_text, i)
        if metrics["final_score"] > score:
            final_summary = i
            score = metrics["final_score"]


    return final_summary, score

#
# if __name__ == "__main__":
#     json_file_path = "../../workspace/raman.json"
#
#     original_text = parse_json(json_file_path)
#
#     summarized_text = summarize(original_text)
#
#     print(f"Original Text: \n\n {original_text}")
#     print("Summarized_text: \n\n", summarized_text)
#
#     metrics = score_summary(original_text, summarized_text)
#     print(f"Final Score: {metrics["final_score"]}")


