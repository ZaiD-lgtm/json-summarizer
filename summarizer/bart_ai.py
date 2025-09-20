import json
import nltk
import spacy
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

def ner_score(entities, total_tokens):
    entity_tokens = sum(len(ent.split()) for ent in entities)
    return entity_tokens / total_tokens if total_tokens > 0 else 0


def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [f"{ent.text} ({ent.label_})" for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE"]]

def extractive(text):
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences)


def abstractive(text, entities):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def choose_summarizer(text):
    tokens = text.split()
    entities = extract_entities(text)
    score = ner_score(entities, len(tokens))
    if score >= 0.15:
        return extractive(text), "extractive", score
    else:
        return abstractive(text, entities), "abstractive", score

def generate_summary_prompt(json_path: str):
    with open(json_path, "r") as file:
        json_data = json.load(file)
    history = json_data['history']
    final_result_text = None
    process_summary_text = "The agent performed a series of steps to find the requested information."

    for i in range(len(history) - 1, -1, -1):
        step = history[i]
        actions = step.get('model_output', {}).get('action', [])
        results = step.get('result', [])

        extract_action_index = -1
        for j, action in enumerate(actions):
            if 'extract_structured_data' in action:
                extract_action_index = j
                break

        if extract_action_index != -1 and extract_action_index < len(results):
            extracted_content = results[extract_action_index].get('extracted_content')
            if extracted_content:
                if "Query:" in extracted_content and "Result:" in extracted_content:
                    final_result_text = extracted_content.split("Result:", 1)[-1].strip()
                else:
                    final_result_text = extracted_content.strip()

                if i > 0:
                    previous_step_memory = history[i - 1].get('model_output', {}).get('memory')
                    if previous_step_memory:
                        process_summary_text = previous_step_memory
                break

    if not final_result_text:
        print("Warning: Could not find any successful 'extract_structured_data' action in the log.")
        final_memory = history[-1].get('model_output', {}).get('memory', "The task was completed.")
        final_result_text = f"No specific data was extracted. The final status was: '{final_memory}'"

    prompt = f"""{process_summary_text}\n{final_result_text}"""
    return prompt.strip(), process_summary_text, final_result_text

if __name__ == "__main__":
    json_file_path = "/home/zaid/Downloads/voice-agent/json_summarizer/summarizer/raman.json"
    json_new = "/home/zaid/Downloads/voice-agent/raman-hospitals-history.json"
    original_text,one,two = generate_summary_prompt(json_new)
    # print(f"one: {one}")
    # print(f"two: {two}")
    one_ = extractive(one)    # summary, model_used, ner_ratio = choose_summarizer(original_text)
    # print(f"{model_used} model used")
    # print("NER Ratio:", ner_ratio)
    # print("Final: \n", summary)
    two = two.replace("\n", " ")
    two_entities = extract_entities(two)
    two_ = abstractive(two, two_entities)

    final = f"{one_} \n\n {two_}"
    print(final)
    # summary, model_used, ner_ratio = choose_summarizer(original_text)
    # print(f"{model_used} model used")
    # print("NER Ratio:", ner_ratio)
    # print("Final: \n", summary)
