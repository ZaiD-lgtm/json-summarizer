import json
import nltk
from summarizer.bart_ai import generate_summary_prompt, extractive, extract_entities, abstractive
from summarizer.metrics_llm_eval import score_summary
nltk.download("punkt")

def evaluate_json(json_file_path: str):
    original_text, one, two = generate_summary_prompt(json_file_path)
    two_ = extractive(two)
    one = one.replace("\n", " ")
    one_entities = extract_entities(one)
    one_ = abstractive(one, one_entities)
    original = f"{one} \n\n {two}"
    final = f"{one_} \n\n {two_}"

    metrics = score_summary(original_text, final)

    print("\n--- Evaluation Metrics ---")
    print(f"BERTScore (F1): {metrics['bert_f1']:.4f}")
    print(f"Flesch Reading Ease: {metrics['flesch_score']:.4f}")
    print(f"Compression Ratio: {metrics['compression_ratio']:.2f}")
    print(f"Coverage: {metrics['coverage']:.2f}")
    print(f"Meteor Score: {metrics['meteor']:.4f}")
    print(f"=== Final Score: {metrics['final_score']:.4f} ===")

    return {
        "original_text": original,
        "summary": final,
        "metrics": metrics
    }


def update_json_with_results(json_file_path: str, summary: str, scores: dict, output_file: str = "updated_json.json"):
    """Update JSON with summary and evaluation results."""
    with open(json_file_path, "r") as f:
        data = json.load(f)

    data["generated_summary"] = summary
    data["summary_model"] = "facebook/bart-large-cnn"
    data["evaluation_scores"] = scores

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def run_pipeline(json_file_path: str, output_file: str = "updated_json.json"):
    result = evaluate_json(json_file_path)
    print("\n--- Summary ---\n", result["summary"])
    update_json_with_results(json_file_path, result["summary"], result["metrics"], output_file)


# if __name__ == "__main__":
    # json_file_path = "/home/zaid/Downloads/voice-agent/json_summarizer/summarizer/raman.json"
    # json_new = "/home/zaid/Downloads/voice-agent/raman-hospitals-history.json"
    #
    # output_file = "/home/zaid/Downloads/voice-agent/workspace/updated.json"
    # run_pipeline(json_new, output_file)