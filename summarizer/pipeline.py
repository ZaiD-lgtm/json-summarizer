import json
import nltk
from .bart_ai import parse_json, summarize, summarize_with_variants
from .metrics_llm_eval import score_summary

nltk.download("punkt")

def evaluate_json(json_file_path: str):
    """Parse JSON, summarize, and evaluate."""
    original_text = parse_json(json_file_path)
    summary, final_score = summarize_with_variants(original_text)
    metrics = score_summary(original_text, summary)

    print("\n--- Evaluation Metrics ---")
    print(f"BERTScore (F1): {metrics['bert_f1']:.4f}")
    print(f"Flesch Reading Ease: {metrics['flesch_score']:.4f}")
    print(f"Compression Ratio: {metrics['compression_ratio']:.2f}")
    print(f"Coverage: {metrics['coverage']:.2f}")
    print(f"Meteor Score: {metrics['meteor']:.4f}")
    print(f"=== Final Score: {metrics['final_score']:.4f} ===")

    return {
        "original_text": original_text,
        "summary": summary,
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
    """Run the full summarization and evaluation pipeline."""
    result = evaluate_json(json_file_path)
    print("\n--- Summary ---\n", result["summary"])
    update_json_with_results(json_file_path, result["summary"], result["metrics"], output_file)

#
#
# if __name__ == "__main__":
#     json_file_path = "../workspace/raman.json"
#     output_file = "../workspace/updated.json"
#     run_pipeline(json_file_path, output_file)