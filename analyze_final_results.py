"""Build the consolidated Phase 3 architecture and error analysis.

This script uses only the Python standard library. It reads the saved four-model
experiment and writes a machine-readable summary, a class-metrics CSV, and a
submission-ready Markdown analysis without retraining any model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


LABELS = ("negative", "neutral", "positive")
SOURCE = Path("artifacts/submitted_50k_comparison/results.json")
OUTPUT_JSON = Path("artifacts/final_architecture_analysis.json")
OUTPUT_CSV = Path("artifacts/final_class_metrics.csv")
OUTPUT_MARKDOWN = Path("FINAL_ARCHITECTURE_ANALYSIS.md")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def class_metrics(matrix: list[list[int]]) -> list[dict[str, int | float | str]]:
    total = sum(sum(row) for row in matrix)
    metrics = []
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        false_negative = sum(matrix[index]) - true_positive
        false_positive = sum(row[index] for row in matrix) - true_positive
        support = sum(matrix[index])
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        metrics.append(
            {
                "label": label,
                "support": support,
                "correct": true_positive,
                "errors": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "share_of_all_errors": safe_divide(false_negative, total - sum(matrix[i][i] for i in range(3))),
            }
        )
    return metrics


def largest_confusions(matrix: list[list[int]], limit: int = 3) -> list[dict[str, int | str]]:
    pairs = []
    for actual in range(len(LABELS)):
        for predicted in range(len(LABELS)):
            if actual != predicted:
                pairs.append(
                    {
                        "actual": LABELS[actual],
                        "predicted": LABELS[predicted],
                        "count": matrix[actual][predicted],
                    }
                )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:limit]


def main() -> None:
    results = json.loads(SOURCE.read_text(encoding="utf-8"))
    if {row["model"] for row in results} != {"MLP", "TextCNN", "BiLSTM", "Transformer"}:
        raise ValueError("Expected exactly the four planned architectures")

    winner = max(results, key=lambda row: row["validation"]["f1_macro"])
    analysis = {
        "source": str(SOURCE).replace("\\", "/"),
        "selection_rule": "highest validation macro F1",
        "selected_model": winner["model"],
        "models": [],
    }
    csv_rows = []
    for row in results:
        matrix = row["test"]["confusion"]
        per_class = class_metrics(matrix)
        model_analysis = {
            "model": row["model"],
            "selected_epoch": row["epoch"],
            "validation_macro_f1": row["validation"]["f1_macro"],
            "test_accuracy": row["test"]["accuracy"],
            "test_macro_f1": row["test"]["f1_macro"],
            "training_seconds": row["training_seconds"],
            "prediction_seconds": row["prediction_seconds"],
            "class_metrics": per_class,
            "largest_confusions": largest_confusions(matrix),
        }
        analysis["models"].append(model_analysis)
        for metric in per_class:
            csv_rows.append({"model": row["model"], **metric})

    selected = next(row for row in analysis["models"] if row["model"] == analysis["selected_model"])
    weakest = min(selected["class_metrics"], key=lambda metric: metric["recall"])
    analysis["selected_model_error_summary"] = {
        "weakest_recall_class": weakest["label"],
        "weakest_recall": weakest["recall"],
        "largest_confusions": selected["largest_confusions"],
        "interpretation": (
            "Neutral reviews remain the main failure mode. Rating-derived labels can disagree with the "
            "review text, so the confusion counts measure agreement with rating proxies rather than perfect "
            "human sentiment ground truth."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Final architecture and error analysis",
        "",
        "## Experiment decision",
        "",
        (
            f"The final four-model experiment selected **{analysis['selected_model']}** by validation macro F1. "
            f"Its held-out test accuracy was **{selected['test_accuracy']:.2%}** and test macro F1 was "
            f"**{selected['test_macro_f1']:.4f}**. All models used the same saved 35,000/7,500/7,500 split, "
            "training-only tokenization, and three rating-derived sentiment classes."
        ),
        "",
        "## Architecture comparison",
        "",
        "| Model | Selected epoch | Validation macro F1 | Test accuracy | Test macro F1 | Train time | Prediction time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in analysis["models"]:
        lines.append(
            f"| {model['model']} | {model['selected_epoch']} | {model['validation_macro_f1']:.4f} | "
            f"{model['test_accuracy']:.2%} | {model['test_macro_f1']:.4f} | "
            f"{model['training_seconds']:.1f} s | {model['prediction_seconds']:.2f} s |"
        )
    lines.extend(
        [
            "",
            "## Selected model class analysis",
            "",
            "| Class | Support | Correct | Errors | Precision | Recall | F1 | Share of model errors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in selected["class_metrics"]:
        lines.append(
            f"| {metric['label'].title()} | {metric['support']} | {metric['correct']} | {metric['errors']} | "
            f"{metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1']:.4f} | "
            f"{metric['share_of_all_errors']:.1%} |"
        )
    lines.extend(["", "## Largest BiLSTM confusions", ""])
    for confusion in selected["largest_confusions"]:
        lines.append(
            f"- Actual **{confusion['actual']}** predicted as **{confusion['predicted']}**: "
            f"{confusion['count']} reviews."
        )
    lines.extend(
        [
            "",
            "Neutral reviews have the lowest recall and account for most mistakes. The model most often maps "
            "neutral reviews to negative, followed by neutral reviews mapped to positive. This pattern supports "
            "the dashboard's human-review queue and the explicit warning that model scores are uncalibrated.",
            "",
            "## Interpretation limits",
            "",
            "The labels come from star ratings: 1-2 negative, 3 neutral, and 4-5 positive. A rating can disagree "
            "with the review wording. The experiment therefore measures reproducible agreement with rating proxies, "
            "not definitive human sentiment accuracy. The Transformer result is a compact model trained from scratch, "
            "not a pretrained language model result. No test metric was used to choose the winning architecture.",
            "",
            "## Reproduction",
            "",
            "Run `python prepare_multicategory_dataset.py` to recreate the documented sample, then run "
            "`python compare_submitted_models.py --splits-dir data/amazon_50k_multicategory --max-vocab 20000 "
            "--max-length 200 --epochs 5 --out artifacts/submitted_50k_comparison`. Finally run "
            "`python analyze_final_results.py` to rebuild this analysis.",
            "",
        ]
    )
    OUTPUT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
