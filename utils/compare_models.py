"""
utils/compare_models.py
------------------------
Comparative analysis script for all four sentiment classification models.

Loads saved result JSON files from the results/ folder and generates:
    1. A formatted comparison table (console output)
    2. A bar chart comparing Accuracy, Precision, Recall, F1-Score
    3. A bar chart comparing Training Time and Prediction Time
    4. A combined comparison summary saved as results/model_comparison.png

Run this after all four models have been trained.

Author : Aryan Sorout
Project : ICT 4442 - Deep Learning Mini Project

Usage:
    python utils/compare_models.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RESULTS_DIR = "results"

# Maps model name -> result JSON file
MODEL_FILES = {
    "MLP"         : os.path.join(RESULTS_DIR, "mlp_results.json"),
    "TextCNN"     : os.path.join(RESULTS_DIR, "textcnn_results.json"),
    "BiLSTM"      : os.path.join(RESULTS_DIR, "bilstm_results.json"),
    "Transformer" : os.path.join(RESULTS_DIR, "transformer_results.json"),
}

# Color scheme matching each model's confusion matrix theme
MODEL_COLORS = {
    "MLP"         : "#4C72B0",   # Blue
    "TextCNN"     : "#55A868",   # Green
    "BiLSTM"      : "#8172B2",   # Purple
    "Transformer" : "#C44E52",   # Orange-Red
}

# Metrics to compare
PERF_METRICS = ["accuracy", "precision", "recall", "f1_score"]
TIME_METRICS = ["train_time_sec", "pred_time_sec"]

METRIC_LABELS = {
    "accuracy"       : "Accuracy",
    "precision"      : "Precision (Macro)",
    "recall"         : "Recall (Macro)",
    "f1_score"       : "F1-Score (Macro)",
    "train_time_sec" : "Training Time (s)",
    "pred_time_sec"  : "Prediction Time (s)",
}


# ─────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────
def load_results() -> dict:
    """
    Load result JSON files for all available models.

    Returns:
        Dictionary of {model_name: result_dict}.
        Missing result files are skipped with a warning.
    """
    results = {}
    for model_name, filepath in MODEL_FILES.items():
        if os.path.exists(filepath):
            with open(filepath) as f:
                results[model_name] = json.load(f)
            print(f"[OK]      Loaded results for {model_name}")
        else:
            print(f"[MISSING] No results found for {model_name} at {filepath}")
    return results


# ─────────────────────────────────────────────
# PRINT COMPARISON TABLE
# ─────────────────────────────────────────────
def print_comparison_table(results: dict):
    """
    Print a formatted comparison table to the console.

    Args:
        results : Dictionary of {model_name: result_dict}.
    """
    models = list(results.keys())
    col_w  = 16

    header_metrics = PERF_METRICS + TIME_METRICS
    header = f"{'Model':<14}" + "".join(
        f"{METRIC_LABELS[m]:>{col_w}}" for m in header_metrics
    )

    print("\n" + "=" * (14 + col_w * len(header_metrics)))
    print("  MODEL COMPARISON — Customer Review Intelligence System")
    print("=" * (14 + col_w * len(header_metrics)))
    print(header)
    print("-" * (14 + col_w * len(header_metrics)))

    for model_name, res in results.items():
        row = f"{model_name:<14}"
        for m in header_metrics:
            val = res.get(m, "N/A")
            row += f"{str(val):>{col_w}}"
        print(row)

    print("=" * (14 + col_w * len(header_metrics)))

    # Highlight best performing model per metric
    print("\n  Best performing model per metric:")
    for m in PERF_METRICS:
        best_model = max(results, key=lambda k: results[k].get(m, 0))
        best_val   = results[best_model].get(m, "N/A")
        print(f"  {METRIC_LABELS[m]:<22}: {best_model} ({best_val})")

    fastest_train = min(results, key=lambda k: results[k].get("train_time_sec", float("inf")))
    fastest_pred  = min(results, key=lambda k: results[k].get("pred_time_sec",  float("inf")))
    print(f"  {'Fastest Training':<22}: {fastest_train} ({results[fastest_train].get('train_time_sec')}s)")
    print(f"  {'Fastest Prediction':<22}: {fastest_pred} ({results[fastest_pred].get('pred_time_sec')}s)")
    print()


# ─────────────────────────────────────────────
# PLOT COMPARISON CHARTS
# ─────────────────────────────────────────────
def plot_comparison(results: dict, save_path: str):
    """
    Generate and save a 2-row comparison chart:
        Row 1: Performance metrics (Accuracy, Precision, Recall, F1)
        Row 2: Time metrics (Training Time, Prediction Time)

    Args:
        results   : Dictionary of {model_name: result_dict}.
        save_path : Path to save the PNG file.
    """
    models = list(results.keys())
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    x      = np.arange(len(models))
    width  = 0.6

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Model Comparison — Customer Review Intelligence System\nICT 4442 Deep Learning Mini Project",
        fontsize=14, fontweight="bold", y=1.01
    )

    all_metrics = PERF_METRICS + TIME_METRICS

    for idx, metric in enumerate(all_metrics):
        row = idx // 4
        col = idx % 4
        ax  = axes[row][col]

        values = [results[m].get(metric, 0) for m in models]
        bars   = ax.bar(x, values, width=width, color=colors, edgecolor="white", linewidth=0.8)

        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        # Annotate bar tops with values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.4f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=8
            )

    # Legend
    legend_patches = [
        mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in models
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=len(models), fontsize=10,
        bbox_to_anchor=(0.5, -0.03)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Comparison chart saved: {save_path}")


# ─────────────────────────────────────────────
# SAVE RESULTS TABLE AS CSV
# ─────────────────────────────────────────────
def save_results_csv(results: dict, save_path: str):
    """
    Save the comparison table as a CSV file for easy import into reports.

    Args:
        results   : Dictionary of {model_name: result_dict}.
        save_path : Path to save the CSV file.
    """
    all_metrics = PERF_METRICS + TIME_METRICS
    header = "Model," + ",".join(METRIC_LABELS[m] for m in all_metrics)
    rows   = [header]

    for model_name, res in results.items():
        row = model_name + "," + ",".join(
            str(res.get(m, "N/A")) for m in all_metrics
        )
        rows.append(row)

    with open(save_path, "w") as f:
        f.write("\n".join(rows))

    print(f"[INFO] Results CSV saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  COMPARATIVE MODEL ANALYSIS")
    print("  Customer Review Intelligence System — ICT 4442")
    print("=" * 60 + "\n")

    results = load_results()

    if not results:
        print("\n[ERROR] No model results found in results/ folder.")
        print("Please train at least one model first using its train_*.py script.")
        exit(1)

    # Print table to console
    print_comparison_table(results)

    # Save comparison chart
    plot_comparison(
        results,
        save_path=os.path.join(RESULTS_DIR, "model_comparison.png")
    )

    # Save CSV
    save_results_csv(
        results,
        save_path=os.path.join(RESULTS_DIR, "model_comparison.csv")
    )

    print("\nDone! Check the results/ folder for:")
    print("  - model_comparison.png  (bar charts)")
    print("  - model_comparison.csv  (table for report)")
