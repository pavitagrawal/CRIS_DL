"""Evaluate a fixed random review sample against independently entered human labels."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from review_queue import LABELS


def draw_sample(rows, seed, maximum=50):
    """Sample across all reviews, including both flagged and unflagged rows."""
    return rows.sample(n=min(maximum, len(rows)), random_state=seed).sort_values("review_row").copy()


def labeled_sample(sample, human_labels):
    result = sample.copy()
    result["human_sentiment"] = [human_labels.get(int(row), "") for row in result["review_row"]]
    if not set(result["human_sentiment"]).issubset(set(LABELS) | {""}):
        raise ValueError("Invalid human label")
    return result


def evaluate_sample(sample):
    """Require all labels to avoid selective partial-sample accuracy claims."""
    if sample.empty or sample["human_sentiment"].eq("").any():
        raise ValueError("Label every sampled review before evaluating")
    truth = sample["human_sentiment"]
    predicted = sample["predicted_sentiment"]
    wrong = truth.ne(predicted)
    flagged = sample["review_status"].eq("Needs review")
    return {
        "count": len(sample),
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0),
        "confusion": confusion_matrix(truth, predicted, labels=list(LABELS)),
        "errors": int(wrong.sum()),
        "flagged": int(flagged.sum()),
        "flagged_errors": int((wrong & flagged).sum()),
        "unflagged_errors": int((wrong & ~flagged).sum()),
    }
