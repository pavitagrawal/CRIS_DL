"""Transparent review flags and human corrections, without changing model predictions."""

import re

LABELS = ("negative", "neutral", "positive")
SCORE_CUTOFF = 0.60  # Provisional triage rule, not a calibrated error probability.
STRONG_COMPLAINT = re.compile(
    r"\bi\s+(?:hate\s+(?:it|this|the)|dislike\s+(?:it|this|that|the))\b"
    r"|\b(?:will\s+not|won't|would\s+not|wouldn't)\s+(?:be\s+)?reorder\w*\b"
    r"|\b(?:would|do)\s+not\s+recommend\b", re.I)


def review_reasons(result, text="", alternate_sentiment=None):
    """Return visible reasons for manual review; these are not model corrections."""
    reasons = []
    if result.get("sentiment_method") == "explicit restaurant cue":
        reasons.append("Explicit restaurant cue differs from model")
    if result["confidence"] < SCORE_CUTOFF:
        reasons.append("Low model score")
    if result["sentiment"] == "neutral" and STRONG_COMPLAINT.search(text):
        reasons.append("Neutral model label with explicit complaint")
    summary = result["aspect_summary"].lower()
    if summary == "mixed":
        reasons.append("Mixed aspect signals")
    elif summary in LABELS and summary != result["sentiment"]:
        reasons.append("Overall/aspect disagreement")
    if alternate_sentiment in LABELS and alternate_sentiment != result["sentiment"]:
        reasons.append("Saved models disagree")
    return reasons


def apply_human_labels(rows, labels):
    """Keep raw predictions and make an explicitly attributed final label."""
    result = rows.copy()
    result["human_sentiment"] = [labels.get(int(row), "") for row in result["review_row"]]
    if not set(result["human_sentiment"]).issubset(set(LABELS) | {""}):
        raise ValueError("Human labels must be negative, neutral, or positive")
    result["final_sentiment"] = result["human_sentiment"].where(
        result["human_sentiment"].ne(""), result["predicted_sentiment"])
    result["label_source"] = result["human_sentiment"].apply(
        lambda value: "human" if value else "model")
    return result
