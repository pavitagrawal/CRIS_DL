"""Evaluate saved models against an independent, balanced restaurant star-rating sample.

Star groups are proxies for sentiment, especially ambiguous at three stars.
No training or checkpoint selection is performed by this script.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from review_intelligence import SubmittedBiLSTMPredictor


LABELS = ["negative", "neutral", "positive"]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/restaurant_rating_proxy_90.csv")
    parser.add_argument("--product-training", default="data/amazon_50k_multicategory/train.csv")
    parser.add_argument("--restaurant-training-archive", default="../../semeval14.zip")
    parser.add_argument("--output", default="artifacts/restaurant_rating_proxy_evaluation.json")
    args = parser.parse_args()
    with open(args.source, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 90
    records = [(r["text"].removeprefix("RATED  ").strip(),
                LABELS[0 if int(re.search(r"[1-5]", r["rating"]).group()) <= 2 else
                       1 if int(re.search(r"[1-5]", r["rating"]).group()) == 3 else 2]) for r in rows]
    assert all(text for text, _ in records)
    assert len({normalize(text) for text, _ in records}) == len(records)
    assert Counter(label for _, label in records) == dict.fromkeys(LABELS, 30)

    # These sources were used before this independent sample was chosen.
    training = set()
    if Path(args.product_training).exists():
        with open(args.product_training, newline="", encoding="utf-8-sig") as handle:
            training.update(normalize(r["review"]) for r in csv.DictReader(handle))
    from train_restaurant_candidate import restaurant_examples
    archive = Path(args.restaurant_training_archive)
    if archive.exists():
        restaurant_train, _ = restaurant_examples(archive)
        training.update(normalize(r["review"]) for r in restaurant_train)
    overlap = sum(normalize(text) in training for text, _ in records)
    if overlap:
        raise ValueError(f"{overlap} test reviews overlap with known training text")

    truth = [label for _, label in records]
    output = {
        "source": "https://github.com/skathirmani/datasets/blob/main/restaurant_reviews.csv",
        "selection": "30 per proxy class from the first 1,069 rated records; deterministic text-hash order within each class; no model predictions used for selection",
        "rating_to_proxy": {"1-2": "negative", "3": "neutral", "4-5": "positive"},
        "count": len(records), "proxy_class_counts": dict(Counter(truth)),
        "exact_normalized_training_overlap": overlap,
        "training_overlap_check_complete": Path(args.product_training).exists() and archive.exists(),
        "label_order": LABELS, "models": {},
    }
    for name, folder in (("short_review_product", "artifacts/short_review_candidate"),
                         ("restaurant_experimental", "artifacts/restaurant_candidate")):
        model = SubmittedBiLSTMPredictor(folder)
        probs = model.probabilities([text for text, _ in records])
        predicted = [model.labels[i] for i in np.argmax(probs, axis=1)]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=LABELS, zero_division=0)
        output["models"][name] = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(np.mean(f1)),
            "predicted_distribution": dict(Counter(predicted)),
            "confusion_rows_gold_columns_predicted": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                          for i, label in enumerate(LABELS)},
            "sample_errors": [{"text": records[i][0], "rating_proxy": truth[i], "predicted": predicted[i]}
                              for i in range(len(records)) if truth[i] != predicted[i]][:12],
        }
    output["limits"] = "Small English restaurant sample from one source, with only 39 low-rated reviews in the source slice. Three stars are an imperfect proxy for neutral sentiment: a three-star review may sound positive, negative, or mixed. Balanced sampling changes prevalence. This is rating-proxy agreement, not human-annotated three-class accuracy. The two models were trained before this sample was examined."
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                          "per_class_recall": {c: v["per_class"][c]["recall"] for c in LABELS}}
                      for k, v in output["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
