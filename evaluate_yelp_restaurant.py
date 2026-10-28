"""Independent binary restaurant check on UCI Sentiment Labelled Sentences Yelp subset.

The trained classifiers may output neutral, which counts as an error because
this source has only positive and negative reference labels.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from review_intelligence import SubmittedBiLSTMPredictor


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original UCI yelp_labelled.txt")
    parser.add_argument("--archive", required=True, help="Original SemEval 2014 archive.zip")
    parser.add_argument("--product-training", required=True, help="Original four-category train.csv")
    parser.add_argument("--output", default="artifacts/independent_yelp_evaluation.json")
    args = parser.parse_args()
    records = []
    for line in Path(args.source).read_text(encoding="utf-8-sig").splitlines():
        text, score = line.rsplit("\t", 1)
        assert score in {"0", "1"}
        records.append((text, "positive" if score == "1" else "negative"))
    assert len(records) == 1000
    unique = {}
    for text, label in records:
        key = normalized(text)
        if key in unique:
            assert unique[key][1] == label
        else:
            unique[key] = (text, label)
    duplicate_count = len(records) - len(unique)
    records = list(unique.values())
    keys = set(unique)

    # Check exact text overlap with all local sources used for restaurant fine-tuning.
    from train_restaurant_candidate import restaurant_examples
    import csv
    restaurant_train, _ = restaurant_examples(args.archive)
    overlap_restaurant = len(keys & {normalized(r["review"]) for r in restaurant_train})
    four_train = Path(args.product_training)
    with four_train.open(encoding="utf-8-sig", newline="") as handle:
        overlap_product = len(keys & {normalized(row["review"]) for row in csv.DictReader(handle)})
    if overlap_restaurant or overlap_product:
        excluded = {normalized(r["review"]) for r in restaurant_train}
        with four_train.open(encoding="utf-8-sig", newline="") as handle:
            excluded.update(normalized(row["review"]) for row in csv.DictReader(handle))
        records = [(text, label) for text, label in records if normalized(text) not in excluded]

    truth = [label for _, label in records]
    output = {"source": "UCI Sentiment Labelled Sentences: yelp_labelled.txt",
              "count": len(records), "duplicate_source_rows_removed": duplicate_count,
              "reference_labels": dict(Counter(truth)),
              "exact_normalized_overlap_with_restaurant_training_source": overlap_restaurant,
              "exact_normalized_overlap_with_product_training_source": overlap_product,
              "label_order": ["negative", "neutral", "positive"], "models": {}}
    for name, directory in (("short_review_product", "artifacts/short_review_candidate"),
                            ("restaurant_experimental", "artifacts/restaurant_candidate")):
        engine = SubmittedBiLSTMPredictor(directory)
        probabilities = engine.probabilities([text for text, _ in records])
        predicted = [engine.labels[i] for i in np.argmax(probabilities, axis=1)]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=["negative", "positive"], zero_division=0)
        output["models"][name] = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "binary_reference_macro_f1": float(f1_score(truth, predicted, labels=["negative", "positive"],
                                                       average="macro", zero_division=0)),
            "predicted_distribution": dict(Counter(predicted)),
            "confusion_rows_gold_negative_positive_columns_pred_negative_neutral_positive":
                confusion_matrix(truth, predicted, labels=["negative", "neutral", "positive"])[[0, 2]].tolist(),
            "per_reference_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                             "f1": float(f1[i]), "support": int(support[i])}
                                    for i, label in enumerate(("negative", "positive"))},
            "sample_errors": [{"review": records[i][0], "gold": truth[i], "predicted": predicted[i]}
                              for i in range(len(records)) if truth[i] != predicted[i]][:15],
        }
    output["limits"] = "Originally 500 positive/500 negative sentences; exact duplicates and any exact training overlaps excluded. Neutral model outputs count as errors. No neutral or mixed gold labels, no full-length reviews, and no claim of three-class accuracy. The models were selected before this set was opened."
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: result[key] for key in ("accuracy", "binary_reference_macro_f1", "predicted_distribution", "per_reference_class")}
                      for name, result in output["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
