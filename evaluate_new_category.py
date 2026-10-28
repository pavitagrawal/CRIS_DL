"""Exploratory cross-category evaluation on phone/accessory review ratings."""

import argparse
import csv
import hashlib
import itertools
import json
import random
from contextlib import closing
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from prepare_dataset import source_lines, LABEL_BY_RATING
from review_intelligence import LABELS, ReviewPredictor
from train import scores


PHONE_SOURCE = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
                "raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz")


def sample_reviews(lines, per_class=500, scan_limit=100000, seed=42, exclude_texts=()):
    """Reservoir sample within a bounded stream; source ordering may bias results."""
    rng = random.Random(seed)
    buckets = {label: [] for label in LABELS}
    counts = {label: 0 for label in LABELS}
    seen = {hashlib.blake2b(str(t).strip().lower().encode(), digest_size=16).digest()
            for t in exclude_texts}
    scanned = 0
    for line in itertools.islice(lines, scan_limit):
        scanned += 1
        item = json.loads(line)
        text = " ".join(str(item.get("text") or "").split())
        try:
            rating = int(item.get("rating"))
        except (ValueError, TypeError):
            continue
        if len(text) < 20 or rating not in LABEL_BY_RATING:
            continue
        digest = hashlib.blake2b(text.lower().encode(), digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        label = LABEL_BY_RATING[rating]
        counts[label] += 1
        row = (text, label)
        if len(buckets[label]) < per_class:
            buckets[label].append(row)
        else:
            position = rng.randrange(counts[label])
            if position < per_class:
                buckets[label][position] = row
    if any(len(buckets[label]) < per_class for label in LABELS):
        raise ValueError(f"Insufficient usable rows in first {scanned:,} source rows: {counts}")
    rows = [row for label in LABELS for row in buckets[label]]
    rng.shuffle(rows)
    return rows, scanned, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=PHONE_SOURCE, help="Official .jsonl.gz URL or local file")
    parser.add_argument("--training-csv", default="data/amazon_beauty_balanced.csv")
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--scan-limit", type=int, default=100000)
    parser.add_argument("--output", default="artifacts/phone_category_evaluation.json")
    parser.add_argument("--artifact-dir", default="artifacts/beauty_baseline",
                        help="Original beauty model by default; pass artifacts for the current model")
    args = parser.parse_args()
    if args.per_class < 1 or args.scan_limit < args.per_class * 3:
        parser.error("Provide a positive per-class count and a larger scan limit.")
    exclude = []
    if Path(args.training_csv).exists():
        with open(args.training_csv, newline="", encoding="utf-8") as stream:
            exclude = [row["review"] for row in csv.DictReader(stream)]
    print(f"Reading up to {args.scan_limit:,} rows from {args.source}", flush=True)
    with closing(source_lines(args.source)) as stream:
        rows, scanned, available = sample_reviews(stream, args.per_class, args.scan_limit,
                                                  exclude_texts=exclude)
    directory = args.artifact_dir if Path(args.artifact_dir, "best_model.keras").exists() else "artifacts"
    predictor = ReviewPredictor(directory)
    probabilities = predictor.probabilities([text for text, _ in rows])
    truth = np.array([LABELS.index(label) for _, label in rows])
    predicted = probabilities.argmax(axis=1)
    output = {
        "source": args.source, "category": "Cell_Phones_and_Accessories",
        "model_artifact_dir": directory,
        "scanned_source_rows": scanned, "usable_unique_in_scanned_prefix": available,
        "per_class_sample": args.per_class, "total_evaluated": len(rows), "seed": 42,
        "labels": LABELS, "mapping": "ratings 1-2 negative, 3 neutral, 4-5 positive",
        "metrics": scores(truth, predicted),
        "classification_report": classification_report(truth, predicted, labels=[0, 1, 2],
                                                       target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(truth, predicted,
                                                                          labels=[0, 1, 2]).tolist(),
        "limitations": "Bounded prefix sampling; source order may bias sample. Ratings are proxy labels. This tests overall sentiment only, not aspect sentiment.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("scanned_source_rows", "usable_unique_in_scanned_prefix",
                                               "total_evaluated", "metrics", "confusion_matrix_true_rows_predicted_columns")}, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
