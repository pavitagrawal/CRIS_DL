"""Compare the narrow aspect agreement rule against the saved overall model."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from evaluate_new_category import PHONE_SOURCE, sample_reviews
from overall_rule import agreed_positive_aspects
from prepare_dataset import source_lines
from review_intelligence import LABELS, ReviewPredictor
from train import load_data, scores


def compare(engine, texts, truth):
    predictions = engine.analyze(texts)
    original = np.array([LABELS.index(row["sentiment"]) for row in predictions])
    revised = original.copy()
    changes = []
    for i, (text, row) in enumerate(zip(texts, predictions)):
        override = agreed_positive_aspects(text, row["sentiment"], row["aspects"])
        if override:
            revised[i] = LABELS.index(override)
            changes.append({"old": row["sentiment"], "new": override,
                            "gold": LABELS[truth[i]]})
    return {"reviews": len(texts), "model": scores(truth, original),
            "with_rule": scores(truth, revised), "changed": len(changes),
            "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beauty-csv", default="data/amazon_beauty_balanced.csv")
    parser.add_argument("--phone-source", default=PHONE_SOURCE)
    parser.add_argument("--output", default="artifacts/overall_rule_evaluation.json")
    args = parser.parse_args()
    beauty, y = load_data(args.beauty_csv, "review", "sentiment")
    _, beauty_rest, _, y_rest = train_test_split(beauty, y, test_size=.30,
                                                stratify=y, random_state=42)
    beauty_val, beauty_test, y_bv, y_bt = train_test_split(
        beauty_rest, y_rest, test_size=.50, stratify=y_rest, random_state=42)
    with closing(source_lines(args.phone_source)) as stream:
        phone_test, _, _ = sample_reviews(stream, 500, 100000, 42, exclude_texts=beauty)
        phone_development, _, _ = sample_reviews(
            stream, 3000, 300000, 43,
            exclude_texts=beauty + [text for text, _ in phone_test])
    phone_text = [text for text, _ in phone_development]
    phone_y = np.array([LABELS.index(label) for _, label in phone_development])
    _, phone_val, _, y_pv = train_test_split(phone_text, phone_y,
                                            test_size=.20, stratify=phone_y, random_state=43)
    engine = ReviewPredictor("artifacts")
    result = {"beauty_validation": compare(engine, beauty_val, y_bv),
              "phone_validation": compare(engine, phone_val, y_pv),
              "beauty_test": compare(engine, beauty_test, y_bt),
              "phone_test": compare(engine, [text for text, _ in phone_test],
                                    np.array([LABELS.index(label) for _, label in phone_test])),
              "limits": "Star-derived proxy labels. Phone test and beauty test have been inspected in earlier development. The rule uses aspect heuristics, not a new model."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in report.items() if key != "changes"}
                      for name, report in result.items() if isinstance(report, dict)}, indent=2))


if __name__ == "__main__":
    main()
