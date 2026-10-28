"""Compare old and expanded keyword coverage on SemEval laptop annotations."""

import argparse
import json
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from aspect_rules import explicit_battery_polarity
from aspect_polarity import explicit_aspect_polarity
from review_intelligence import LABELS, ReviewPredictor, aspect_clauses, aspect_matches
from train_aspect import SOURCE, load_examples


# The original shipped dictionary is frozen for a reproducible comparison.
BASELINE = {
    "Camera": ["camera", "photo", "picture", "lens", "selfie"],
    "Battery": ["battery", "charging", "charge", "power"],
    "Design": ["design", "look", "appearance", "style", "build"],
    "Performance": ["performance", "speed", "fast", "slow", "lag", "processor"],
    "Price": ["price", "cost", "expensive", "cheap", "value"],
    "Customer support": ["support", "service", "customer care", "helpdesk"],
    "Sound": ["sound", "audio", "speaker", "headphone", "bass"],
    "Screen": ["screen", "display", "brightness", "resolution"],
    "Delivery": ["delivery", "shipping", "arrived", "packaging"],
    "Quality": ["quality", "durable", "durability", "material"],
}


def evaluate(sentences, keywords):
    gold_total = matched = predicted_total = 0
    labels = Counter()
    for sentence in sentences:
        text = sentence["text"]
        spans = [(term["start"], term["end"]) for term in sentence["terms"]
                 if term["start"] is not None and term["end"] is not None]
        matches = aspect_matches(text, keywords)
        gold_total += len(spans)
        predicted_total += len(matches)
        # One predicted span can credit at most one gold term, and vice versa.
        assigned = {}

        def assign(prediction, seen):
            _, start, end = matches[prediction]
            for gold, (a, b) in enumerate(spans):
                if gold in seen or start >= b or end <= a:
                    continue
                seen.add(gold)
                if gold not in assigned or assign(assigned[gold], seen):
                    assigned[gold] = prediction
                    return True
            return False

        for prediction in range(len(matches)):
            assign(prediction, set())
        matched += len(assigned)
        labels.update(matches[index][0] for index in assigned.values())
    precision = matched / predicted_total if predicted_total else 0
    recall = matched / gold_total if gold_total else 0
    return {"gold_terms": gold_total, "gold_terms_found": matched,
            "predicted_mentions": predicted_total, "predicted_mentions_matched": matched,
            "mention_precision": precision, "gold_term_recall": recall,
            "mention_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "matched_mentions_by_aspect": dict(labels.most_common())}


def evaluate_polarity(sentences, predictor, keywords, use_cues=False):
    """Score clause polarity where a predicted term overlaps a gold term."""
    jobs = []
    gold_total = 0
    for sentence in sentences:
        text = sentence["text"]
        matches = aspect_matches(text, keywords)
        clauses = aspect_clauses(text, keywords)
        locations = []
        cursor = 0
        for clause in clauses:
            start = text.lower().find(clause.lower(), cursor)
            if start >= 0:
                locations.append((start, start + len(clause), clause))
                cursor = start + len(clause)
        for term in sentence["terms"]:
            gold_total += 1
            start, end = term["start"], term["end"]
            hit = next(((aspect, a, b) for aspect, a, b in matches if a < end and b > start), None)
            clause = next((value for a, b, value in locations if a <= start and end <= b), None)
            if hit and clause:
                jobs.append((term["label"], hit[0], clause))
    probabilities = predictor.probabilities([clause for _, _, clause in jobs])
    truth, predicted = [], []
    cue_count = cue_correct = 0
    for (label, aspect, clause), probability in zip(jobs, probabilities):
        truth.append(LABELS.index(label))
        override = (explicit_aspect_polarity(clause, aspect, keywords) if use_cues else
                    explicit_battery_polarity(clause) if aspect == "Battery" else None)
        if use_cues and override:
            cue_count += 1
            cue_correct += override == label
        predicted.append(LABELS.index(override) if override else int(np.argmax(probability)))
    correct = sum(a == b for a, b in zip(truth, predicted))
    return {"gold_terms": gold_total, "covered_with_clause": len(jobs),
            "correct_polarity_among_covered": correct,
            "explicit_cue_uses": cue_count, "explicit_cue_correct": cue_correct,
            "covered_accuracy": correct / len(jobs) if jobs else 0,
            "covered_macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2],
                                                 average="macro", zero_division=0)) if jobs else 0,
            "correct_detected_polarity_per_gold_term": correct / gold_total if gold_total else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Local SemEval archive.zip; download if omitted")
    parser.add_argument("--output", default="artifacts/aspect_extraction_evaluation.json")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        archive = args.archive or str(Path(directory) / "semeval14.zip")
        if not args.archive:
            urllib.request.urlretrieve(SOURCE, archive)
        sentences = load_examples(archive)
    train, rest = train_test_split(sentences, test_size=.30, random_state=42)
    validation, test = train_test_split(rest, test_size=.50, random_state=42)
    predictor = ReviewPredictor(args.artifact_dir)
    output = {"source": "SemEval 2014 Task 4 Laptop_Train_v2.xml",
              "split": "70/15/15 by sentence, seed 42; conflict polarity excluded",
              "training_sentences_used_for_keyword_selection": len(train),
              "validation": {"baseline": evaluate(validation, BASELINE),
                             "expanded": evaluate(validation, None)},
              "test": {"baseline": evaluate(test, BASELINE),
                       "expanded": evaluate(test, None)},
              "polarity_validation": {"baseline": evaluate_polarity(validation, predictor, BASELINE),
                                      "expanded": evaluate_polarity(validation, predictor, None),
                                      "expanded_with_cues": evaluate_polarity(validation, predictor, None, True)},
              "polarity_test": {"baseline": evaluate_polarity(test, predictor, BASELINE),
                                "expanded": evaluate_polarity(test, predictor, None),
                                "expanded_with_cues": evaluate_polarity(test, predictor, None, True)},
              "limits": "One-to-one term span overlap measures detection, not semantic category correctness. Polarity compares the beauty-trained clause model plus battery phrases with a conservative general cue rule. Uncovered terms are not scored as having a sentiment. The laptop test had previously been inspected."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
