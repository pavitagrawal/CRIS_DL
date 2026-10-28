import tempfile
import unittest
from pathlib import Path

import pandas as pd

import numpy as np

from review_intelligence import (RESTAURANT_ASPECT_KEYWORDS, ReviewPredictor,
                                 aspect_clauses, aspects_in, aspect_matches, aspect_summary)
from review_intelligence import suggested_aspect_profile, clear_restaurant_sentiment
from train import load_data, scores
from prepare_dataset import prepare
from train_aspect import marked_text
from aspect_rules import explicit_battery_polarity
from aspect_polarity import explicit_aspect_polarity
from evaluate_new_category import sample_reviews
from evaluate_aspect_extraction import evaluate
from overall_rule import agreed_positive_aspects
from review_queue import apply_human_labels, review_reasons
from human_evaluation import draw_sample, evaluate_sample, labeled_sample


class LogicTests(unittest.TestCase):
    def test_narrow_restaurant_cue(self):
        self.assertEqual(clear_restaurant_sentiment("The food was tasty"), "positive")
        self.assertEqual(clear_restaurant_sentiment("The food was not tasty"), "negative")
        self.assertIsNone(clear_restaurant_sentiment("The food was tasty but the service was terrible"))
        self.assertIsNone(clear_restaurant_sentiment("The camera was great"))
    def test_food_category_hint_and_explicit_cue(self):
        self.assertEqual(suggested_aspect_profile("The food was delicious", "Products and electronics"),
                         "Restaurants")
        self.assertIsNone(suggested_aspect_profile("The service was slow", "Products and electronics"))
        self.assertEqual(explicit_aspect_polarity("The food was delicious", "Food", RESTAURANT_ASPECT_KEYWORDS),
                         "positive")
        self.assertEqual(explicit_aspect_polarity("The food was not delicious", "Food", RESTAURANT_ASPECT_KEYWORDS),
                         "negative")
    def test_mixed_review_separates_aspects(self):
        clauses = aspect_clauses("The camera is great but the battery is bad.")
        self.assertEqual([aspects_in(c) for c in clauses], [["Camera"], ["Battery"]])

    def test_restaurant_profile_uses_restaurant_labels(self):
        review = "The food is great but the service is bad."
        clauses = aspect_clauses(review, RESTAURANT_ASPECT_KEYWORDS)
        self.assertEqual([aspects_in(c, RESTAURANT_ASPECT_KEYWORDS) for c in clauses],
                         [["Food"], ["Service"]])
        self.assertEqual(aspects_in("The staff was helpful", RESTAURANT_ASPECT_KEYWORDS), ["Service"])

    def test_separate_complete_aspect_clauses_joined_by_and(self):
        clauses = aspect_clauses("The screen is bright and performance is fast.")
        self.assertEqual(clauses, ["The screen is bright", "performance is fast."])
        self.assertEqual(aspect_clauses("The camera and battery are great."),
                         ["The camera and battery are great."])
        self.assertEqual(aspect_clauses("The screen is large and performance is unknown."),
                         ["The screen is large and performance is unknown."])

    def test_word_boundaries(self):
        self.assertEqual(aspects_in("This is a showcase"), [])

    def test_new_aspects_and_longest_term(self):
        self.assertEqual(aspects_in("The keyboard and hard drive are great"), ["Keyboard", "Storage"])
        self.assertEqual(aspects_in("This is a keyboardless case"), [])
        self.assertEqual(len(aspect_matches("Customer service was slow", {"Support": ["service", "customer service"]})), 1)

    def test_one_gold_term_cannot_credit_two_matches(self):
        sentences = [{"text": "screen size", "terms": [{"start": 0, "end": 11}]}]
        result = evaluate(sentences, {"Screen": ["screen"], "Size": ["size"]})
        self.assertEqual(result["gold_terms_found"], 1)
        self.assertEqual(result["predicted_mentions"], 2)
        self.assertEqual(result["mention_precision"], 0.5)

    def test_rating_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"text": [f"Review {i}" for i in range(60)],
                          "stars": [1, 3, 5] * 20}).to_csv(path, index=False)
            _, labels = load_data(path, "text", "stars", True)
            self.assertEqual(list(labels[:3]), [0, 1, 2])

    def test_macro_metrics(self):
        result = scores([0, 1, 2], [0, 1, 2])
        self.assertEqual(result["f1_macro"], 1.0)

    def test_public_data_preparation(self):
        import json
        source = (json.dumps({"text": f"Unique review number {i} is long enough", "rating": i % 5 + 1,
                              "parent_asin": f"P{i}"}).encode() for i in range(75))
        rows, report = prepare(source, per_class=3, seed=42)
        self.assertEqual(report["sampled_by_class"], {"negative": 3, "neutral": 3, "positive": 3})
        self.assertEqual(len(rows), 9)
        self.assertEqual({r["sentiment"] for r in rows}, {"negative", "neutral", "positive"})

    def test_aspect_marker_uses_specific_occurrence(self):
        text = "The screen looks good but the screen is dim"
        start = text.rfind("screen")
        marked = marked_text(text, "screen", start, start + len("screen"))
        self.assertIn("good but the  aspectstart screen aspectend  is dim", marked)

    def test_explicit_battery_phrases_and_abstention(self):
        examples = {
            "the battery drains quickly": "negative",
            "the battery doesn't last long": "negative",
            "short battery life": "negative",
            "the battery lasts all day": "positive",
            "great battery life": "positive",
            "the battery is not good": "negative",
            "I removed the battery to reset it": None,
            "the battery is not bad": None,
            "the battery doesn't drain quickly": None,
            "the camera is excellent": None,
        }
        for clause, expected in examples.items():
            with self.subTest(clause=clause):
                self.assertEqual(explicit_battery_polarity(clause), expected)

    def test_explicit_aspect_cues_abstain_on_ambiguity(self):
        cases = {
            ("The keyboard is comfortable", "Keyboard"): "positive",
            ("The keyboard is uncomfortable", "Keyboard"): "negative",
            ("Customer support was unhelpful", "Customer support"): "negative",
            ("The keyboard is not good", "Keyboard"): "negative",
            ("The keyboard is not bad", "Keyboard"): None,
            ("The keyboard is good and bad", "Keyboard"): None,
            ("The keyboard is comfortable but the screen is dim", "Keyboard"): None,
            ("The keyboard has no backlight", "Keyboard"): None,
        }
        for (clause, aspect), expected in cases.items():
            with self.subTest(clause=clause):
                self.assertEqual(explicit_aspect_polarity(clause, aspect), expected)

    def test_appliance_noise_aspect_cues(self):
        cases = {
            "The mixer makes a very loud noise": "negative",
            "This dryer is noisy": "negative",
            "This dryer is quiet": "positive",
            "The fan makes no noise": "positive",
            "The fan is not too noisy": "positive",
            "The noise level is listed on the box": None,
            "The screen is bright and the mixer makes a loud noise": None,
        }
        for clause, expected in cases.items():
            with self.subTest(clause=clause):
                self.assertEqual(explicit_aspect_polarity(clause, "Sound"), expected)

    def test_cross_category_sampler_excludes_training_text(self):
        import json
        lines = (json.dumps({"text": f"Phone review number {i} has enough words",
                             "rating": i % 5 + 1}).encode() for i in range(60))
        rows, scanned, _ = sample_reviews(lines, per_class=3, scan_limit=60,
                                          exclude_texts=["Phone review number 0 has enough words"])
        self.assertEqual(scanned, 60)
        self.assertEqual(len(rows), 9)
        self.assertNotIn("Phone review number 0 has enough words", [r[0] for r in rows])

    def test_agreed_aspects_rule_is_narrow(self):
        text = "The screen is bright and performance is fast."
        items = [{"aspect": "Screen", "sentiment": "positive", "method": "explicit cue",
                  "evidence": "The screen is bright"},
                 {"aspect": "Performance", "sentiment": "positive", "method": "explicit cue",
                  "evidence": "performance is fast."}]
        self.assertEqual(agreed_positive_aspects(text, "neutral", items), "positive")
        self.assertIsNone(agreed_positive_aspects(text, "negative", items))
        self.assertIsNone(agreed_positive_aspects("The screen is bright but performance is fast.",
                                                  "neutral", items))

    def test_aspect_summary_keeps_mixed_opinions_separate(self):
        self.assertEqual(aspect_summary([{"sentiment": "positive"},
                                         {"sentiment": "positive"}]), "Positive")
        self.assertEqual(aspect_summary([{"sentiment": "positive"},
                                         {"sentiment": "negative"}]), "Mixed")
        self.assertEqual(aspect_summary([]), "No aspects found")

    def test_batch_aspect_attribution(self):
        engine = ReviewPredictor.__new__(ReviewPredictor)
        engine.labels = ["negative", "neutral", "positive"]
        calls = []

        def fake_probabilities(texts, batch_size=256):
            calls.append(list(texts))
            return np.array([[0.8, 0.1, 0.1] if "battery" in t.lower()
                             else [0.1, 0.1, 0.8] for t in texts])

        engine.probabilities = fake_probabilities
        result = engine.analyze(["Camera great but battery bad", "Screen bright"])
        self.assertEqual([a["sentiment"] for a in result[0]["aspects"]], ["positive", "negative"])
        self.assertEqual(result[1]["aspects"][0]["aspect"], "Screen")
        self.assertEqual(len(calls), 2)  # A review batch and an aspect-clause batch.

    def test_restaurant_profile_inference(self):
        engine = ReviewPredictor.__new__(ReviewPredictor)
        engine.labels = ["negative", "neutral", "positive"]
        engine.probabilities = lambda texts, batch_size=256: np.array(
            [[.8, .1, .1] if "bad" in text else [.1, .1, .8] for text in texts])
        result = engine.analyze(["The food is great but the service is bad."],
                                aspect_profile="Restaurants")[0]
        self.assertEqual([(a["aspect"], a["sentiment"]) for a in result["aspects"]],
                         [("Food", "positive"), ("Service", "negative")])

    def test_review_queue_flags_disagreement_and_low_score(self):
        example = {"sentiment": "neutral", "confidence": .5955,
                   "aspect_summary": "Positive"}
        self.assertEqual(review_reasons(example),
                         ["Low model score", "Overall/aspect disagreement"])
        self.assertEqual(review_reasons({"sentiment": "positive", "confidence": .9,
                                         "aspect_summary": "Positive"}), [])
        self.assertEqual(review_reasons({"sentiment": "neutral", "confidence": .9,
                                         "aspect_summary": "Mixed"}), ["Mixed aspect signals"])
        confident_neutral = {"sentiment": "neutral", "confidence": .8,
                             "aspect_summary": "No aspects found"}
        self.assertEqual(review_reasons(confident_neutral, "I hate it. The button broke."),
                         ["Neutral model label with explicit complaint"])
        self.assertEqual(review_reasons(confident_neutral, "I will not be reordering."),
                         ["Neutral model label with explicit complaint"])
        self.assertEqual(review_reasons(confident_neutral, "I dislike that bottle."),
                         ["Neutral model label with explicit complaint"])
        self.assertEqual(review_reasons(confident_neutral, "It works fine."), [])

    def test_human_correction_preserves_model_prediction(self):
        rows = pd.DataFrame({"review_row": [1, 2],
                             "predicted_sentiment": ["neutral", "negative"]})
        corrected = apply_human_labels(rows, {1: "positive"})
        self.assertEqual(corrected["predicted_sentiment"].tolist(), ["neutral", "negative"])
        self.assertEqual(corrected["final_sentiment"].tolist(), ["positive", "negative"])
        self.assertEqual(corrected["label_source"].tolist(), ["human", "model"])

    def test_random_evaluation_includes_unflagged_errors(self):
        rows = pd.DataFrame({"review_row": [1, 2, 3, 4],
                             "predicted_sentiment": ["neutral", "positive", "positive", "negative"],
                             "review_status": ["Needs review", "Model result", "Model result", "Needs review"]})
        sampled = draw_sample(rows, seed=4)
        self.assertEqual(set(sampled["review_row"]), {1, 2, 3, 4})
        partial = labeled_sample(sampled, {1: "positive"})
        with self.assertRaises(ValueError):
            evaluate_sample(partial)
        complete = labeled_sample(sampled, {1: "positive", 2: "positive",
                                            3: "negative", 4: "negative"})
        report = evaluate_sample(complete)
        self.assertEqual(report["errors"], 2)
        self.assertEqual(report["flagged_errors"], 1)
        self.assertEqual(report["unflagged_errors"], 1)
        self.assertEqual(report["accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
