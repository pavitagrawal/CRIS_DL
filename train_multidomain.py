"""Exploratory BiLSTM retraining with disjoint phone review source ranges."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from evaluate_new_category import PHONE_SOURCE, sample_reviews
from prepare_dataset import source_lines
from review_intelligence import LABELS, ReviewPredictor
from train import build_model, load_data, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beauty-csv", default="data/amazon_beauty_balanced.csv")
    parser.add_argument("--phone-source", default=PHONE_SOURCE)
    parser.add_argument("--output", default="artifacts/multidomain_candidate")
    parser.add_argument("--baseline-dir", default="artifacts/beauty_baseline")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.keras.utils.set_random_seed(42)
    beauty_text, beauty_y = load_data(args.beauty_csv, "review", "sentiment")
    beauty_train, beauty_rest, y_beauty_train, y_beauty_rest = train_test_split(
        beauty_text, beauty_y, test_size=.30, stratify=beauty_y, random_state=42)
    beauty_val, beauty_test, y_beauty_val, y_beauty_test = train_test_split(
        beauty_rest, y_beauty_rest, test_size=.50, stratify=y_beauty_rest, random_state=42)
    print("Sampling separate phone source ranges...", flush=True)
    with closing(source_lines(args.phone_source)) as stream:
        phone_test, n_test, test_available = sample_reviews(
            stream, per_class=500, scan_limit=100000, seed=42, exclude_texts=beauty_text)
        phone_all, n_development, dev_available = sample_reviews(
            stream, per_class=3000, scan_limit=300000, seed=43,
            exclude_texts=beauty_text + [text for text, _ in phone_test])
    phone_text = [text for text, _ in phone_all]
    phone_y = np.array([LABELS.index(label) for _, label in phone_all])
    phone_train, phone_val, y_phone_train, y_phone_val = train_test_split(
        phone_text, phone_y, test_size=.20, stratify=phone_y, random_state=43)
    train_text = beauty_train + phone_train
    y_train = np.concatenate((y_beauty_train, y_phone_train))
    val_text = beauty_val + phone_val
    y_val = np.concatenate((y_beauty_val, y_phone_val))

    tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)

    def encode(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=120,
                             padding="post", truncating="post")

    model = build_model("BiLSTM", 15000, 120)
    model.fit(encode(train_text), y_train, validation_data=(encode(val_text), y_val),
              epochs=args.epochs, batch_size=128, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                            restore_best_weights=True)])
    baseline_dir = args.baseline_dir if Path(args.baseline_dir, "best_model.keras").exists() else "artifacts"
    old = ReviewPredictor(baseline_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    comparison = {}
    for name, texts, labels in (("beauty_validation", beauty_val, y_beauty_val),
                                 ("phone_validation", phone_val, y_phone_val),
                                 ("beauty_test", beauty_test, y_beauty_test),
                                 ("phone_test", [text for text, _ in phone_test],
                                  np.array([LABELS.index(label) for _, label in phone_test]))):
        before = old.probabilities(texts).argmax(axis=1)
        after = model.predict(encode(texts), batch_size=256, verbose=0).argmax(axis=1)
        comparison[name] = {"count": len(labels), "current": scores(labels, before),
                            "candidate": scores(labels, after),
                            "current_confusion": confusion_matrix(labels, before, labels=[0, 1, 2]).tolist(),
                            "candidate_confusion": confusion_matrix(labels, after, labels=[0, 1, 2]).tolist()}
    examples = ["The screen is bright and performance is fast.",
                "The keyboard is comfortable but the battery drains quickly."]
    comparison["examples"] = [{"text": text,
                                "current": LABELS[int(old.probabilities([text]).argmax())],
                                "candidate": LABELS[int(model.predict(encode([text]), verbose=0).argmax())]}
                               for text in examples]
    comparison["design"] = {
        "source": args.phone_source, "labels": "1-2 negative; 3 neutral; 4-5 positive",
        "phone_test_source_rows": "first 100,000 source rows",
        "phone_development_source_rows": "next 300,000 source rows",
        "scanned_test": n_test, "scanned_development": n_development,
        "usable_in_test_range": test_available, "usable_in_development_range": dev_available,
        "training_counts": {"beauty": len(beauty_train), "phone": len(phone_train)},
        "validation_counts": {"beauty": len(beauty_val), "phone": len(phone_val)},
        "test_counts": {"beauty": len(beauty_test), "phone": len(phone_test)},
        "seed": 42, "phone_development_seed": 43,
        "baseline_artifact_dir": baseline_dir,
        "notes": "No exact normalized beauty or phone test review text in phone development sample; source ranges are separate, but related products and near-duplicates may remain. Phone test set has already been inspected in earlier experiments."}
    model.save(out / "best_model.keras")
    (out / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({"best_model": "BiLSTM", "labels": LABELS,
                                                   "max_length": 120, "candidate": True}, indent=2), encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in result.items() if k in ("current", "candidate")}
                      for name, result in comparison.items() if isinstance(result, dict)}, indent=2))


if __name__ == "__main__":
    main()
