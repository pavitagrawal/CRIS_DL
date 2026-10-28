"""Train and compare four models on the same stratified data split."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from review_intelligence import LABELS, clean_text


def load_data(path, text_column, label_column, rating_labels=False):
    df = pd.read_csv(path)
    missing = [column for column in (text_column, label_column) if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
    df = df[[text_column, label_column]].dropna().copy()
    df[text_column] = df[text_column].map(clean_text)
    df = df[df[text_column].str.len() > 0]
    if rating_labels:
        ratings = pd.to_numeric(df[label_column], errors="coerce")
        valid = ratings.isin([1, 2, 3, 4, 5])
        df = df.loc[valid].copy()
        df[label_column] = ratings.loc[valid].map({1: "negative", 2: "negative",
                                                    3: "neutral", 4: "positive", 5: "positive"})
    else:
        df[label_column] = df[label_column].astype(str).str.lower().str.strip()
        unknown = sorted(set(df[label_column]) - set(LABELS))
        if unknown:
            raise ValueError(f"Unsupported labels: {unknown}. Use negative, neutral, positive, or --rating-labels.")
    counts = df[label_column].value_counts()
    if set(counts.index) != set(LABELS) or min(counts) < 10:
        raise ValueError("Provide at least 10 nonempty examples for EACH of negative, neutral and positive; more are recommended.")
    return df[text_column].tolist(), df[label_column].map({label: i for i, label in enumerate(LABELS)}).to_numpy()


def build_model(kind, vocab_size, max_length, embedding_dim=128):
    from tensorflow import keras
    layers = keras.layers
    recurrent = {
        "RNN": layers.SimpleRNN(64),
        "LSTM": layers.LSTM(64),
        "GRU": layers.GRU(64),
        "BiLSTM": layers.Bidirectional(layers.LSTM(64)),
    }[kind]
    model = keras.Sequential([
        layers.Input(shape=(max_length,)),
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        recurrent,
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax"),
    ], name=kind)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def scores(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(precision),
            "recall_macro": float(recall), "f1_macro": float(f1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Labeled CSV for training")
    parser.add_argument("--text-column", default="review")
    parser.add_argument("--label-column", default="sentiment")
    parser.add_argument("--rating-labels", action="store_true", help="Map rating 1-2/3/4-5 to negative/neutral/positive")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=160)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_words < 100 or args.max_length < 1:
        parser.error("Epochs, batch size and max length must be positive; max words must be at least 100.")

    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.keras.utils.set_random_seed(42)
    texts, y = load_data(args.data, args.text_column, args.label_column, args.rating_labels)
    train_text, rest_text, y_train, y_rest = train_test_split(
        texts, y, test_size=0.30, stratify=y, random_state=42)
    val_text, test_text, y_val, y_test = train_test_split(
        rest_text, y_rest, test_size=0.50, stratify=y_rest, random_state=42)
    tokenizer = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)  # Prevent vocabulary leakage from validation/test.

    def encode(items):
        return pad_sequences(tokenizer.texts_to_sequences(items), maxlen=args.max_length,
                             padding="post", truncating="post")

    x_train, x_val, x_test = map(encode, (train_text, val_text, test_text))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for kind in ("RNN", "LSTM", "GRU", "BiLSTM"):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_model(kind, args.max_words, args.max_length)
        start = time.perf_counter()
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
                  batch_size=args.batch_size, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                              restore_best_weights=True)])
        elapsed = time.perf_counter() - start
        val_pred = model.predict(x_val, verbose=0).argmax(axis=1)
        metrics = scores(y_val, val_pred)
        results.append({"model": kind, **{f"validation_{k}": v for k, v in metrics.items()},
                        "training_seconds": round(elapsed, 2)})
        model.save(out / f"{kind}.keras")
        print(kind, metrics)

    comparison = pd.DataFrame(results).sort_values("validation_f1_macro", ascending=False)
    winner = str(comparison.iloc[0]["model"])
    best = tf.keras.models.load_model(out / f"{winner}.keras")
    test_pred = best.predict(x_test, verbose=0).argmax(axis=1)
    test_metrics = scores(y_test, test_pred)
    matrix = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(out / "test_confusion_matrix.csv")
    comparison.to_csv(out / "model_comparison.csv", index=False)
    best.save(out / "best_model.keras")
    (out / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    metadata = {"best_model": winner, "labels": LABELS, "max_length": args.max_length,
                "selection_metric": "validation macro F1", "test_metrics": test_metrics,
                "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best model: {winner}; held-out test metrics: {test_metrics}; saved to {out}")


if __name__ == "__main__":
    main()
