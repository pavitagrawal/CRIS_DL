"""Train an aspect-conditioned polarity classifier on SemEval 2014 laptop reviews."""

import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from review_intelligence import LABELS, aspects_in

SOURCE = "https://zenodo.org/records/14655773/files/archive.zip?download=1"
XML_NAME = "Laptop_Train_v2.xml"


def marked_text(text, target, start=None, end=None):
    """Give the classifier both the target and its position in the sentence."""
    if start is None or end is None or text[start:end].lower() != target.lower():
        start = text.lower().find(target.lower())
        if start < 0:
            return f"target {target} context {text}"
        end = start + len(target)
    return f"target {target} context {text[:start]} aspectstart {text[start:end]} aspectend {text[end:]}"


def load_examples(archive):
    with zipfile.ZipFile(archive) as file:
        root = ET.fromstring(file.read(XML_NAME))
    sentences = []
    for element in root.findall(".//sentence"):
        text = element.findtext("text") or ""
        terms = []
        for term in element.findall(".//aspectTerm"):
            label = term.get("polarity")
            if label not in LABELS:
                continue  # SemEval conflict is outside our three-class task.
            name = term.get("term", "")
            if not name:
                continue
            try:
                start, end = int(term.get("from")), int(term.get("to"))
            except (TypeError, ValueError):
                start = end = None
            terms.append({"target": name, "label": label, "start": start, "end": end})
        if terms:
            sentences.append({"text": text, "terms": terms})
    return sentences


def flatten(sentences):
    return [(marked_text(s["text"], t["target"], t["start"], t["end"]),
             LABELS.index(t["label"]), s["text"], t["target"])
            for s in sentences for t in s["terms"]]


def metrics(y, predicted):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0))}


def model(vocab_size, max_length):
    from tensorflow import keras
    layers = keras.layers
    network = keras.Sequential([
        layers.Input(shape=(max_length,)),
        layers.Embedding(vocab_size, 96, mask_zero=True),
        layers.Bidirectional(layers.LSTM(48)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(.35),
        layers.Dense(3, activation="softmax"),
    ])
    network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Local SemEval archive.zip; download if omitted")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--max-words", type=int, default=12000)
    args = parser.parse_args()
    import tempfile
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.keras.utils.set_random_seed(42)
    if args.archive:
        archive = args.archive
    else:
        temporary = tempfile.TemporaryDirectory()
        archive = str(Path(temporary.name) / "semeval14.zip")
        print("Downloading SemEval 2014 laptop annotations...", flush=True)
        urllib.request.urlretrieve(SOURCE, archive)
    sentences = load_examples(archive)
    # Split sentences BEFORE expanding to term rows so a sentence cannot cross splits.
    train_s, rest_s = train_test_split(sentences, test_size=.30, random_state=42)
    val_s, test_s = train_test_split(rest_s, test_size=.50, random_state=42)
    train, val, test = map(flatten, (train_s, val_s, test_s))
    tok = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tok.fit_on_texts([r[0] for r in train])

    def encode(rows):
        return pad_sequences(tok.texts_to_sequences([r[0] for r in rows]),
                             maxlen=args.max_length, padding="post", truncating="post")

    x_train, x_val, x_test = map(encode, (train, val, test))
    y_train, y_val, y_test = [np.array([r[1] for r in rows]) for rows in (train, val, test)]
    network = model(args.max_words, args.max_length)
    network.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
                batch_size=32, verbose=2,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                              restore_best_weights=True)])
    validation = metrics(y_val, network.predict(x_val, verbose=0).argmax(axis=1))
    testing = metrics(y_test, network.predict(x_test, verbose=0).argmax(axis=1))
    covered = [i for i, row in enumerate(test) if aspects_in(row[3])]
    covered_metrics = metrics(y_test[covered], network.predict(x_test[covered], verbose=0).argmax(axis=1))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    network.save(out / "aspect_model.keras")
    (out / "aspect_tokenizer.json").write_text(tok.to_json(), encoding="utf-8")
    details = {"source": "SemEval 2014 Task 4 laptop training annotations", "labels": LABELS,
               "split": "70/15/15 by sentence; seed 42", "counts": {"train_terms": len(train),
               "validation_terms": len(val), "test_terms": len(test), "covered_test_terms": len(covered)},
               "validation": validation, "test_all_gold_targets": testing,
               "test_dictionary_covered_targets": covered_metrics,
               "max_length": args.max_length, "max_words": args.max_words,
               "note": "Polarity metrics assume a known gold aspect term; extraction must be measured separately."}
    (out / "aspect_metadata.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
