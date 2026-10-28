"""Experimental restaurant fine-tuning using unanimous SemEval aspect labels as proxies.

Usage: python train_restaurant_candidate.py --archive path/to/semeval14.zip
The 800 restaurant Phase B sentences have no polarity labels in this archive.
"""
import argparse
import copy
import csv
import json
import random
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from fine_tune_short_reviews import LABELS, predict, score
from models.model_bilstm import BiLSTM

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/short_review_candidate"
OUT = ROOT / "artifacts/restaurant_candidate"


def restaurant_examples(archive):
    with zipfile.ZipFile(archive) as handle:
        root = ET.fromstring(handle.read("Restaurants_Train_v2.xml"))
    examples = []
    counts = Counter()
    seen = set()
    for sentence in root.findall(".//sentence"):
        review = " ".join((sentence.findtext("text") or "").split())
        labels = [term.get("polarity") for term in sentence.findall(".//aspectTerm")]
        if not labels or any(label not in LABELS for label in labels):
            counts["no_usable_aspect_or_conflict"] += 1
            continue
        if len(set(labels)) != 1:
            counts["mixed_aspects_excluded"] += 1
            continue
        key = review.casefold()
        if not review or key in seen:
            counts["duplicate_or_empty"] += 1
            continue
        seen.add(key)
        examples.append({"review": review, "sentiment": labels[0]})
    return examples, dict(counts)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = False

    examples, excluded = restaurant_examples(args.archive)
    train, rest = train_test_split(examples, test_size=.30, random_state=45,
                                   stratify=[r["sentiment"] for r in examples])
    validation, test = train_test_split(rest, test_size=.50, random_state=45,
                                         stratify=[r["sentiment"] for r in rest])
    assert not ({r["review"].casefold() for r in train} &
                {r["review"].casefold() for r in validation + test})
    meta = json.loads((BASE / "metadata.json").read_text())
    tok = tokenizer_from_json((BASE / "tokenizer.json").read_text())
    checkpoint = torch.load(BASE / "bilstm.pt", map_location="cpu", weights_only=True)
    model = BiLSTM(checkpoint["vocab_size"], embed_dim=128, hidden_dim=128, num_layers=2)
    model.load_state_dict(checkpoint["state_dict"])

    def encode(rows):
        sequences = tok.texts_to_sequences([r["review"] for r in rows])
        return torch.tensor(pad_sequences(sequences, maxlen=meta["sequence_length"],
                                          padding="post", truncating="post"), dtype=torch.long)

    def labels(rows):
        return np.array([LABELS.index(r["sentiment"]) for r in rows])

    four = ROOT / "data/amazon_50k_multicategory"
    replay_source = read_csv(four / "train.csv")
    replay = [r for label in LABELS for r in random.sample(
        [r for r in replay_source if r["sentiment"] == label], 700)]
    training_rows = replay + train * 3
    xtrain = encode(training_rows)
    ytrain = torch.tensor(labels(training_rows), dtype=torch.long)
    sets = {"restaurant_validation": validation, "restaurant_test": test,
            "product_validation": read_csv(four / "validation.csv"),
            "product_test": read_csv(four / "test.csv")}
    xs = {name: encode(rows) for name, rows in sets.items()}
    ys = {name: labels(rows) for name, rows in sets.items()}
    baseline = {name: score(ys[name], predict(model, xs[name])) for name in sets}
    print("baseline", json.dumps(baseline), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lossfn = torch.nn.CrossEntropyLoss()
    history = []
    selected = None
    selected_epoch = None
    best = baseline["restaurant_validation"]["macro_f1"]
    start = time.perf_counter()
    for epoch in range(1, 4):
        model.train()
        for indices in torch.randperm(len(training_rows)).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = lossfn(model(xtrain[indices]), ytrain[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        restaurant = score(ys["restaurant_validation"], predict(model, xs["restaurant_validation"]))
        product = score(ys["product_validation"], predict(model, xs["product_validation"]))
        history.append({"epoch": epoch, "restaurant_validation": restaurant,
                        "product_validation": product})
        print("epoch", epoch, json.dumps(history[-1]), flush=True)
        if (restaurant["macro_f1"] > best and
            product["macro_f1"] >= baseline["product_validation"]["macro_f1"] - .025 and
            product["neutral_recall"] >= baseline["product_validation"]["neutral_recall"] - .05):
            best = restaurant["macro_f1"]
            selected = copy.deepcopy(model.state_dict())
            selected_epoch = epoch

    result = {"source": "SemEval 2014 Restaurants_Train_v2.xml, unanimous non-conflict aspect polarities used as sentence proxy labels",
              "limitations": "Aspect labels are not independently judged overall sentence labels. Mixed-aspect and unlabeled sentences are excluded. Restaurant Phase B has no polarity gold in this archive. Restaurant test is untouched by model selection, but the corpus has been inspected during aspect development.",
              "seed": 45, "excluded": excluded,
              "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
              "split_labels": {key: dict(Counter(r["sentiment"] for r in rows)) for key, rows in
                               (("train", train), ("validation", validation), ("test", test))},
              "baseline": baseline, "history": history, "selected_epoch": selected_epoch,
              "training_seconds": time.perf_counter() - start}
    OUT.mkdir(parents=True, exist_ok=True)
    if selected is not None:
        model.load_state_dict(selected)
        result["candidate"] = {name: score(ys[name], predict(model, xs[name])) for name in sets}
        torch.save({**checkpoint, "state_dict": selected, "epoch": selected_epoch}, OUT / "bilstm.pt")
        (OUT / "tokenizer.json").write_text((BASE / "tokenizer.json").read_text())
        (OUT / "metadata.json").write_text(json.dumps({**meta, "fine_tuned_from": str(BASE.relative_to(ROOT)),
                                                       "domain": "restaurants", "label_type": "unanimous aspect polarity proxy"}, indent=2) + "\n")
        print("candidate", json.dumps(result["candidate"]), flush=True)
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("selected", selected_epoch, "seconds", result["training_seconds"], flush=True)


if __name__ == "__main__":
    main()
