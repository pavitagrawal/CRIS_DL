"""Fine-tune the 50k BiLSTM on disjoint UCI sentences with replay data.

The 200-review upload is excluded from training, validation, and fresh test.
Selection uses UCI validation accuracy with a four-category validation guard.
"""
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from models.model_bilstm import BiLSTM


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/submitted_50k_comparison"
OUT = ROOT / "artifacts/short_review_candidate"
LABELS = ["negative", "neutral", "positive"]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def score(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=range(3), average="macro", zero_division=0)),
            "neutral_recall": float(recall_score(y, pred, labels=[1], average="macro", zero_division=0)),
            "confusion": confusion_matrix(y, pred, labels=range(3)).tolist()}


def predict(model, x, batch_size=128):
    model.eval()
    out = []
    with torch.inference_mode():
        for part in x.split(batch_size):
            out.append(model(part).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = False
    info = json.loads((BASE / "metadata.json").read_text())
    tokenizer = tokenizer_from_json((BASE / "tokenizer.json").read_text())
    checkpoint = torch.load(BASE / "bilstm.pt", map_location="cpu", weights_only=True)
    assert checkpoint["labels"] == LABELS
    model = BiLSTM(checkpoint["vocab_size"], embed_dim=128, hidden_dim=128, num_layers=2)
    model.load_state_dict(checkpoint["state_dict"])
    def encode(rows):
        seq = tokenizer.texts_to_sequences([r["review"] for r in rows])
        return torch.tensor(pad_sequences(seq, maxlen=info["sequence_length"], padding="post", truncating="post"), dtype=torch.long)
    def labels(rows):
        return np.array([LABELS.index(r.get("sentiment", r.get("human_sentiment"))) for r in rows])

    short = ROOT / "data/uci_short_split"
    train_short, val_short, test_short = [read(short / f"{name}.csv") for name in ("train", "validation", "test")]
    selected = read(ROOT / "data/uci_amazon_200_unseen_reviews.csv")
    selected_keys = {r["review"].strip().casefold() for r in selected}
    assert len(selected_keys) == len(selected) == 200
    assert not selected_keys.intersection(r["review"].strip().casefold() for r in train_short + val_short + test_short)
    human = read(ROOT / "data/human_evaluation_sample.csv")
    assert all(r["review"].strip().casefold() in selected_keys for r in human)

    four = ROOT / "data/amazon_50k_multicategory"
    replay_source = read(four / "train.csv")
    replay = []
    for name in LABELS:
        group = [r for r in replay_source if r["sentiment"] == name]
        replay.extend(random.sample(group, 1200))
    train = replay + train_short * 3
    four_val, four_test = read(four / "validation.csv"), read(four / "test.csv")
    xtrain, ytrain = encode(train), torch.tensor(labels(train), dtype=torch.long)
    eval_sets = {"short_validation": val_short, "short_test": test_short,
                 "short_200": selected,
                 "four_validation": four_val, "four_test": four_test}
    xs = {name: encode(rows) for name, rows in eval_sets.items()}
    ys = {"short_validation": labels(val_short), "short_test": labels(test_short),
          "four_validation": labels(four_val), "four_test": labels(four_test)}
    human_x = encode(human); human_y = labels(human)
    baseline = {name: score(ys[name], predict(model, xs[name])) for name in ys}
    baseline["human_50"] = score(human_y, predict(model, human_x))
    print("baseline", json.dumps(baseline), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lossfn = torch.nn.CrossEntropyLoss()
    history = []; selected_state = None; chosen_epoch = None; best_short = baseline["short_validation"]["accuracy"]
    started = time.perf_counter()
    for epoch in range(1, 4):
        model.train()
        for indices in torch.randperm(len(xtrain)).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = lossfn(model(xtrain[indices]), ytrain[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        short_val = score(ys["short_validation"], predict(model, xs["short_validation"]))
        four_val_score = score(ys["four_validation"], predict(model, xs["four_validation"]))
        row = {"epoch": epoch, "short_validation": short_val, "four_validation": four_val_score}
        history.append(row)
        print("epoch", json.dumps(row), flush=True)
        guard = (four_val_score["macro_f1"] >= baseline["four_validation"]["macro_f1"] - .02
                 and four_val_score["neutral_recall"] >= baseline["four_validation"]["neutral_recall"] - .04)
        if guard and short_val["accuracy"] > best_short:
            best_short = short_val["accuracy"]
            chosen_epoch = epoch
            selected_state = copy.deepcopy(model.state_dict())
    elapsed = time.perf_counter() - started
    result = {"protocol": "UCI 587 training/100 validation/100 fresh test; 200 excluded; four-category replay 1200 per class; three epochs",
              "baseline": baseline, "history": history, "selected_epoch": chosen_epoch,
              "train_seconds": elapsed}
    if selected_state is not None:
        model.load_state_dict(selected_state)
        candidate = {name: score(ys[name], predict(model, xs[name])) for name in ys}
        candidate["human_50"] = score(human_y, predict(model, human_x))
        result["candidate"] = candidate
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({**checkpoint, "state_dict": selected_state, "epoch": chosen_epoch}, OUT / "bilstm.pt")
        (OUT / "tokenizer.json").write_text((BASE / "tokenizer.json").read_text())
        (OUT / "metadata.json").write_text(json.dumps({**info, "fine_tuned_from": str(BASE.relative_to(ROOT)),
                                                       "sequence_length": info["sequence_length"]}, indent=2) + "\n")
        print("candidate", json.dumps(candidate), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("selected epoch", chosen_epoch, "elapsed", elapsed, flush=True)


if __name__ == "__main__":
    main()
