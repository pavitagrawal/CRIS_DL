"""Train submitted MLP/TextCNN/BiLSTM/Transformer on one shared split.

Requires torch in addition to requirements.txt. Does not replace the app model.
Run: python compare_submitted_models.py --data data/amazon_beauty_balanced.csv
Or: python compare_submitted_models.py --splits-dir data/amazon_50k_multicategory \
        --max-vocab 20000 --max-length 200 --out artifacts/submitted_50k_comparison
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from review_intelligence import LABELS
from train import load_data
from models.model_mlp import MLP
from models.model_textcnn import TextCNN
from models.model_bilstm import BiLSTM
from models.model_transformer import SentimentTransformer


BUILDERS = {
    "MLP": lambda n, length: MLP(n, embed_dim=128, hidden_dim=256),
    "TextCNN": lambda n, length: TextCNN(n, embed_dim=128, num_filters=128),
    "BiLSTM": lambda n, length: BiLSTM(n, embed_dim=128, hidden_dim=128, num_layers=2),
    "Transformer": lambda n, length: SentimentTransformer(n, embed_dim=128, num_heads=4,
                                                     num_layers=2, ffn_dim=256, max_seq_len=length+1),
}


def predict(model, data, batch_size):
    model.eval()
    result = []
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(data), batch_size):
            result.extend(model(data[i:i + batch_size]).argmax(dim=1).tolist())
    return np.array(result), time.perf_counter() - started


def metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1), "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/amazon_beauty_balanced.csv")
    ap.add_argument("--splits-dir", help="Use fixed train.csv, validation.csv, test.csv from this directory")
    ap.add_argument("--out", default="artifacts/submitted_comparison")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-vocab", type=int, default=15000)
    ap.add_argument("--max-length", type=int, default=120)
    args = ap.parse_args()
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if args.splits_dir:
        folder = Path(args.splits_dir)
        tr, ytr = load_data(folder / "train.csv", "review", "sentiment")
        val, yval = load_data(folder / "validation.csv", "review", "sentiment")
        test, ytest = load_data(folder / "test.csv", "review", "sentiment")
    else:
        texts, y = load_data(args.data, "review", "sentiment")
        tr, rest, ytr, yrest = train_test_split(texts, y, test_size=.3, random_state=42, stratify=y)
        val, test, yval, ytest = train_test_split(rest, yrest, test_size=.5, random_state=42, stratify=yrest)
    tokenizer = Tokenizer(num_words=args.max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(tr)
    def encode(items):
        return torch.from_numpy(pad_sequences(tokenizer.texts_to_sequences(items),
                   maxlen=args.max_length, padding="post", truncating="post").astype("int64"))
    xtr, xval, xtest = map(encode, (tr, val, test))
    ytr_t = torch.from_numpy(ytr.astype("int64"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(tokenizer.to_json())
    comparison = []
    for name, build in BUILDERS.items():
        torch.manual_seed(42)
        model = build(min(args.max_vocab, len(tokenizer.word_index) + 2), args.max_length)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4 if name == "Transformer" else 5e-4)
        lossfn = torch.nn.CrossEntropyLoss()
        best_f1 = -1; best_state = None; best_epoch = None
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(xtr))
            for ids in order.split(args.batch_size):
                optim.zero_grad(set_to_none=True)
                loss = lossfn(model(xtr[ids]), ytr_t[ids])
                loss.backward()
                if name in ("BiLSTM", "Transformer"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optim.step()
            vpred, _ = predict(model, xval, args.batch_size)
            vm = metrics(yval, vpred)
            print(name, "epoch", epoch, "validation F1", round(vm["f1_macro"], 4), flush=True)
            if vm["f1_macro"] > best_f1:
                best_f1, best_epoch = vm["f1_macro"], epoch
                best_state = copy.deepcopy(model.state_dict())
        train_seconds = time.perf_counter() - started
        model.load_state_dict(best_state)
        vpred, _ = predict(model, xval, args.batch_size)
        tpred, pred_seconds = predict(model, xtest, args.batch_size)
        vm, tm = metrics(yval, vpred), metrics(ytest, tpred)
        torch.save({"state_dict": best_state, "vocab_size": min(args.max_vocab, len(tokenizer.word_index)+2),
                    "labels": LABELS, "epoch": best_epoch}, out / f"{name.lower()}.pt")
        row = {"model": name, "epoch": best_epoch, "training_seconds": train_seconds,
               "prediction_seconds": pred_seconds, "validation": vm, "test": tm}
        comparison.append(row)
        print(name, "TEST", json.dumps(row), flush=True)
        (out / "results.json").write_text(json.dumps(comparison, indent=2) + "\n")
    winner = max(comparison, key=lambda row: row["validation"]["f1_macro"])["model"]
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "selected_epoch", "validation_accuracy",
          "validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall",
          "test_macro_f1", "training_seconds", "prediction_seconds"])
        writer.writeheader()
        for row in comparison:
            t, v = row["test"], row["validation"]
            writer.writerow({"model": row["model"], "selected_epoch": row["epoch"],
                "validation_accuracy": v["accuracy"], "validation_macro_f1": v["f1_macro"],
                "test_accuracy": t["accuracy"], "test_macro_precision": t["precision_macro"],
                "test_macro_recall": t["recall_macro"], "test_macro_f1": t["f1_macro"],
                "training_seconds": row["training_seconds"],
                "prediction_seconds": row["prediction_seconds"]})
    provenance = json.loads((Path(args.splits_dir) / "provenance.json").read_text()) if args.splits_dir and (Path(args.splits_dir) / "provenance.json").exists() else {}
    (out / "metadata.json").write_text(json.dumps({"source": args.splits_dir or args.data,
      "model_seed":42,"split_seed":provenance.get("seed", 42),"split_sizes":{"train":len(tr),"validation":len(val),"test":len(test)},
      "tokenizer_fit":"training only","vocabulary_limit":args.max_vocab,"sequence_length":args.max_length,
      "model_source":"team repository commit 6221161, with padding-mask fixes to MLP, BiLSTM, Transformer", "selection":"validation macro F1", "winner":winner,
      "note":"architecture comparison produces the base checkpoint used by later PyTorch fine-tuning"}, indent=2) + "\n")
    print("Winner by validation macro F1:", winner, flush=True)


if __name__ == "__main__": main()
