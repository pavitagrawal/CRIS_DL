"""
models/train_mlp.py
--------------------
Training and evaluation script for the MLP sentiment classifier.

Trains the MLP model, tracks metrics per epoch, evaluates on the
test set, and saves results (classification report + confusion matrix)
to the results/ folder.

Author : Pavit Agrawal
Project : ICT 4442 - Deep Learning Mini Project

Usage:
    python models/train_mlp.py --data_path data/reviews.csv
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Add project root to path so imports work from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import run_preprocessing_pipeline
from models.model_mlp import MLP, count_parameters


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
CONFIG = {
    "embed_dim"    : 128,
    "hidden_dim"   : 256,
    "dropout_rate" : 0.4,
    "num_classes"  : 3,
    "batch_size"   : 64,
    "epochs"       : 15,
    "lr"           : 1e-3,
    "weight_decay" : 1e-4,
    "sample_size"  : 50000,
    "max_vocab"    : 20000,
    "max_seq_len"  : 200,
}

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def make_loaders(data):
    """Wrap numpy arrays into PyTorch DataLoaders."""
    def to_loader(X, y, shuffle=False):
        X_t = torch.tensor(X, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=shuffle)

    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader   = to_loader(data["X_val"],   data["y_val"])
    test_loader  = to_loader(data["X_test"],  data["y_test"])
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += X_batch.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += X_batch.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# FULL TEST EVALUATION (metrics + confusion matrix)
# ─────────────────────────────────────────────
def full_evaluation(model, loader, label_encoder):
    model.eval()
    all_preds, all_labels = [], []

    start = time.time()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    pred_time = time.time() - start

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc        = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    return {
        "accuracy"       : round(acc, 4),
        "precision"      : round(prec, 4),
        "recall"         : round(rec, 4),
        "f1_score"       : round(f1, 4),
        "pred_time_sec"  : round(pred_time, 4),
        "confusion_matrix": cm.tolist(),
        "report"         : report,
        "cm_array"       : cm,
    }


# ─────────────────────────────────────────────
# SAVE CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("MLP — Confusion Matrix (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


# ─────────────────────────────────────────────
# SAVE TRAINING CURVES PLOT
# ─────────────────────────────────────────────
def save_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title("MLP — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_title("MLP — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train(data_path: str):
    print(f"\n[INFO] Device: {DEVICE}")
    print(f"[INFO] Config: {CONFIG}\n")

    # ── 1. Preprocess data ──────────────────────────────────────────
    data = run_preprocessing_pipeline(
        filepath    = data_path,
        sample_size = CONFIG["sample_size"],
        max_vocab   = CONFIG["max_vocab"],
        max_seq_len = CONFIG["max_seq_len"],
    )
    train_loader, val_loader, test_loader = make_loaders(data)

    # ── 2. Build model ──────────────────────────────────────────────
    model = MLP(
        vocab_size   = data["vocab_size"],
        embed_dim    = CONFIG["embed_dim"],
        hidden_dim   = CONFIG["hidden_dim"],
        num_classes  = CONFIG["num_classes"],
        dropout_rate = CONFIG["dropout_rate"],
    ).to(DEVICE)

    print(f"[INFO] Model parameters: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── 3. Training loop ────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    best_model_path = os.path.join(RESULTS_DIR, "mlp_best.pt")
    train_start   = time.time()

    print("─" * 65)
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("─" * 65)

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"{epoch:>6} {tr_loss:>12.4f} {tr_acc:>10.4f} {vl_loss:>10.4f} {vl_acc:>9.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_model_path)

    total_train_time = time.time() - train_start
    print("─" * 65)
    print(f"[INFO] Training done in {total_train_time:.2f}s | Best Val Acc: {best_val_acc:.4f}\n")

    # ── 4. Load best model and evaluate on test set ─────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    metrics = full_evaluation(model, test_loader, data["label_encoder"])

    print("=" * 50)
    print("  MLP — TEST SET RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}  (macro)")
    print(f"  Recall    : {metrics['recall']}  (macro)")
    print(f"  F1-Score  : {metrics['f1_score']}  (macro)")
    print(f"  Pred Time : {metrics['pred_time_sec']}s")
    print(f"  Train Time: {round(total_train_time, 2)}s")
    print("=" * 50)
    print("\nClassification Report:\n")
    print(metrics["report"])

    # ── 5. Save results ─────────────────────────────────────────────
    results_summary = {
        "model"          : "MLP",
        "accuracy"       : metrics["accuracy"],
        "precision"      : metrics["precision"],
        "recall"         : metrics["recall"],
        "f1_score"       : metrics["f1_score"],
        "train_time_sec" : round(total_train_time, 2),
        "pred_time_sec"  : metrics["pred_time_sec"],
        "config"         : CONFIG,
    }

    results_path = os.path.join(RESULTS_DIR, "mlp_results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"[INFO] Results saved to: {results_path}")

    save_confusion_matrix(
        metrics["cm_array"],
        class_names=list(data["label_encoder"].classes_),
        save_path=os.path.join(RESULTS_DIR, "mlp_confusion_matrix.png"),
    )

    save_training_curves(
        history,
        save_path=os.path.join(RESULTS_DIR, "mlp_training_curves.png"),
    )

    print("\n[INFO] All results saved to results/ folder.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP sentiment classifier")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the Amazon reviews CSV or JSON file"
    )
    args = parser.parse_args()
    train(args.data_path)
