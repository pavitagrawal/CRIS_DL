"""Dependency-light completeness checks for the final project bundle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(relative: str, minimum_bytes: int = 1) -> Path:
    path = ROOT / relative
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise AssertionError(f"Missing or incomplete required file: {relative}")
    return path


def read_json(relative: str):
    return json.loads(require(relative).read_text(encoding="utf-8-sig"))


def main() -> None:
    required = [
        "app.py",
        "README.md",
        "FINAL_PROJECT_GUIDE.md",
        "FINAL_COMPLETION_AUDIT.md",
        "FINAL_ARCHITECTURE_ANALYSIS.md",
        "compare_submitted_models.py",
        "prepare_multicategory_dataset.py",
        "models/model_mlp.py",
        "models/model_textcnn.py",
        "models/model_bilstm.py",
        "models/model_transformer.py",
        "tests/test_logic.py",
        "artifacts/short_review_candidate/bilstm.pt",
        "artifacts/short_review_candidate/tokenizer.json",
        "artifacts/submitted_50k_comparison/comparison.csv",
        "artifacts/final_architecture_analysis.json",
    ]
    for relative in required:
        require(relative)

    results = read_json("artifacts/submitted_50k_comparison/results.json")
    names = {row["model"] for row in results}
    if names != {"MLP", "TextCNN", "BiLSTM", "Transformer"}:
        raise AssertionError(f"Unexpected architecture set: {sorted(names)}")
    winner = max(results, key=lambda row: row["validation"]["f1_macro"])["model"]
    metadata = read_json("artifacts/submitted_50k_comparison/metadata.json")
    if metadata["winner"] != winner:
        raise AssertionError("Saved winner does not match validation macro F1")
    if winner != "BiLSTM":
        raise AssertionError(f"Expected saved final winner BiLSTM, found {winner}")

    runtime = require("artifacts/short_review_candidate/bilstm.pt", 1_000_000)
    tokenizer = read_json("artifacts/short_review_candidate/tokenizer.json")
    if not tokenizer or runtime.stat().st_size < 1_000_000:
        raise AssertionError("Runtime model bundle is incomplete")

    for path in ROOT.rglob("*.json"):
        if ".git" not in path.parts:
            json.loads(path.read_text(encoding="utf-8-sig"))

    print(f"Project bundle validated: 4 architectures, winner {winner}, runtime checkpoint present")


if __name__ == "__main__":
    main()
