"""Prepare a balanced academic sample from McAuley Lab Amazon Reviews 2023."""

import argparse
import csv
import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path

SOURCE_URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
              "raw/review_categories/All_Beauty.jsonl.gz")
LABEL_BY_RATING = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def source_lines(source):
    """Stream a local or remote gzip JSONL source without extracting it."""
    if source.startswith(("https://", "http://")):
        response = urllib.request.urlopen(source, timeout=90)
    else:
        response = open(source, "rb")
    with response, gzip.GzipFile(fileobj=response) as gz:
        yield from gz


def prepare(lines, per_class, seed=42):
    rng = random.Random(seed)
    buckets = {label: [] for label in ("negative", "neutral", "positive")}
    counts = {label: 0 for label in buckets}
    seen = set()
    scanned = 0
    for line in lines:
        scanned += 1
        item = json.loads(line)
        text = " ".join(str(item.get("text") or "").split())
        try:
            rating = int(item.get("rating"))
        except (ValueError, TypeError):
            continue
        if rating not in LABEL_BY_RATING or len(text) < 20:
            continue
        digest = hashlib.blake2b(text.lower().encode("utf-8"), digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        label = LABEL_BY_RATING[rating]
        counts[label] += 1
        row = {"review": text, "sentiment": label, "rating": rating,
               "product_id": str(item.get("parent_asin") or item.get("asin") or "")}
        if len(buckets[label]) < per_class:
            buckets[label].append(row)
        else:
            position = rng.randrange(counts[label])
            if position < per_class:
                buckets[label][position] = row
    if any(len(buckets[label]) < per_class for label in buckets):
        raise ValueError(f"Insufficient usable reviews for {per_class} per class: {counts}")
    rows = [row for label in buckets for row in buckets[label]]
    rng.shuffle(rows)
    return rows, {"scanned_rows": scanned, "usable_unique_by_class": counts,
                  "sampled_by_class": {label: len(buckets[label]) for label in buckets},
                  "total_sampled": len(rows), "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_URL, help="Official .jsonl.gz URL or local .jsonl.gz path")
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/amazon_beauty_balanced.csv")
    args = parser.parse_args()
    if args.per_class < 10:
        parser.error("--per-class must be at least 10")
    print(f"Reading {args.source}", flush=True)
    rows, report = prepare(source_lines(args.source), args.per_class, args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["review", "sentiment", "rating", "product_id"])
        writer.writeheader()
        writer.writerows(rows)
    report.update({"source": args.source, "label_rule": "1-2 negative; 3 neutral; 4-5 positive",
                   "min_review_characters": 20, "deduplication": "case-insensitive exact text"})
    destination.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
