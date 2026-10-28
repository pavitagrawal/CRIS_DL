"""Sample a documented, balanced 50,000-review Amazon Reviews 2023 benchmark.

Uses a fixed prefix of each official category file and reservoir sampling.
Run: python prepare_multicategory_dataset.py
"""
import csv
import hashlib
import json
import random
from pathlib import Path
from contextlib import closing
from sklearn.model_selection import train_test_split
from prepare_dataset import source_lines, LABEL_BY_RATING

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"
CATEGORIES = ("All_Beauty", "Books", "Electronics", "Home_and_Kitchen")
QUOTAS = {"negative": 4167, "neutral": 4166, "positive": 4167}
SCAN_LIMIT = 200000
SEED = 20261021


def sample_category(category, seen):
    rng = random.Random(SEED + CATEGORIES.index(category))
    buckets = {label: [] for label in QUOTAS}
    counts = {label: 0 for label in QUOTAS}
    url = BASE + category + ".jsonl.gz"
    scanned = 0
    with closing(source_lines(url)) as lines:
        for line in lines:
            if scanned >= SCAN_LIMIT:
                break
            scanned += 1
            entry = json.loads(line)
            text = " ".join(str(entry.get("text") or "").split())
            try:
                rating = int(entry.get("rating"))
            except (TypeError, ValueError):
                continue
            if rating not in LABEL_BY_RATING or len(text) < 20:
                continue
            digest = hashlib.blake2b(text.lower().encode(), digest_size=16).digest()
            if digest in seen:
                continue
            seen.add(digest)
            label = LABEL_BY_RATING[rating]
            counts[label] += 1
            row = {"review": text, "rating": rating, "sentiment": label,
                   "category": category, "product_id": str(entry.get("parent_asin") or "")}
            if len(buckets[label]) < QUOTAS[label]:
                buckets[label].append(row)
            else:
                index = rng.randrange(counts[label])
                if index < QUOTAS[label]:
                    buckets[label][index] = row
    if any(len(buckets[label]) != QUOTAS[label] for label in QUOTAS):
        raise ValueError(f"{category} has insufficient labels: {counts}")
    return [row for bucket in buckets.values() for row in bucket], {
        "source_url": url, "scanned_rows": scanned, "usable_unique_by_label": counts,
        "sampled_by_label": {label: len(buckets[label]) for label in QUOTAS}}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["review", "rating", "sentiment", "category", "product_id"])
        writer.writeheader(); writer.writerows(rows)


def main():
    folder = Path("data/amazon_50k_multicategory"); folder.mkdir(parents=True, exist_ok=True)
    seen = set(); rows = []; sources = {}
    for category in CATEGORIES:
        print("Sampling", category, flush=True)
        part, report = sample_category(category, seen)
        rows.extend(part); sources[category] = report
        print(category, report["usable_unique_by_label"], flush=True)
    assert len(rows) == 50000
    rng = random.Random(SEED);rng.shuffle(rows)
    write_csv(folder / "reviews_50000.csv", rows)
    stratify = [row["category"] + ":" + row["sentiment"] for row in rows]
    train, rest = train_test_split(rows, test_size=.3, stratify=stratify, random_state=SEED)
    val, test = train_test_split(rest, test_size=.5,
        stratify=[row["category"] + ":" + row["sentiment"] for row in rest], random_state=SEED)
    for name, group in (("train",train),("validation",val),("test",test)):
        write_csv(folder / (name + ".csv"), group)
    report = {"source":"Amazon Reviews 2023, McAuley Lab / UCSD",
       "note":"New sample, not the exact 50,000 rows used in the earlier interim report",
       "categories":sources,"seed":SEED,"scan_limit_per_category":SCAN_LIMIT,
       "sampling":"reservoir sample within first 200,000 records of each category, balanced by class",
       "min_characters":20,"deduplication":"case-insensitive exact text across all categories",
       "labels":"1-2 negative; 3 neutral; 4-5 positive (rating proxy)",
       "split":"stratified by category and sentiment, review-level, 70/15/15; products may overlap across splits",
       "sizes":{"all":len(rows),"train":len(train),"validation":len(val),"test":len(test)}}
    (folder / "provenance.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved",report["sizes"],flush=True)

if __name__ == "__main__":main()
