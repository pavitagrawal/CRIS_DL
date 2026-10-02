"""
utils/preprocessing.py
-----------------------
Data Preprocessing Pipeline for the Customer Review Intelligence System.

This module handles loading, cleaning, tokenizing, and splitting the
Amazon Customer Reviews dataset for use across all four models.

Author : Pavit Agrawal
Project : ICT 4442 - Deep Learning Mini Project
"""

import os
import re
import time
import numpy as np
import pandas as pd
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download required NLTK resources (runs once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RANDOM_SEED   = 42
MAX_VOCAB     = 20000    # Maximum vocabulary size
MAX_SEQ_LEN   = 200      # Maximum token sequence length after padding/truncating
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

# Star rating → sentiment label mapping
RATING_TO_LABEL = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


# ─────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset(filepath: str, sample_size: int = 50000) -> pd.DataFrame:
    """
    Load the Amazon Customer Reviews dataset from a CSV or JSON file.

    Args:
        filepath    : Path to the dataset file (.csv or .json).
        sample_size : Number of samples to use (stratified). Default 50,000.

    Returns:
        A pandas DataFrame with columns: ['review_text', 'rating', 'label'].
    """
    print(f"[INFO] Loading dataset from: {filepath}")
    start = time.time()

    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(filepath, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json/.jsonl")

    # Standardise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Select and rename relevant columns
    text_col   = _find_column(df, ["reviewtext", "review_text", "text", "reviewbody"])
    rating_col = _find_column(df, ["overall", "rating", "stars", "star_rating"])

    df = df[[text_col, rating_col]].rename(
        columns={text_col: "review_text", rating_col: "rating"}
    )

    # Drop rows with missing values
    df.dropna(subset=["review_text", "rating"], inplace=True)

    # Convert rating to int and filter valid values (1-5)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").dropna().astype(int)
    df = df[df["rating"].between(1, 5)]

    # Map ratings to sentiment labels
    df["label"] = df["rating"].map(RATING_TO_LABEL)

    # Stratified sampling to balance classes
    if len(df) > sample_size:
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["label"],
            random_state=RANDOM_SEED
        )

    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Dataset loaded: {len(df):,} samples in {time.time()-start:.2f}s")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")
    return df


def _find_column(df: pd.DataFrame, candidates: list) -> str:
    """Return the first candidate column name that exists in the DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the expected columns found: {candidates}. "
                   f"Available columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 2: TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str, remove_stopwords: bool = True,
               apply_stemming: bool = False) -> str:
    """
    Clean a raw review string.

    Steps applied:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove non-alphabetic characters (keep spaces)
        5. Collapse multiple spaces
        6. (Optional) Remove English stopwords
        7. (Optional) Apply Porter Stemming

    Args:
        text              : Raw review string.
        remove_stopwords  : Whether to remove stopwords. Default True.
        apply_stemming    : Whether to apply Porter stemming. Default False.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 4. Keep only alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = " ".join(w for w in text.split() if w not in stop_words)

    # 7. Stemming
    if apply_stemming:
        stemmer = PorterStemmer()
        text = " ".join(stemmer.stem(w) for w in text.split())

    return text


def clean_dataset(df: pd.DataFrame, remove_stopwords: bool = True,
                  apply_stemming: bool = False) -> pd.DataFrame:
    """
    Apply clean_text() to every review in the DataFrame.

    Args:
        df                : DataFrame with 'review_text' column.
        remove_stopwords  : Passed to clean_text(). Default True.
        apply_stemming    : Passed to clean_text(). Default False.

    Returns:
        DataFrame with an added 'clean_text' column.
    """
    print("[INFO] Cleaning review text...")
    start = time.time()
    df = df.copy()
    df["clean_text"] = df["review_text"].apply(
        lambda x: clean_text(x, remove_stopwords, apply_stemming)
    )
    # Remove rows where cleaning produced empty strings
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"[INFO] Text cleaning done in {time.time()-start:.2f}s. "
          f"Remaining samples: {len(df):,}\n")
    return df


# ─────────────────────────────────────────────
# STEP 3: TOKENIZATION & VOCABULARY BUILDING
# ─────────────────────────────────────────────
def build_vocabulary(texts: list, max_vocab: int = MAX_VOCAB) -> dict:
    """
    Build a word-to-index vocabulary from a list of cleaned text strings.

    Special tokens:
        <PAD> → index 0  (padding)
        <UNK> → index 1  (unknown words)

    Args:
        texts     : List of cleaned text strings (training set only).
        max_vocab : Maximum vocabulary size (excluding special tokens).

    Returns:
        word2idx : Dictionary mapping word → integer index.
    """
    print(f"[INFO] Building vocabulary (max_vocab={max_vocab:,})...")
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # Most common words up to max_vocab
    most_common = counter.most_common(max_vocab)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx

    print(f"[INFO] Vocabulary built: {len(word2idx):,} tokens "
          f"(including <PAD> and <UNK>)\n")
    return word2idx


def tokenize_and_pad(texts: list, word2idx: dict,
                     max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Convert a list of cleaned text strings to a 2-D integer array
    (post-truncation / zero-padding to a fixed length).

    Args:
        texts       : List of cleaned text strings.
        word2idx    : Vocabulary dictionary (word → index).
        max_seq_len : Fixed sequence length. Default MAX_SEQ_LEN.

    Returns:
        NumPy array of shape (num_samples, max_seq_len).
    """
    sequences = []
    unk_idx   = word2idx.get("<UNK>", 1)
    pad_idx   = word2idx.get("<PAD>", 0)

    for text in texts:
        tokens = text.split()[:max_seq_len]                 # truncate
        indices = [word2idx.get(t, unk_idx) for t in tokens]
        # Pad on the right
        padded = indices + [pad_idx] * (max_seq_len - len(indices))
        sequences.append(padded)

    return np.array(sequences, dtype=np.int32)


# ─────────────────────────────────────────────
# STEP 4: LABEL ENCODING
# ─────────────────────────────────────────────
def encode_labels(labels: pd.Series):
    """
    Encode string sentiment labels to integer class indices.

    Mapping: negative=0, neutral=1, positive=2 (alphabetical order).

    Args:
        labels : Pandas Series of string labels.

    Returns:
        encoded_labels : NumPy array of integer labels.
        encoder        : Fitted LabelEncoder (for decoding predictions later).
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    print(f"[INFO] Label classes: {list(encoder.classes_)}")
    return encoded, encoder


# ─────────────────────────────────────────────
# STEP 5: TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
def split_data(X: np.ndarray, y: np.ndarray):
    """
    Split data into training, validation, and test sets using
    stratified sampling to preserve class distribution.

    Ratios: 70% train / 15% val / 15% test

    Args:
        X : Feature array (tokenized sequences).
        y : Label array (integer encoded).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), stratify=y, random_state=RANDOM_SEED
    )

    # Second split: 50% of temp = 15% val, 50% of temp = 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"[INFO] Data split complete:")
    print(f"       Train : {X_train.shape[0]:>6,} samples")
    print(f"       Val   : {X_val.shape[0]:>6,} samples")
    print(f"       Test  : {X_test.shape[0]:>6,} samples\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# FULL PIPELINE (Single entry point)
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(filepath: str, sample_size: int = 50000,
                               max_vocab: int = MAX_VOCAB,
                               max_seq_len: int = MAX_SEQ_LEN):
    """
    Execute the complete preprocessing pipeline end-to-end.

    Steps:
        1. Load dataset
        2. Clean text
        3. Build vocabulary (on training split only to prevent data leakage)
        4. Tokenize & pad sequences
        5. Encode labels
        6. Split into train / val / test sets

    Args:
        filepath    : Path to the dataset CSV or JSON file.
        sample_size : Number of rows to sample from the dataset.
        max_vocab   : Maximum vocabulary size.
        max_seq_len : Fixed token sequence length.

    Returns:
        A dictionary containing:
            X_train, X_val, X_test : Tokenized and padded integer arrays.
            y_train, y_val, y_test : Integer label arrays.
            word2idx               : Vocabulary mapping.
            label_encoder          : Fitted sklearn LabelEncoder.
            vocab_size             : Total vocabulary size.
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE — Customer Review Intelligence System")
    print("=" * 60)

    # 1. Load
    df = load_dataset(filepath, sample_size=sample_size)

    # 2. Clean
    df = clean_dataset(df)

    # 3. Encode labels first (needed for stratified split)
    y_encoded, label_encoder = encode_labels(df["label"])

    # 4. Preliminary split to build vocabulary ONLY on train portion
    texts = df["clean_text"].tolist()
    X_temp_arr = np.arange(len(texts))  # placeholder indices

    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        X_temp_arr, y_encoded,
        test_size=(1 - TRAIN_RATIO), stratify=y_encoded,
        random_state=RANDOM_SEED
    )
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx, y_temp,
        test_size=0.5, stratify=y_temp,
        random_state=RANDOM_SEED
    )

    train_texts = [texts[i] for i in X_train_idx]
    val_texts   = [texts[i] for i in X_val_idx]
    test_texts  = [texts[i] for i in X_test_idx]

    # 5. Build vocabulary on TRAINING data only
    word2idx = build_vocabulary(train_texts, max_vocab=max_vocab)

    # 6. Tokenize & pad all splits
    print("[INFO] Tokenizing and padding sequences...")
    X_train = tokenize_and_pad(train_texts, word2idx, max_seq_len)
    X_val   = tokenize_and_pad(val_texts,   word2idx, max_seq_len)
    X_test  = tokenize_and_pad(test_texts,  word2idx, max_seq_len)

    print(f"[INFO] Shapes → X_train: {X_train.shape}, "
          f"X_val: {X_val.shape}, X_test: {X_test.shape}\n")

    print("=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)

    return {
        "X_train"       : X_train,
        "X_val"         : X_val,
        "X_test"        : X_test,
        "y_train"       : y_train,
        "y_val"         : y_val,
        "y_test"        : y_test,
        "word2idx"      : word2idx,
        "label_encoder" : label_encoder,
        "vocab_size"    : len(word2idx),
        "num_classes"   : len(label_encoder.classes_),
    }


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Smoke test with synthetic data
    print("[TEST] Running smoke test with synthetic data...\n")

    synthetic = pd.DataFrame({
        "reviewText": [
            "This product is absolutely amazing! Great quality and fast delivery.",
            "Terrible experience. Broke after one day. Do not buy.",
            "It is okay, nothing special. Average product for the price.",
            "Love it! Best purchase I made this year. Highly recommend.",
            "Very disappointing. Poor build quality and bad customer support.",
            "Decent product. Works as described but nothing extraordinary.",
        ] * 500,
        "overall": [5, 1, 3, 5, 1, 3] * 500,
    })

    # Save synthetic CSV temporarily
    os.makedirs("data", exist_ok=True)
    synthetic.to_csv("data/synthetic_test.csv", index=False)

    # Run pipeline
    results = run_preprocessing_pipeline(
        filepath="data/synthetic_test.csv",
        sample_size=3000,
        max_vocab=500,
        max_seq_len=50,
    )

    print(f"Vocab size  : {results['vocab_size']}")
    print(f"Num classes : {results['num_classes']}")
    print(f"X_train     : {results['X_train'].shape}")
    print(f"Classes     : {list(results['label_encoder'].classes_)}")
    print("\n[TEST] Smoke test passed!")
