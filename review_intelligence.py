"""Shared preprocessing, inference and transparent keyword based aspect analysis."""

import json
import re
from pathlib import Path

import numpy as np

from aspect_rules import explicit_battery_polarity
from aspect_polarity import explicit_aspect_polarity


ASPECT_KEYWORDS = {
    "Camera": ["camera", "photo", "picture", "lens", "selfie", "webcam"],
    "Battery": ["battery", "batteries", "charging", "charge", "charger", "power"],
    "Design": ["design", "look", "appearance", "style", "build"],
    "Performance": ["performance", "speed", "fast", "slow", "lag", "processor"],
    "Price": ["price", "cost", "expensive", "cheap", "value"],
    "Customer support": ["customer support", "tech support", "customer service", "support", "service", "customer care", "helpdesk"],
    "Sound": ["sound", "audio", "speaker", "speakers", "headphone", "bass",
              "noise", "noisy", "quiet"],
    "Screen": ["screen", "display", "brightness", "resolution"],
    "Delivery": ["delivery", "shipping", "arrived", "packaging"],
    "Quality": ["quality", "durable", "durability", "material"],
    "Keyboard": ["keyboard", "keys", "keypad", "typing"],
    "Touchpad": ["touchpad", "touch pad", "trackpad", "mousepad", "mouse pad", "mouse"],
    "Storage": ["hard drive", "hard disk", "ssd", "storage", "disk space", "memory", "ram"],
    "Software": ["software", "program", "programs", "application", "applications", "operating system", "windows", "vista", "updates"],
    "Warranty": ["warranty", "warrenty", "guarantee"],
    "Connectivity": ["wifi", "wi-fi", "internet", "bluetooth", "usb", "ports"],
    "Portability": ["portability", "portable", "weight", "lightweight", "size"],
    "Graphics": ["graphics", "gpu", "gaming"],
    "Features": ["features", "feature", "specs"],
}
RESTAURANT_ASPECT_KEYWORDS = {
    "Food": ["food", "meal", "meals", "dish", "dishes", "menu", "pizza", "sushi", "dessert", "desserts", "appetizer", "appetizers", "entree", "entrees", "steak", "fish", "chicken", "bread", "bagel", "bagels", "rice", "salad", "salads", "sauce", "flavor", "taste", "portions", "portion", "ingredients", "cuisine", "dinner", "lunch", "brunch", "chef"],
    "Service": ["service", "staff", "waiter", "waiters", "waitress", "waitstaff", "wait staff", "server", "servers", "hostess", "manager", "management", "bartender", "reservation", "reservations", "served"],
    "Atmosphere": ["atmosphere", "ambience", "ambiance", "decor", "music", "seating", "space", "setting", "vibe"],
    "Price": ["price", "prices", "priced", "bill", "cost", "expensive", "value"],
    "Drinks": ["drink", "drinks", "wine", "wines", "wine list", "beer", "cocktail", "cocktails"],
    "Delivery": ["delivery", "delivered", "takeout", "take-out"],
}
ASPECT_PROFILES = {"Products and electronics": ASPECT_KEYWORDS,
                   "Restaurants": RESTAURANT_ASPECT_KEYWORDS}
LABELS = ["negative", "neutral", "positive"]


def suggested_aspect_profile(text, selected):
    """Suggest the other profile only when it has distinct matches and this one has none."""
    if aspects_in(text, ASPECT_PROFILES[selected]):
        return None
    other = next(name for name in ASPECT_PROFILES if name != selected)
    current_terms = {term for words in ASPECT_PROFILES[selected].values() for term in words}
    distinct = {name: [term for term in words if term not in current_terms]
                for name, words in ASPECT_PROFILES[other].items()}
    return other if aspects_in(text, distinct) else None


def clean_text(value):
    """Keep words and punctuation, including apostrophes for negation."""
    value = str(value) if value is not None else ""
    return re.sub(r"\s+", " ", value).strip()


def aspect_clauses(review, keywords=None):
    """Split on sentence boundaries and contrast/connective words."""
    parts = [part.strip() for part in re.split(
        r"(?<=[.!?;])\s+|\s*;\s*|\s*,\s*|\s+\b(?:but|however|although|whereas)\b\s+",
        clean_text(review), flags=re.I) if part.strip()]
    clauses = []
    for part in parts:
        sides = re.split(r"\s+and\s+", part, maxsplit=1, flags=re.I)
        if len(sides) == 2 and all(len(side.split()) >= 3 for side in sides):
            left, right = (aspects_in(side, keywords) for side in sides)
            if (len(left) == len(right) == 1 and left != right and
                    explicit_aspect_polarity(sides[0], left[0], keywords) and
                    explicit_aspect_polarity(sides[1], right[0], keywords)):
                clauses.extend(side.strip() for side in sides)
                continue
        clauses.append(part)
    return clauses


def aspect_matches(text, keywords=ASPECT_KEYWORDS):
    """Return aspect and matched term spans, preferring longer overlapping terms."""
    if keywords is None:
        keywords = ASPECT_KEYWORDS
    found = []
    for aspect, terms in keywords.items():
        matches = []
        for word in sorted(terms, key=len, reverse=True):
            for match in re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I):
                if not any(match.start() < end and match.end() > start for start, end in matches):
                    matches.append((match.start(), match.end()))
        for start, end in sorted(matches):
            found.append((aspect, start, end))
    return found


def aspects_in(text, keywords=None):
    keywords = ASPECT_KEYWORDS if keywords is None else keywords
    matched = {aspect for aspect, _, _ in aspect_matches(text, keywords)}
    return [aspect for aspect in keywords if aspect in matched]


def aspect_summary(items):
    """Summarize detected aspect labels without changing overall model output."""
    if not items:
        return "No aspects found"
    labels = {item["sentiment"] for item in items}
    return next(iter(labels)).title() if len(labels) == 1 else "Mixed"


def clear_restaurant_sentiment(review):
    """Return a clear explicit cue for one short, single-aspect restaurant clause."""
    clauses = aspect_clauses(review, RESTAURANT_ASPECT_KEYWORDS)
    if len(clauses) != 1 or len(review.split()) > 30:
        return None
    found = aspects_in(clauses[0], RESTAURANT_ASPECT_KEYWORDS)
    if len(found) != 1:
        return None
    return explicit_aspect_polarity(clauses[0], found[0], RESTAURANT_ASPECT_KEYWORDS)


class ReviewPredictor:
    def __init__(self, artifact_dir="artifacts"):
        from tensorflow import keras
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        directory = Path(artifact_dir)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.max_length = metadata["max_length"]
        self.labels = metadata["labels"]
        self.tokenizer = tokenizer_from_json((directory / "tokenizer.json").read_text(encoding="utf-8"))
        self.model = keras.models.load_model(directory / "best_model.keras")

    def probabilities(self, texts, batch_size=256):
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if not texts:
            return np.empty((0, len(self.labels)))
        sequences = self.tokenizer.texts_to_sequences([clean_text(t) for t in texts])
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")
        return self.model.predict(padded, batch_size=batch_size, verbose=0)

    def analyze(self, texts, batch_size=256, aspect_profile="Products and electronics"):
        """Batch predict reviews and their keyword-matched clauses."""
        keywords = ASPECT_PROFILES[aspect_profile]
        texts = [clean_text(t) for t in texts]
        overall = self.probabilities(texts, batch_size)
        clause_jobs = []
        for row, review in enumerate(texts):
            for clause in aspect_clauses(review, keywords):
                for aspect in aspects_in(clause, keywords):
                    clause_jobs.append((row, aspect, clause))
        clause_probs = self.probabilities([item[2] for item in clause_jobs], batch_size)
        findings = [[] for _ in texts]
        for (row, aspect, clause), probs in zip(clause_jobs, clause_probs):
            index = int(np.argmax(probs))
            battery_override = explicit_battery_polarity(clause) if aspect == "Battery" else None
            override = battery_override or explicit_aspect_polarity(clause, aspect, keywords)
            findings[row].append({"aspect": aspect, "sentiment": override or self.labels[index],
                                  "confidence": None if override else float(probs[index]),
                                  "method": "explicit phrase" if battery_override else
                                            "explicit cue" if override else "clause model",
                                  "evidence": clause})
        result = []
        for review, probs, aspect_items in zip(texts, overall, findings):
            index = int(np.argmax(probs))
            model_sentiment = self.labels[index]
            cue = clear_restaurant_sentiment(review) if aspect_profile == "Restaurants" else None
            result.append({"sentiment": cue or model_sentiment, "model_sentiment": model_sentiment,
                           "sentiment_method": "explicit restaurant cue" if cue and cue != model_sentiment else "model",
                           "confidence": float(probs[index]),
                           "aspect_summary": aspect_summary(aspect_items), "aspects": aspect_items})
        return result


class SubmittedBiLSTMPredictor(ReviewPredictor):
    """Serve the 50,000-review PyTorch checkpoint with its own training tokenizer."""

    def __init__(self, artifact_dir="artifacts/submitted_50k_comparison"):
        import torch
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        from models.model_bilstm import BiLSTM

        directory = Path(artifact_dir)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(directory / "bilstm.pt", map_location="cpu", weights_only=True)
        self.max_length = metadata["sequence_length"]
        self.labels = checkpoint["labels"]
        self.tokenizer = tokenizer_from_json((directory / "tokenizer.json").read_text(encoding="utf-8"))
        self.model = BiLSTM(checkpoint["vocab_size"], embed_dim=128, hidden_dim=128, num_layers=2)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def probabilities(self, texts, batch_size=128):
        import torch
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if not texts:
            return np.empty((0, len(self.labels)))
        sequences = self.tokenizer.texts_to_sequences([clean_text(t) for t in texts])
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")
        results = []
        with torch.inference_mode():
            for start in range(0, len(padded), batch_size):
                logits = self.model(torch.from_numpy(padded[start:start + batch_size].astype("int64")))
                results.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(results, axis=0)
