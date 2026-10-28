"""Development check for a narrow, explicit restaurant sentiment rule.

Run before deciding whether to deploy a rule; test sets were previously inspected.
"""
import csv
import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from review_intelligence import (RESTAURANT_ASPECT_KEYWORDS, SubmittedBiLSTMPredictor,
                                 aspect_clauses, aspects_in)
from aspect_polarity import explicit_aspect_polarity


def explicit_single_aspect(text):
    clauses = aspect_clauses(text, RESTAURANT_ASPECT_KEYWORDS)
    if len(clauses) != 1 or len(text.split()) > 30:
        return None
    aspects = aspects_in(clauses[0], RESTAURANT_ASPECT_KEYWORDS)
    if len(aspects) != 1:
        return None
    return explicit_aspect_polarity(clauses[0], aspects[0], RESTAURANT_ASPECT_KEYWORDS)


def collect():
    with open('data/restaurant_rating_proxy_90.csv', encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    rating = [(row['text'].removeprefix('RATED  ').strip(),
               'negative' if int(re.search('[1-5]', row['rating']).group()) <= 2 else
               'neutral' if int(re.search('[1-5]', row['rating']).group()) == 3 else 'positive')
              for row in rows]
    seen = set()
    yelp = []
    with open('data/uci_yelp_labelled.txt', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().rsplit('\t', 1)
            key = ' '.join(text.casefold().split())
            if key not in seen:
                yelp.append((text, 'positive' if label == '1' else 'negative'))
                seen.add(key)
    assert len(rating) == 90 and len(yelp) == 996
    return {'rating_proxy_90': rating, 'human_binary_yelp_996': yelp}


def score(rows, original, adjusted):
    gold = [label for _, label in rows]
    labels = ['negative', 'neutral', 'positive']
    changed = [i for i, (a, b) in enumerate(zip(original, adjusted)) if a != b]
    return {
        'before_accuracy': accuracy_score(gold, original),
        'after_accuracy': accuracy_score(gold, adjusted),
        'before_macro_f1': f1_score(gold, original, labels=labels, average='macro', zero_division=0),
        'after_macro_f1': f1_score(gold, adjusted, labels=labels, average='macro', zero_division=0),
        'before_recall': dict(zip(labels, recall_score(gold, original, labels=labels, average=None, zero_division=0))),
        'after_recall': dict(zip(labels, recall_score(gold, adjusted, labels=labels, average=None, zero_division=0))),
        'changed': len(changed),
        'corrected': sum(original[i] != gold[i] and adjusted[i] == gold[i] for i in changed),
        'harmed': sum(original[i] == gold[i] and adjusted[i] != gold[i] for i in changed),
        'examples': [{'review': rows[i][0], 'gold': gold[i], 'before': original[i], 'after': adjusted[i]}
                     for i in changed[:25]],
    }


def main():
    model = SubmittedBiLSTMPredictor('artifacts/short_review_candidate')
    restaurant = SubmittedBiLSTMPredictor('artifacts/restaurant_candidate')
    out = {}
    for dataset, rows in collect().items():
        predictions = model.probabilities([text for text, _ in rows])
        original = [model.labels[i] for i in np.argmax(predictions, axis=1)]
        adjusted = [explicit_single_aspect(text) or label for (text, _), label in zip(rows, original)]
        out[dataset] = score(rows, original, adjusted)
        second = restaurant.probabilities([text for text, _ in rows])
        ensemble = [model.labels[i] for i in np.argmax((predictions + second) / 2, axis=1)]
        out[dataset + '_ensemble'] = score(rows, original, ensemble)
        print(dataset, {k: v for k, v in out[dataset].items() if k != 'examples'})
        print(dataset + '_ensemble', {k: v for k, v in out[dataset + '_ensemble'].items() if k != 'examples'})
    with open('artifacts/restaurant_guardrail_exploration.json', 'w', encoding='utf-8') as file:
        json.dump(out, file, indent=2)


if __name__ == '__main__':
    main()
