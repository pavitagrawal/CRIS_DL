"""Measure how often restaurant model disagreement identifies default-model errors.

The Yelp source has only Positive/Negative reference labels. Three-star ratings
in the other source are imperfect stand-ins for Neutral, not human annotations.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from review_intelligence import SubmittedBiLSTMPredictor
from review_queue import review_reasons


def datasets(yelp_path):
    with open('data/restaurant_rating_proxy_90.csv', newline='', encoding='utf-8-sig') as handle:
        rating = list(csv.DictReader(handle))
    rating_rows = [(r['text'].removeprefix('RATED  ').strip(),
                    'negative' if int(re.search('[1-5]', r['rating']).group()) <= 2 else
                    'neutral' if int(re.search('[1-5]', r['rating']).group()) == 3 else 'positive')
                   for r in rating]
    unique = {}
    with open(yelp_path, encoding='utf-8-sig') as handle:
        for line in handle:
            text, label = line.rstrip('\n').rsplit('\t', 1)
            unique.setdefault(' '.join(text.casefold().split()),
                              (text, 'positive' if label == '1' else 'negative'))
    assert len(rating_rows) == 90 and len(unique) == 996
    return {'restaurant_rating_proxy_90': rating_rows,
            'uci_yelp_binary_996': list(unique.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yelp', default='data/uci_yelp_labelled.txt')
    parser.add_argument('--output', default='artifacts/restaurant_model_disagreement_evaluation.json')
    args = parser.parse_args()
    default = SubmittedBiLSTMPredictor('artifacts/short_review_candidate')
    alternate = SubmittedBiLSTMPredictor('artifacts/restaurant_candidate')
    report = {'method': 'Raw overall predictions from two saved models; no labels are changed',
              'sources': {'uci_yelp_binary_996': 'UCI Sentiment Labelled Sentences (CC BY 4.0), Yelp subset',
                          'restaurant_rating_proxy_90': '30 each from 1-2, 3, and 4-5 star groups'},
              'results': {}}
    for name, rows in datasets(args.yelp).items():
        texts = [text for text, _ in rows]
        first = [default.labels[i] for i in np.argmax(default.probabilities(texts), axis=1)]
        second = [alternate.labels[i] for i in np.argmax(alternate.probabilities(texts), axis=1)]
        disagreement = [a != b for a, b in zip(first, second)]
        error = [pred != gold for pred, (_, gold) in zip(first, rows)]
        analyzed = default.analyze(texts, aspect_profile='Restaurants')
        previous_flags = [bool(review_reasons(result, text))
                          for result, text in zip(analyzed, texts)]
        combined_flags = [bool(review_reasons(result, text, alt))
                          for result, text, alt in zip(analyzed, texts, second)]
        flagged = sum(disagreement)
        flagged_errors = sum(d and e for d, e in zip(disagreement, error))
        report['results'][name] = {
            'reviews': len(rows), 'default_errors': sum(error), 'model_disagreements': flagged,
            'default_errors_with_disagreement': flagged_errors,
            'error_rate_among_disagreements': flagged_errors / flagged,
            'overall_error_rate': sum(error) / len(rows),
            'error_recall_from_disagreement': flagged_errors / sum(error),
            'existing_queue_flagged': sum(previous_flags),
            'existing_queue_errors_flagged': sum(f and e for f, e in zip(previous_flags, error)),
            'combined_queue_flagged': sum(combined_flags),
            'combined_queue_errors_flagged': sum(f and e for f, e in zip(combined_flags, error)),
            'newly_flagged_reviews': sum(f and not old for f, old in zip(combined_flags, previous_flags)),
            'newly_flagged_errors': sum(f and e and not old for f, e, old in zip(combined_flags, error, previous_flags)),
        }
    report['limits'] = ('Sources were inspected before this queue option was added; these are exploratory results. '
                        'A disagreement is neither a human label nor proof of error. It raises false alarms and '
                        'misses errors both models make. The combined review queue has not been separately evaluated.')
    target = Path(args.output)
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()
