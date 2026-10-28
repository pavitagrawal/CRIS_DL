"""Fine-tune the deployed BiLSTM on human-labeled Yelp sentences.

The Yelp split is stratified and deduplicated. Selection uses validation only;
the held-out Yelp test and product test are reported after selection.
"""
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from fine_tune_short_reviews import LABELS, predict, read, score
from models.model_bilstm import BiLSTM

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/short_review_candidate'
OUT = ROOT / 'artifacts/restaurant_yelp_candidate'


def main():
    random.seed(20261023)
    np.random.seed(20261023)
    torch.manual_seed(20261023)
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = False
    info = json.loads((BASE / 'metadata.json').read_text())
    tokenizer = tokenizer_from_json((BASE / 'tokenizer.json').read_text())
    checkpoint = torch.load(BASE / 'bilstm.pt', map_location='cpu', weights_only=True)
    model = BiLSTM(checkpoint['vocab_size'], embed_dim=128, hidden_dim=128, num_layers=2)
    model.load_state_dict(checkpoint['state_dict'])

    def encode(rows):
        seq = tokenizer.texts_to_sequences([row['review'] for row in rows])
        return torch.tensor(pad_sequences(seq, maxlen=info['sequence_length'], padding='post', truncating='post'), dtype=torch.long)

    def labels(rows):
        return np.array([LABELS.index(row.get('sentiment', row.get('human_sentiment'))) for row in rows])

    seen = set()
    yelp = []
    for line in (ROOT / 'data/uci_yelp_labelled.txt').read_text(encoding='utf-8-sig').splitlines():
        review, label = line.rsplit('\t', 1)
        key = ' '.join(review.casefold().split())
        if key not in seen:
            seen.add(key)
            yelp.append({'review': review, 'sentiment': 'positive' if label == '1' else 'negative'})
    train, remainder = train_test_split(yelp, test_size=.4, random_state=20261023, stratify=labels(yelp))
    validation, test = train_test_split(remainder, test_size=.5, random_state=20261023, stratify=labels(remainder))
    other = read(ROOT / 'data/restaurant_rating_proxy_90.csv')
    other_keys = {' '.join(row['text'].casefold().split()) for row in other}
    train = [row for row in train if ' '.join(row['review'].casefold().split()) not in other_keys]
    source = read(ROOT / 'data/amazon_50k_multicategory/train.csv')
    replay = []
    for sentiment in LABELS:
        replay.extend(random.sample([row for row in source if row['sentiment'] == sentiment], 1200))
    replay_keys = {' '.join(row['review'].casefold().split()) for row in replay}
    train = [row for row in train if ' '.join(row['review'].casefold().split()) not in replay_keys]
    train_rows = replay + train * 4
    sets = {'yelp_validation': validation, 'yelp_test': test,
            'product_validation': read(ROOT / 'data/amazon_50k_multicategory/validation.csv'),
            'product_test': read(ROOT / 'data/amazon_50k_multicategory/test.csv'),
            'human_50': read(ROOT / 'data/human_evaluation_sample.csv')}
    # The rating-proxy file may use different column names; keep it out of selection.
    x = {name: encode(rows) for name, rows in sets.items()}
    y = {name: labels(rows) for name, rows in sets.items()}
    base = {name: score(y[name], predict(model, x[name])) for name in sets}
    print('baseline', json.dumps(base), flush=True)
    xtrain = encode(train_rows)
    ytrain = torch.tensor(labels(train_rows), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    chosen = None
    selected_state = None
    best_f1 = base['yelp_validation']['macro_f1']
    started = time.perf_counter()
    for epoch in range(1, 4):
        model.train()
        for indices in torch.randperm(len(xtrain)).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtrain[indices]), ytrain[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        val = score(y['yelp_validation'], predict(model, x['yelp_validation']))
        product = score(y['product_validation'], predict(model, x['product_validation']))
        history.append({'epoch': epoch, 'yelp_validation': val, 'product_validation': product})
        print('epoch', json.dumps(history[-1]), flush=True)
        safe = (product['macro_f1'] >= base['product_validation']['macro_f1'] - .015
                and product['neutral_recall'] >= base['product_validation']['neutral_recall'] - .03)
        if safe and val['macro_f1'] > best_f1:
            chosen, best_f1, selected_state = epoch, val['macro_f1'], copy.deepcopy(model.state_dict())
    result = {'protocol': 'Deduplicated UCI Yelp stratified 60/20/20; Yelp train with 1200 product replay per class; select on validation only',
              'counts': {'yelp_train': len(train), 'yelp_validation': len(validation), 'yelp_test': len(test),
                         'product_replay': len(replay)},
              'baseline': base, 'history': history, 'selected_epoch': chosen,
              'train_seconds': time.perf_counter() - started}
    if selected_state is not None:
        model.load_state_dict(selected_state)
        result['candidate'] = {name: score(y[name], predict(model, x[name])) for name in sets}
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({**checkpoint, 'state_dict': selected_state, 'epoch': chosen}, OUT / 'bilstm.pt')
        (OUT / 'tokenizer.json').write_text((BASE / 'tokenizer.json').read_text())
        (OUT / 'metadata.json').write_text(json.dumps({**info, 'fine_tuned_from': str(BASE.relative_to(ROOT)),
                                                       'yelp_training_rows': len(train)}, indent=2) + '\n')
        print('candidate', json.dumps(result['candidate']), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('selected', chosen, 'seconds', result['train_seconds'], flush=True)


if __name__ == '__main__':
    main()
