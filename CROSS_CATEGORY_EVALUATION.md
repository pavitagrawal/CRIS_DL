# Exploratory evaluation on phone and accessory reviews

## Question

How does the **original beauty-trained BiLSTM** classify reviews in a different product category? This historical evaluation preceded retraining on phone reviews. The original model is now in `artifacts/beauty_baseline/`, while the app uses the retrained BiLSTM. This is an evaluation of overall review sentiment only. The app's aspect keywords and aspect polarity were not evaluated here.

## Method

- Source: McAuley Lab Amazon Reviews 2023, `Cell_Phones_and_Accessories` review category, using the official compressed JSONL file linked in `evaluate_new_category.py`.
- Scan the first 100,000 source records, normalize whitespace, reject texts shorter than 20 characters and duplicate texts, including any matching the 9,000 beauty training review texts when the separate training CSV is present.
- Map stars to proxy sentiment labels: 1–2 negative, 3 neutral, 4–5 positive. With seed 42, reservoir-sample 500 usable reviews per class from the scanned prefix; 1,500 total.
- Run the saved BiLSTM and tokenizer without further training. Compare its three-class predictions to the rating-derived labels. No phone reviews were used to choose the model.

Reproduce from the project folder after installing `requirements.txt` and placing the separate beauty training CSV at `data/amazon_beauty_balanced.csv`:

```bash
python evaluate_new_category.py
```

The script defaults to the original model under `artifacts/beauty_baseline/`. Pass `--artifact-dir artifacts` to evaluate the current retrained model separately and choose a different `--output` path so the historical result is preserved.

The exact machine-readable result is `artifacts/phone_category_evaluation.json`.

## Measured result

| Dataset | Evaluation reviews | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| All Beauty, held-out test split | 1,350 | 67.4% | 0.675 |
| Phone/accessory, new category | 1,500 | 63.4% | 0.639 |

Phone/accessory confusion matrix, with true labels in rows and predictions in columns:

| True / predicted | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: |
| Negative | 328 | 151 | 21 |
| Neutral | 132 | 305 | 63 |
| Positive | 48 | 134 | 318 |

The model classified 951 of 1,500 phone/accessory reviews in agreement with their star-derived labels. Neutral was the most frequently confused class, with 305 of 500 correct. Its phone sample scores are lower than its beauty held-out scores.

## Interpretation and limits

This supports a narrow conclusion: the saved model has some useful transfer to these sampled phone/accessory reviews, though its measured performance here is modest. It is not proof that it will perform equally well on any uploaded phone dataset or reliably explain product aspects.

The two test sets differ in category and sampling method, so subtracting their scores does not isolate a causal category effect. Sampling only the first 100,000 source records may reflect source order. The 1,500-review evaluation is balanced by class, whereas real uploads may have a very different rating mix. Stars are proxy labels: a 3-star review can express strong praise and a complaint together, and its text may disagree with the rating. The basic beauty training split was random by review, not held out by product. The script excludes exact normalized review-text overlap with the beauty training CSV when that separate file is present; it does not guarantee no related products or near-duplicates.

Next, if time permits, test on a representative held-out sample and manually check a small set of text sentiments. Evaluate aspect extraction and polarity separately against human-labeled examples before making claims about reliable aspect insights.
