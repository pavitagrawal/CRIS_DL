# Independent restaurant rating check

Source: [restaurant_reviews.csv](https://github.com/skathirmani/datasets/blob/main/restaurant_reviews.csv), an English restaurant review collection with review text and 1–5 star ratings. This experiment uses 90 reviews: 30 each from 1–2, 3, and 4–5 stars. The first 1,069 rated source records were eligible, with 39 low-rated records; a deterministic text hash chose 30 from each group without consulting model predictions. No exact normalized text overlap was found with the available product and SemEval restaurant training sources.

The labels in this experiment are **rating proxies**: 1–2 stars → Negative, 3 stars → Neutral, 4–5 stars → Positive. They are not manually judged sentence sentiment. Three-star text may itself be strongly positive, negative, or mixed.

| Saved model | Rating-proxy agreement | Macro F1 | Negative recall | Neutral recall | Positive recall |
|---|---:|---:|---:|---:|---:|
| Short-review tuned BiLSTM (current default) | 61.1% | 0.609 | 70.0% | 43.3% | 70.0% |
| Experimental restaurant BiLSTM | 58.9% | 0.573 | 56.7% | 33.3% | 86.7% |

The current default remains the better choice on this sample and on the separate binary UCI Yelp test. Neutral is still weak. This small, balanced, single-source check does **not** establish accuracy on all restaurant reviews or all categories. The models were trained before selecting these test reviews, and this check did not retrain either model.

Run from the project root after the saved checkpoints are present:

```powershell
.\.venv\Scripts\python.exe evaluate_restaurant_rating_proxy.py
```

The sample and full metrics are in `data/restaurant_rating_proxy_90.csv` and `artifacts/restaurant_rating_proxy_evaluation.json`. The script checks for exact overlap against `data/amazon_50k_multicategory/train.csv` and the original SemEval archive if available. If either source is missing, the JSON marks the overlap check incomplete.

Next useful improvement: gather a larger, independently human-labeled three-class restaurant test set, especially genuine neutral and mixed reviews. Report that separately before changing the default or making a general-accuracy claim.
