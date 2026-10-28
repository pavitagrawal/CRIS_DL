# Restaurant model-disagreement review queue

The app can optionally run **both saved restaurant-capable BiLSTMs** on an uploaded restaurant CSV. It keeps the selected model's original prediction and dashboard percentages. A disagreement adds a visible **Saved models disagree** reason to the human review queue and includes `alternate_model_sentiment` in the downloadable CSV. It does not automatically change any review's label. The second inference pass makes large uploads slower, so the option starts off.

| Evaluation source | Reviews | Default-model errors | Models disagree | Errors among disagreements | Error rate in flagged set | Overall error rate |
|---|---:|---:|---:|---:|---:|---:|
| UCI Yelp sentences with human Positive/Negative labels | 996 | 243 | 198 | 94 | 47.5% | 24.4% |
| Restaurant star-rating proxy sample, 30 per class | 90 | 35 | 19 | 10 | 52.6% | 38.9% |

The first source is binary: a Neutral prediction counts as an error. The second maps 1–2 stars to Negative, 3 to Neutral, 4–5 to Positive; those ratings are **not human sentiment judgments**. Both sources had already been inspected during development, so these queue results are exploratory. The disagreement flag misses **149/243** default-model errors on Yelp and **25/35** on the rating-proxy sample. It also sends many correct reviews for unnecessary inspection. The combined queue has not been **independently** validated on a new source. See `artifacts/restaurant_model_disagreement_evaluation.json` and `evaluate_model_disagreement.py` for the full counts and reproduction.

We also measured the *combined* queue on these inspected sources. On Yelp, the existing flags marked 215 reviews and caught 114/243 model errors; adding disagreements marked 311 reviews and caught 150/243 errors. The 96 additional reviews contained **36 additional errors**. On the 90 rating-proxy reviews, the combined queue grew from 67 to 68 and caught one additional proxy disagreement (26 versus 25). These are exploratory results on the same sources, not a fresh validation of the combined queue.

Two tempting automatic prediction changes were tested and **not deployed**:

1. Replace a short, single-aspect restaurant review's model label with its explicit aspect cue. This changed 0/90 rating-proxy reviews and 7/996 Yelp sentences, fixing four and harming two, a gain too small to justify overriding the trained model.
2. Average the two models' class probabilities. On the rating-proxy sample, accuracy fell from 61.1% to 58.9% and Negative recall fell from 70.0% to 53.3%. On Yelp, accuracy rose only from 75.6% to 76.2% while Negative recall fell from 77.1% to 71.8%. The tradeoff was not justified.

The current default remains the short-review tuned BiLSTM. For consequential decisions, a person should read the flagged reviews and a random sample of unflagged reviews. Models agreeing is not proof that they are correct.
