# Experimental restaurant sentiment model

The CSV demonstration showed that the product-trained short-review BiLSTM makes substantial mistakes on restaurant sentences. We fine-tuned a separate **restaurant candidate**, retained the existing product models, and added it as a visible choice when `Restaurants` is selected in the app. It is experimental because the training file supplies human **aspect** polarities, not independently judged overall sentence sentiment.

## Data and label meaning

Source: SemEval 2014 Task 4, `Restaurants_Train_v2.xml`. For a sentence with one or more annotated aspect terms, we use its shared polarity as a **proxy sentence label** only if every aspect term has the same positive, negative, or neutral label. Sentences with mixed or conflict aspects or no annotated aspect term are excluded. This does not turn the proxy into ground truth for the overall meaning: a sentence may contain other opinions outside the annotated terms. After deduplicating case-insensitive text, the training script creates a stratified 70/15/15 restaurant split with seed 45. The script records all exclusion counts and splits in `artifacts/restaurant_candidate/evaluation.json`.

The model starts from the saved short-review tuned BiLSTM and retains its tokenizer. Each epoch combines restaurant training sentences (repeated three times) with a replay of 700 original product training reviews per class. We tested three epochs at learning rate 5e-5. Epoch 3 was chosen by restaurant **validation macro F1**, subject to the four-category product validation macro F1 dropping by at most 0.025 and neutral recall by at most 0.05. The restaurant and product tests were used only after selection. The product default checkpoint is unchanged.

| Set | Prior model accuracy | Restaurant candidate accuracy | Prior macro F1 | Candidate macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Restaurant proxy validation | 58.8% | 71.6% | 0.472 | 0.570 |
| Restaurant proxy test | 60.7% | 74.2% | 0.484 | 0.605 |
| Four-category product validation | 67.53% | 66.91% | 0.673 | 0.666 |
| Four-category product test | 67.59% | 66.83% | 0.673 | 0.665 |

On the restaurant proxy test, **neutral recall is only 33.3%** (12 of 36). The model has a strong positive bias on the 800-sentence restaurant Phase B upload: it predicts 546 positive, 182 negative, and 72 neutral. That file has aspect-term spans but no polarity labels, so this distribution cannot be called accuracy. It correctly calls the bread and the decent Japanese food examples positive but still calls “Did I mention that the coffee is OUTSTANDING?” negative. The aspect vocabulary also lacks Coffee. Mixed sentences were excluded from restaurant training, so do not assume reliable overall labels for balanced praise and criticism.

**Decision:** expose the candidate in the sidebar as `Experimental restaurant BiLSTM` when `Restaurants` is selected, with a warning explaining the proxy labels. Users can compare the short-review product BiLSTM or 50,000-review model using the same upload. In the model-comparison tab, the restaurant proxy test is labeled separately from the architecture comparison. This result is not proof of accuracy on arbitrary restaurants, full customer reviews, or all categories. A larger independently labeled restaurant **overall-sentiment** test, including mixed reviews, is needed before claiming deployment reliability.

**Independent follow-up (22 October 2026):** On 996 unique UCI Yelp sentences with positive/negative human labels, the short-review model scored **75.6%** and the experimental restaurant model **72.5%**. Negative recall fell from **77.1%** to **57.3%** with the restaurant candidate. Therefore the experimental checkpoint is no longer the default when Restaurants is selected. See `INDEPENDENT_YELP_EVALUATION.md`; neither evaluation establishes Neutral or Mixed accuracy.

Reproduce from the project folder with `python train_restaurant_candidate.py --archive path/to/semeval14.zip` if the original four-category training split is present. The shipped app only needs `app.py` and `artifacts/restaurant_candidate/` from the update ZIP; retraining is optional.
