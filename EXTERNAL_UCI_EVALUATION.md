# Historical full UCI evaluation before short-review training

# Independent public review check

Source: UCI Machine Learning Repository, **Sentiment Labelled Sentences** (Kotzias et al., 2015), Amazon product-review sentence subset, https://doi.org/10.24432/C57604 . The source dataset is CC BY 4.0. The official archive `sentiment labelled sentences/amazon_cells_labelled.txt` has 1,000 sentences: 500 labeled negative (0) and 500 positive (1). Its authors selected clear positive and negative examples and intentionally did not include neutral examples.

We converted that file to `uci_amazon_product_reviews_1000.csv` with `review` and `reference_sentiment` columns. We did not use its labels for model fitting, tokenizer fitting, architecture selection, or tuning. Exact review text overlap with the project's original 9,000 beauty sample and phone training/validation/test splits was zero; there are 990 distinct texts in the UCI file. We ran the then-deployed BiLSTM on all 1,000 texts without retraining.

| UCI label (row) | Predicted negative | Predicted neutral | Predicted positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Negative | 357 | 100 | 43 | 500 |
| Positive | 28 | 49 | 423 | 500 |

**Exact three-class output versus binary reference accuracy: 780/1,000 = 78.0%.** Every neutral prediction is counted as a miss, because this dataset provides no neutral reference label. The historical model predicted 385 negative, 149 neutral, and 466 positive. Among the 851 non-neutral predictions, 780 were correct (91.7%); that conditional figure excludes 149 abstentions and is not overall accuracy.

This is a useful independent sentiment check on short product-review sentences, not a representative measure of accuracy on arbitrary uploaded CSVs or longer mixed reviews. It does not validate aspect extraction. Do not compare its 78% directly with beauty/phone test percentages without noting that the datasets, class makeup, and reference-label methods differ. The predictions file was later refreshed for the current model and now has a `data_role` column. Its train and validation rows are not independent evaluation results. The present model's 153-sentence test split is in `SHORT_REVIEW_EVALUATION.md`.
