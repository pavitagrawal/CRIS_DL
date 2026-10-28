# Independent restaurant sentence check

The earlier restaurant fine-tune improved on a held-out part of SemEval, but its sentence labels were proxies derived from unanimous **aspect** labels. We therefore compared the saved models on a different source: the Yelp subset of the [UCI Sentiment Labelled Sentences dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), distributed under CC BY 4.0. The source gives 500 human-labeled positive and 500 negative sentences, and no neutral or mixed class.

Four duplicated sentences were removed after confirming their repeated labels agree. We evaluated the remaining **996 unique sentences** (497 Negative, 499 Positive). Case-folded whitespace-normalized exact text overlap with the SemEval restaurant training examples and the original 35,000-product training split was **zero**. Both checkpoints were already selected before this source was opened. Neutral model predictions count as errors in accuracy.

| Model | Accuracy | Binary-reference macro F1 | Negative recall | Positive recall | Neutral predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short-review tuned BiLSTM | **75.6%** | **0.786** | **77.1%** | 74.1% | 76 |
| Experimental restaurant BiLSTM | 72.5% | 0.753 | 57.3% | **87.6%** | 94 |

The restaurant fine-tune is **worse on this independent binary test**, especially for negative sentences. Its distribution shifted toward Positive. The earlier SemEval proxy test improvement (60.7% to 74.2%) did not reliably transfer to this separately labeled source. We keep the checkpoint available for comparison, but the app now defaults to the short-review tuned model when `Restaurants` is selected. The app's model-comparison tab includes the independent numbers and cautions.

Neither result measures **Neutral** or **Mixed** accuracy, full-length restaurant reviews, calibration, or performance on arbitrary customer datasets. The 800-sentence restaurant CSV previously uploaded has no overall sentiment labels, so its displayed positive percentage is a prediction distribution, not ground truth. We should not retrain to fit these Yelp test examples and then reuse this same set as independent confirmation.

Reproduce with `python evaluate_yelp_restaurant.py --source path/to/yelp_labelled.txt --archive path/to/semeval14.zip --product-training path/to/data/amazon_50k_multicategory/train.csv` from the project folder. Results, including class confusion counts and example errors, are in `artifacts/independent_yelp_evaluation.json`.
