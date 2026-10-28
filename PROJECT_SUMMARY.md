# Project summary to share with the team

## Goal

Build a reusable customer review analysis app. It accepts either one review or a new, unlabeled CSV, predicts positive/neutral/negative sentiment, highlights common product aspects and their sentiment, and shows aggregate results. The four neural network models are trained and compared once on a separate labeled dataset. Users of the completed app do not retrain a model when they upload new reviews.

## Inputs and outputs

| Stage | Input | Output |
| --- | --- | --- |
| Training | CSV with review text and a sentiment label, or 1–5 rating | Four trained model files, validation comparison, selected model, held-out test metrics and confusion matrix |
| Single review | Text entered in the app | Overall model sentiment and score; separate aspect summary; recognized aspect mentions with clause sentiment and the matching text |
| Dataset analysis | Unlabeled CSV; user chooses review column | Review count, three-class distribution, sentiment by aspect mention, separate aspect summaries, charts and downloadable review-level predictions |

## Method

1. Clean whitespace, tokenize reviews, pad to a fixed length and learn word embeddings.
2. Train Simple RNN, LSTM, GRU and bidirectional LSTM using the same train/validation/test split and shared training settings.
3. Compare validation macro F1 (plus accuracy, macro precision/recall and training time). Select the winner using validation data. Evaluate the selected model once on the held-out test set.
4. Save the model and tokenizer. Run them on new reviews in batches.
5. Split reviews into short clauses, match a configurable aspect keyword dictionary, and classify each matched clause with the chosen sentiment model. Aggregate positive/neutral/negative **mention counts** per aspect.

This last step is a transparent baseline for aspect insights. It does not train or validate a dedicated aspect-based sentiment model. For the project report, describe the aspect method separately from the four-model sentiment comparison.

## Expected demonstration

Single review input: “The camera is excellent but the battery drains quickly.” The interface shows one overall sentiment, then a Camera mention and a Battery mention with the two matching clauses. The predicted labels and scores are learned from the actual dataset, so they cannot be promised in advance. The overall class might be positive, negative or neutral.

CSV demonstration: upload `sample_upload.csv` or any CSV with a text column and select that column. The screen shows a count of nonempty reviews, sentiment percentages and chart, aspect counts and chart, most positive/negative aspect mention totals, and a downloadable file with columns `predicted_sentiment`, `model_score` and `aspect_mentions`. If there are 5,000 nonempty input reviews, the count will read 5,000. Values such as 63% positive or 91.3% model accuracy are examples only until training and evaluation have been run.

## First measured result

On the 9,000-review balanced All Beauty sample, BiLSTM had the highest validation macro F1 (0.693). Its held-out test accuracy was 67.4%, with macro F1 0.675. These numbers are for rating-derived sentiment labels in the beauty category. The bundle retains this original model and its four-model comparison under `artifacts/beauty_baseline/`; the current app runs the later retrained model described below. The clear phrase “battery drains quickly” receives a **negative** aspect label through a transparent, narrow rule; aspect polarity still needs broader work.

An exploratory follow-up used human-labeled SemEval laptop aspects. The original keyword detector recognized 25.4% of labeled aspect mentions in the full annotated file. A separate target-marked BiLSTM did not improve validation macro F1 on detectable aspects, so it is **not deployed**. We then expanded the app's keyword groups using the training split's terms. On the held-out laptop split, span-level term recall rose from **25.9% to 59.6%**, with measured mention precision **89.6%** for the expanded detector. Among detected terms with a containing clause, sentiment accuracy was initially **60.6%**; detection and correct sentiment together covered **36.1% of all gold terms**. Conservative explicit sentiment cues then raised covered-set polarity accuracy from **56.2% to 58.1%** on laptop validation, and from **60.6% to 62.1%** on the previously inspected test. The app now marks cue-based labels without a model score and correctly handles “keyboard is comfortable” as positive. This laptop test had already been inspected during earlier experiments, so these follow-up figures are exploratory rather than a fresh independent benchmark. See `ASPECT_EVALUATION.md` for the method and limits.

We then ran the **same saved BiLSTM**, without retraining, on 1,500 balanced phone/accessory reviews from a bounded prefix of another Amazon Reviews 2023 category. Accuracy was **63.4%** and macro F1 **0.639** against star-derived labels. The beauty test was 67.4% and 0.675. Different sampling methods, source ordering, and rating proxy labels limit this comparison; the phone result is an exploratory overall sentiment test, not a validation of aspect analysis or of every new CSV. See `CROSS_CATEGORY_EVALUATION.md`.

We subsequently retrained the BiLSTM on **6,300 beauty + 7,200 phone** training reviews, keeping category-specific validation and test sets separate. Beauty test accuracy rose from **67.4% to 70.0%**, and phone test accuracy from **63.4% to 65.9%**. We then trained a new BiLSTM on **43,497 beauty and phone reviews**; this model was then fine tuned with appliance reviews. It reached **71.3%** beauty test accuracy and **69.1%** phone test accuracy, with macro F1 **0.715** and **0.695**. The phone test was previously inspected, and both categories use star-derived labels. On 50 AI-drafted beauty labels, agreement fell from 78% to 76%; the draft is not human ground truth. The bright-screen/fast-performance example still predicts **neutral** overall, an error for that positive text. See `MULTIDOMAIN_EVALUATION.md` and `LARGER_TRAINING_EVALUATION.md`.

The app now shows **Aspect summary: Positive** for that bright-screen example, alongside **Overall sentiment (model): Neutral**. A candidate rule that would force the overall label to positive lacked validation support and harmed one inspected beauty test case, so it is not deployed. The summary is derived from aspect labels and kept separate from the BiLSTM output.

## Team deliverables

- Source code and dependency list
- Labeled training dataset with provenance and a description of how labels were obtained
- Saved model and tokenizer produced by training
- Four-model validation comparison and selected model's test confusion matrix
- App demo on a separate, unlabeled review CSV
- Report discussing results and limits, especially product domain shift and the aspect heuristic
- Exploratory cross-category evaluation script and measured phone/accessory result

## Chosen training source

McAuley Lab's [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), **All_Beauty** category. `prepare_dataset.py` streamed the official review file, scanned 701,528 rows, and sampled 3,000 usable reviews per mapped class, producing 9,000 balanced rows with seed 42 and a provenance/counts summary. Ratings 1–2 become negative, 3 neutral, and 4–5 positive. These are rating-derived proxy labels. The prepared CSV is supplied separately from the source zip.

## How to run

To see the interface now, extract the zip and double-click `demo.html`. Its clearly marked outputs use simple word rules and do not represent the neural networks. For the real app, follow `README.md`: create a Python environment, install `requirements.txt`, and run `python -m streamlit run app.py`. The selected trained model is included. To retrain, run `train.py` with the separately provided labeled CSV and correct column names. TensorFlow is required for both training and inference.

## Current extension

The app now uses the appliance-extended BiLSTM and supports explicit Sound cues for noise complaints. On star-labeled exploratory tests, current accuracy is 71.56% beauty, 71.80% phone/accessories and 65.93% appliances; the independent UCI binary-reference check is 78.0% (neutral outputs count as errors). The mixer noise example remains neutral overall but shows a negative Sound aspect. See `APPLIANCE_EVALUATION.md`. No general accuracy claim across arbitrary categories follows from these sets.

## Latest model update

The selected BiLSTM was fine tuned on a train split of human-labeled short UCI product-review sentences plus replayed beauty/phone/appliance training reviews. On the exploratory 153-sentence UCI test split, accuracy rose from 79.7% to 85.0%; phone test accuracy fell from 71.8% to 71.1%. The trained model now calls “The mixer is noisy” negative without an overall rule, though other short reviews still fail. See `SHORT_REVIEW_EVALUATION.md`.

## Submitted architecture comparison

The submitted synopsis specifies MLP, TextCNN, BiLSTM and Transformer. These four architectures are compared on one 9,000-review beauty split, with saved metrics and checkpoints; BiLSTM wins validation macro F1. This is distinct from the earlier RNN/LSTM/GRU/BiLSTM exploration and from the later fine-tuned PyTorch BiLSTM served by the app. The comparison does not reproduce the interim report's unidentified original 50,000-row sample. See `SUBMITTED_MODEL_COMPARISON.md`.

## Prepared 50,000-review benchmark

A newly sampled, separately packaged Amazon Reviews 2023 dataset covers All Beauty, Books, Electronics and Home & Kitchen with 12,500 reviews per category and saved 70/15/15 category-and-sentiment stratified splits. This is not the exact interim-report sample. All four submitted architectures were trained from scratch on these saved splits and evaluated on the held-out test set. See `SUBMITTED_50K_EVALUATION.md` and the **Model comparison** tab. The app serves a fine-tuned version of the winning PyTorch BiLSTM. Aspect detection remains a separate keyword and cue method.

## Short-review fine-tuning

Following a user evaluation on 50 public product sentences, we fine-tuned the 50,000-review BiLSTM on different UCI sentences with balanced replay from its original training split. On a fresh 100-sentence UCI test, accuracy rose from 78% to 85%; on the already inspected user sample it rose from 82% to 88%. Four-category test macro F1 fell from 0.6767 to 0.6726, and neutral recall fell from 55.36% to 51.44%. The current app serves the later Yelp-tuned continuation of this model. See `SHORT_REVIEW_FT_EVALUATION.md`.
