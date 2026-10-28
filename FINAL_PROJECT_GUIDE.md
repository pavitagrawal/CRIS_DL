# Customer Review Intelligence — current project guide

## What the app does

The Streamlit app accepts one English review or an unlabeled CSV. For CSV uploads, the user selects the text column. One saved BiLSTM predicts Negative, Neutral, or Positive. Keyword and phrase rules identify aspect mentions. The dashboard shows sentiment counts, aspect counts, charts, flagged reviews, and a downloadable annotated CSV. Users can label a sample to check errors and correct individual results.

The sidebar selects **Products and electronics** or **Restaurants** for aspect vocabulary. In Restaurants mode, a narrow clear phrase rule can correct the displayed overall label of a short, single-aspect review. A correction is disclosed and flagged for human review. This is still one underlying prediction model. The rule changes one of 200 held-out Yelp predictions (correctly) and none of the 90 restaurant rating examples; it is not a broad restaurant solution.

## Run on Windows

Open PowerShell in the extracted project folder. If the folder already has a working `.venv`, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On a new installation, use Python 3.11 or 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The saved checkpoint is included; retraining is unnecessary to demonstrate the app. The full 50,000-review training CSV is a separate dataset download.

## Demo

1. Choose **Products and electronics** and analyze `The camera is excellent but the battery drains quickly.` Show overall sentiment and aspect evidence. Mixed aspect sentiment can differ from the overall model label.
2. Upload `sample_upload.csv`, select `review`, and click **Analyze dataset**. Show the percentages, aspect table, review queue, and downloadable CSV. The upload needs no sentiment labels.
3. Choose **Restaurants** and enter `The food was tasty`. The app displays Positive and discloses that a clear phrase corrected the model's Negative label. Try `The food was tasty but the service was terrible` to show that the rule abstains on mixed clauses.
4. In **Model comparison**, show the original RNN/LSTM/GRU/BiLSTM experiment and the later four-architecture, 50,000-review experiment. These are historical training comparisons, not alternative live predictors.

## What was trained

The original plan's **RNN, LSTM, GRU, BiLSTM** were trained on the same balanced 9,000-review **Amazon Reviews 2023 All Beauty** sample (3,000 rating-derived labels per class). BiLSTM won the validation macro F1 comparison. Its held-out test accuracy was 67.4%; see `artifacts/model_comparison.csv` and `artifacts/beauty_baseline/`.

Later, to support broader product types, the team also trained **MLP, TextCNN, BiLSTM, and attention-based Transformer** on a new 50,000-review Amazon Reviews 2023 sample from **All Beauty, Books, Electronics, Home & Kitchen**. Fixed stratified splits contain 35,000 train, 7,500 validation, and 7,500 test reviews. That exact 50,000-review sample was newly constructed; it is not the unidentified exact dataset behind the earlier submitted report. BiLSTM won by validation macro F1. The current app serves a later fine-tuned version of that BiLSTM. See `SUBMITTED_50K_EVALUATION.md` and `README.md` for methods and scores.

The checkpoint was subsequently tuned on short product sentences and 597 human-labeled Yelp restaurant sentences with replayed product training reviews. On 200 held-out Yelp sentences it reached 77.5% accuracy; on the 7,500 product test it reached 67.3% accuracy and 48.4% Neutral recall. Ninety restaurant reviews grouped by star rating produced 64.4% agreement; this is a rating proxy, not a human-labeled three-class accuracy result. The phrase rule does not change product predictions. See `artifacts/final_model_evaluation.json` and `artifacts/restaurant_explicit_cue_evaluation.json`.

## Dataset and limits

The app ZIP contains code, saved checkpoints, experiment results, small evaluation and demo files. **It does not contain the full 9,000-review beauty training CSV or the 50,000-review product training CSV.** The 50,000-review splits are available as a separate dataset bundle. Ratings 1–2 map to Negative, 3 to Neutral, and 4–5 to Positive; these labels can disagree with review wording. The app's aspects use keywords and rules, not a separately trained aspect classifier. Restaurants, mixed reviews, neutral reviews, and unseen categories still need human inspection.
