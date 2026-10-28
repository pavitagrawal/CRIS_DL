# AI-Based Customer Review Intelligence System

**Start here:** [FINAL_PROJECT_GUIDE.md](FINAL_PROJECT_GUIDE.md) describes the current one-model app, demo, datasets, and measured limitations. This README also records earlier experiments.

This working tree combines the team repository at `pavitagrawal/CRIS_DL` with the
stronger local application and evaluation work. The original PyTorch layout remains in
`models/`, `utils/`, and `notebooks/`; the production-style Streamlit workflow, curated
runtime checkpoint, aspect evidence, review queue, and measured evaluations are kept
alongside it. The canonical model classes now live only in `models/`, including fixes
that prevent padded tokens from influencing MLP pooling, BiLSTM attention, and
Transformer attention.

For a clean Windows setup, run `.\scripts\bootstrap.ps1`; before committing, run
`.\scripts\check.ps1` and follow [CONTRIBUTING.md](CONTRIBUTING.md). This repository
does not commit or push automatically.

**Phase 3 deliverables:** [FINAL_COMPLETION_AUDIT.md](FINAL_COMPLETION_AUDIT.md) maps the interim roadmap to the final evidence, and [FINAL_ARCHITECTURE_ANALYSIS.md](FINAL_ARCHITECTURE_ANALYSIS.md) consolidates the four-model metrics and class-level errors. Submission-ready report and presentation files are under `docs/final/`.

This project began with four recurrent networks for three-class sentiment, then ran a larger four-architecture experiment. The app runs one saved fine-tuned BiLSTM on a manually entered review or a new CSV. The dashboard shows overall sentiment, keyword matched aspect mentions, charts, and a downloadable annotated CSV.

## What it builds

1. Train RNN, LSTM, GRU, and BiLSTM on the same labeled review dataset and stratified 70/15/15 train/validation/test split.
2. Tokenize training text, pad sequences, and learn word embeddings in each network. Choose the model with the highest **validation macro F1**. Report its held-out test metrics and confusion matrix.
3. Save the winning model and tokenizer. Analyze new reviews without labels or retraining.
4. For aspects, match a configurable set of keywords to short clauses and classify each matched clause with the winning sentiment model. This is a heuristic, **not** a trained aspect-based sentiment model. The expanded dictionary has measured term-detection coverage on laptop reviews; its polarity remains imperfect. The dashboard displays the evidence and counts mentions.

## Data needed

### Chosen public dataset for this project

We use McAuley Lab's [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), specifically the **All_Beauty** review file. Its records include review `text`, 1–5 `rating`, and product ID. The official compressed file was processed successfully: **701,528** rows scanned; after removing short, empty, and duplicate texts, **133,565 negative**, **52,374 neutral**, and **429,546 positive** usable rows remained. `prepare_dataset.py` samples **3,000 per class** with seed 42, producing 9,000 unique rows. It writes `data/amazon_beauty_balanced.csv` and a counts/provenance summary. The generated CSV is provided separately from the source zip for this project. Cite McAuley Lab in the report and consult the source terms before wider distribution.

```bash
python prepare_dataset.py
python train.py --data data/amazon_beauty_balanced.csv
python -m streamlit run app.py
```

The downloaded category and training can take time. You can first use `--per-class 500` for a quicker pipeline check, then run the default sample for the reported experiment. The held-out split in `train.py` is a random stratified review split. We also ran a separate exploratory phone/accessory evaluation below; an arbitrary electronics CSV may differ from that sample.

You may instead supply your own dataset:

Supply a labeled **training** CSV. Default columns are `review` and `sentiment`; sentiment must be `negative`, `neutral`, or `positive`. Each class needs at least 10 usable examples for the splits to work, but a meaningful project needs many more balanced, high-quality examples. Alternatively provide a numeric 1–5 rating column and use `--rating-labels`: 1–2 maps to negative, 3 to neutral, 4–5 to positive. Explain in the report that this creates labels from ratings, which may not perfectly reflect review text.

Example training CSV:

```csv
review,sentiment
"The camera is great",positive
"Battery life is terrible",negative
"It works as described",neutral
```

The above three rows illustrate the format; they are **not enough** to train. Keep the training/test data separate from any final uploaded demonstration dataset. Avoid duplicates and product overlap between splits if your evaluation aims to measure transfer to new product categories; the basic random split does not guarantee that.

## Setup and run

To preview the user interface immediately, double-click `demo.html` in the extracted folder. Select **Single review** and click **Analyze review**, or select **Upload CSV** then **Use included sample reviews**. This standalone browser preview uses simple word rules; it does not load the trained models or provide research results.

For the actual neural-network application, use the steps below.

Use Python 3.10–3.12 in a fresh environment with a TensorFlow compatible platform. On Windows, running in WSL2 or Google Colab is often simplest for training. From the project directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --data path/to/labeled_reviews.csv --text-column review --label-column sentiment
python -m streamlit run app.py
```

For ratings: `python train.py --data path/to/reviews.csv --text-column text --label-column stars --rating-labels`.

The current project bundle includes the short-review fine-tuned `artifacts/best_model.keras`, tokenizer, metadata, and evaluation files. It also includes the original beauty-trained model and its historical four-model comparison under `artifacts/beauty_baseline/`. To run the real app immediately, install the dependencies and run `python -m streamlit run app.py` from this directory. The beauty training CSV is a separate download. If training in Colab, bring the generated `artifacts` folder beside `app.py`. Use the **same dependency versions** for training and serving when possible.

### First measured comparison

This first run used the 9,000-row balanced All Beauty sample, seed 42, a stratified 70/15/15 review split (6,300/1,350/1,350), 5 maximum epochs with early stopping, batch size 128, 15,000-token vocabulary and sequence length 120. It ran on CPU. The highest **validation macro F1** selected BiLSTM. The selected model scored **67.4% accuracy and 0.675 macro F1** on the untouched test split. These results measure rating-derived labels within this beauty sample; they do not establish performance on electronics reviews or aspect sentiment.

| Model | Validation accuracy | Validation macro F1 | Training time |
| --- | ---: | ---: | ---: |
| RNN | 62.3% | 0.626 | 9.89 s |
| LSTM | 64.2% | 0.638 | 16.15 s |
| GRU | 66.0% | 0.658 | 15.17 s |
| BiLSTM | 69.1% | 0.693 | 26.85 s |

Times reflect this CPU environment and are not directly comparable to other machines. The saved artifacts contain the exact unrounded metrics and confusion matrix. On “The keyboard is comfortable but the battery drains quickly,” the app returns overall neutral, Keyboard positive, and Battery negative. The keyboard label comes from an explicit sentiment cue and the battery label from a narrow battery phrase rule; neither is a newly trained aspect model. The app identifies which method produced each aspect label.

### Updated overall sentiment model

We first retrained the selected BiLSTM on 6,300 beauty and 7,200 phone/accessory training reviews. Beauty test accuracy rose **67.4% → 70.0%**, and the previously inspected phone test rose **63.4% → 65.9%**. We then trained a new BiLSTM on 43,497 beauty and phone reviews. It scored **71.3%** on beauty and **69.1%** on phone, with macro F1 **0.715** and **0.695**. That larger-data model was subsequently fine tuned with appliance reviews. The positive screen/performance example is still incorrectly neutral. On 50 AI-drafted labels, agreement slipped from 78% to 76%; that draft is not human ground truth. See `LARGER_TRAINING_EVALUATION.md` and `MULTIDOMAIN_EVALUATION.md` for methods and limits. The original four-model comparison remains the architecture selection experiment.

### First test on a new product category

Without retraining, we evaluated the saved BiLSTM on **1,500 phone/accessory reviews** (500 per rating-derived class) from the first 100,000 rows of McAuley Lab's `Cell_Phones_and_Accessories` category. Accuracy was **63.4%** and macro F1 was **0.639**. The earlier beauty test scored 67.4% and 0.675, respectively. These are different samples, so the gap is descriptive rather than a controlled estimate of the effect of changing categories. Neutral reviews were hardest in this phone sample (305 of 500 correctly identified). See `CROSS_CATEGORY_EVALUATION.md` and `artifacts/phone_category_evaluation.json` for the method and full counts.

To reproduce this optional evaluation after downloading the separate beauty training CSV, run `python evaluate_new_category.py`. It reads an official public compressed review file, samples within a bounded prefix, and tests **overall sentiment only**. It does not change the app or retrain the model.

### Short-review model update

The current BiLSTM was fine tuned with 697 human-labeled UCI Amazon review sentences (and replayed beauty, phone and appliance training data). It scored **130/153 = 85.0%** against a UCI test split kept out of this fine tuning run, compared with 79.7% for the previous model on those same sentences. Phone test accuracy decreased from 71.8% to 71.1%; beauty and appliance figures changed slightly. The old model had been evaluated on all 1,000 UCI sentences before the split, so this comparison is **exploratory**, not a fresh independent benchmark. The model itself now calls “The mixer is noisy” negative; “The mixer makes a very loud noise” remains neutral. The aspect analyzer is unchanged. See `SHORT_REVIEW_EVALUATION.md` and the **Model comparison** tab for details. The original four-model experiment remains the architecture comparison.

Upload `data/uci_amazon_product_reviews_1000.csv` and select `review` to demonstrate CSV predictions. Its 1,000 rows include examples used for training and validation, so do not present its full-upload accuracy as an independent test. The app dashboard reports predicted shares, not accuracy.

## Expected app output

**Single review:** enter “The camera is excellent but the battery drains quickly.” The app displays an overall model class and score, a separate summary of detected aspect labels, and matched clauses for Camera and Battery. Actual classes and scores depend on the trained model. A mixed review is **not guaranteed** to be neutral.

**CSV mode:** upload a new CSV, select the review column, and click Analyze dataset. For each nonempty review, the app predicts sentiment; it shows total analyzed, positive/neutral/negative percentages, a distribution chart, aspect mention counts, most positive and negative mentions, and a downloadable CSV with `predicted_sentiment`, `aspect_summary`, and other results. The aspect summary does not alter overall model predictions. Any example percentages such as 63/20/17 are mockups, not measured results.

### Step: review questionable predictions

After analyzing a CSV, open **Human review queue**. The app flags reviews if its top model score is below 60%, detected aspects have mixed labels, or the aspect summary disagrees with the overall prediction. These are provisional rules for choosing what to inspect, **not** a tested guarantee that the other reviews are correct. The score is uncalibrated.

The queue also flags a **neutral** model prediction when its review contains a narrow explicit complaint such as “I hate it,” “I dislike that,” “would not recommend,” or “will not be reordering.” It does not change the model's neutral answer. In an exploratory audit of 120 previously held-out beauty reviews, this added three flags (30 to 33 total). On 50 of those reviews labeled by an AI draft, it increased flagged model/draft disagreements from 5 of 11 to 8 of 11, while leaving 3 disagreements unflagged. The rule was chosen after inspecting this sample, so those figures are **not** an independent performance estimate. AI draft labels and star-rating labels are imperfect substitutes for human judgments.

Choose a label in **Your sentiment label** for any flagged review you read. Navigate queue pages to handle more than 100 flags. Download **predictions and human corrections** before closing the browser: edits are kept only in the current browser session. The export contains `predicted_sentiment` (original model output), `human_sentiment` (your correction, blank if none), `final_sentiment` (human label if supplied; otherwise model label), and `label_source` (human or model). The dashboard percentages still describe raw model predictions; human corrections do not retrain the model or change those charts.

Try the included `sample_upload.csv`: the bright-screen review should be flagged if the current model gives its known neutral result while detected screen/performance aspects are positive. Select **positive** as your human label, download the CSV, and compare the `predicted_sentiment` and `final_sentiment` columns. Outputs can vary if you replace the trained model. A few corrected examples are a start for collecting human labels, not evidence of general accuracy.

### Step: check accuracy against human judgments

Upload an independent CSV of reviews, click **Analyze dataset**, and open **Evaluate model with your own labels**. The app draws a fixed random sample of up to 50 reviews from the whole upload, including reviews that the queue did not flag. Read the review text and select a negative, neutral, or positive label for each. Model predictions are hidden in this editor. You can download the partially completed evaluation CSV at any time; the evaluation choices are kept in the browser session until you download them.

Once every sampled review has a label, the app displays accuracy, macro F1, a confusion matrix (human labels in rows), and counts of model errors that the queue did and did not flag. With the four-row demonstration file, this only checks that the workflow runs; use a genuinely new, larger dataset and label the full 50-review sample for an initial exploratory estimate. Carefully judge mixed reviews, and keep this evaluation set out of later retraining if you want an independent check. A 50-review sample still has considerable statistical uncertainty and may not represent other product categories.

**Model comparison:** view the original four-model validation comparison, then the earlier retraining comparison and the new same-sample comparison of previous and deployed BiLSTMs, with deployed-model test confusion matrices.

## Known limits and next steps

- The saved model depends on representative labeled data. Domain shift, sarcasm, short fragments, and mixed opinions can reduce accuracy.
- On the four-row sample CSV, the larger-data overall model calls “The screen is bright and performance is fast” neutral, even though both aspect clauses are positive. This remains a visible overall-sentiment error.
- The displayed softmax score is **not calibrated probability**. No 91% confidence claim should be made without calibration.
- The review queue is an unvalidated triage aid; low scores and aspect disagreement can include correct model predictions, while other wrong predictions may go unflagged. Review a representative random sample too before using aggregate results for decisions.
- Aspect coverage is limited to the keyword dictionary in `review_intelligence.py`. The expanded dictionary detects 59.6% of annotated laptop aspect terms on the held-out split with 89.6% measured mention precision. Clause classification can be wrong when a clause mentions two aspects or lacks context; polarity accuracy on the expanded covered laptop terms was 60.6% on that split. See `ASPECT_EVALUATION.md` for the exact denominators and limits.
- A conservative battery-life phrase rule now handles clear expressions such as “battery drains quickly” and “battery lasts all day.” It abstains on ambiguous wording and does not make the aspect system generally reliable.
- `ASPECT_EVALUATION.md` reports the first laptop baseline, the unshipped aspect-specific BiLSTM prototype, and the expanded keyword detector's measured coverage and polarity. Conservative explicit cues raised covered-set polarity accuracy from 56.2% to 58.1% on laptop validation; they are not a general solution.
- The first phone/accessory evaluation is exploratory. For stronger cross-category evidence, use a predeclared, representative category sample and human-checked sentiment labels; for within-category generalization, also consider product-held-out splits.

## Project structure

| File | Purpose |
| --- | --- |
| `train.py` | Load labels, train four models, compare, and save artifacts |
| `review_intelligence.py` | Inference and explainable keyword based aspect method |
| `review_queue.py` | Provisional review flags and attribution of human corrections |
| `human_evaluation.py` | Random sample selection and metrics against user-entered labels |
| `aspect_polarity.py` | Narrow positive/negative cue and negation handling for single-aspect clauses |
| `app.py` | Streamlit single review, CSV, and comparison dashboard |
| `sample_upload.csv` | Unlabeled CSV for testing the upload flow after training |
| `demo.html` | Standalone browser preview with clearly labeled rule-based outputs |
| `prepare_dataset.py` | Streams a reproducible balanced sample of Amazon Reviews 2023 All Beauty |
| `train_aspect.py` | Trains an exploratory model on human-labeled laptop aspects |
| `ASPECT_EVALUATION.md` | Measured limitations and decision for the aspect component |
| `evaluate_aspect_extraction.py` | Compare original and expanded aspect detection and clause polarity on annotated laptops |
| `evaluate_new_category.py` | Reproducible exploratory overall sentiment test on phone/accessory reviews |
| `CROSS_CATEGORY_EVALUATION.md` | Phone/accessory result, confusion matrix, and limitations |
| `train_multidomain.py` | Retrain BiLSTM on beauty and phone data with separate validation/test samples |
| `MULTIDOMAIN_EVALUATION.md` | Earlier same-sample model comparison and limitations |
| `LARGER_TRAINING_EVALUATION.md` | Larger-data retraining results and limitations |
| `evaluate_overall_rule.py` | Exploratory check of a candidate aspect agreement override, not deployed |
| `tests/test_logic.py` | Lightweight checks for labels, clauses, and metrics |
| `models/` | Canonical MLP, TextCNN, BiLSTM, and Transformer definitions plus PyTorch trainers |
| `utils/` | Shared CRIS_DL preprocessing and result-comparison utilities |
| `notebooks/` | Team experiment notebooks from CRIS_DL |
| `scripts/` | Windows environment bootstrap and pre-commit test helpers |
| `CONTRIBUTING.md` | Safe local workflow for staging and making your own commits |

### Appliance extension

The current BiLSTM also trained on 7,115 star-labeled appliance reviews. Its exploratory appliance test accuracy is **65.93%** (macro F1 **0.66059**); the previous model scored 65.73% (macro F1 0.66134). The Sound aspect rules now show a negative complaint for “very loud noise,” while the overall mixer sentence remains neutral. Try `appliance_reviews_300_demo.csv` through **Upload CSV**, select `review`, and see `APPLIANCE_EVALUATION.md` for the split and limitations.

The earlier appliance extension results in `APPLIANCE_EVALUATION.md` are historical. The current model metrics and its tradeoffs appear in `SHORT_REVIEW_EVALUATION.md`.

### Alignment with the submitted four-model plan

The **Model comparison** tab includes the earlier shared-split beauty experiment for the submitted **MLP, TextCNN, BiLSTM and attention-based Transformer** architectures. The team model classes, comparison runner, saved checkpoints, beauty CSV, and detailed results are included. BiLSTM wins this earlier experiment by validation macro F1. The RNN/LSTM/GRU/BiLSTM table is historical exploration. The app serves one fine-tuned PyTorch BiLSTM; the historical architecture checkpoints and results remain for research. See `SUBMITTED_MODEL_COMPARISON.md` for the preliminary protocol and `SUBMITTED_50K_EVALUATION.md` for the new one.

### New 50,000-review multi-category sample

`prepare_multicategory_dataset.py` reproducibly streams official McAuley Lab Amazon Reviews 2023 category files for All Beauty, Books, Electronics and Home & Kitchen. A separately downloadable data bundle contains 50,000 sampled reviews and fixed 35,000/7,500/7,500 train/validation/test files plus `provenance.json`. It is **a newly created sample, not the unprovided exact dataset behind the interim report's preliminary numbers**. We trained all four submitted architectures from scratch on these fixed splits. The new comparison appears in the **Model comparison** tab and `SUBMITTED_50K_EVALUATION.md`. The split is review-level with category-and-class stratification; the same product can occur in multiple splits. Rating-derived labels are proxies for the text's sentiment.

### Current single-model application

When **Restaurants** is selected, a short review with one clause, one recognized restaurant aspect, and one clear uncontradicted sentiment cue may use that explicit cue for the displayed overall sentiment. The app reports the underlying BiLSTM label and score and flags a changed label for human review. This rule fixes “The food was tasty” and leaves “The food was tasty but the service was terrible” with the model. On 200 held-out Yelp sentences it changed one prediction, correcting it; on the 90 restaurant rating reviews it changed none. These sets were already inspected during development, so this is a targeted check, not proof of broad accuracy improvement. Product reviews are unaffected. See `artifacts/restaurant_explicit_cue_evaluation.json`.

The app serves one saved BiLSTM checkpoint for **Single review** and **Upload CSV**. Install `requirements.txt` in the same Python environment used to run Streamlit, then start `python -m streamlit run app.py`. The category selector changes the aspect vocabulary, not the overall sentiment model. Aspect rules are separate from the trained sentiment model.

The current checkpoint continues the short-review-tuned BiLSTM, further fine-tuned on 597 deduplicated human-labeled Yelp sentences with 3,600 replayed product training reviews (1,200 in each sentiment class), repeating the Yelp examples four times. The split reserves 199 Yelp sentences for validation and 200 for a held-out test. Epoch selection uses Yelp validation macro F1 subject to product validation macro F1 and neutral-recall guards. It does not use the Yelp test, 90 restaurant rating reviews, or the user's 50 inspected labels to select an epoch. The exact split and replay recipe are in `fine_tune_yelp_existing.py` and `artifacts/restaurant_yelp_candidate/evaluation.json`. The 50,000-review training split is distributed as a separate data bundle.

Held-out Yelp accuracy improved from **71.5% to 77.5%** on 200 binary human-labeled sentences. On 7,500 rating-labeled product reviews, accuracy changed from **67.6% to 67.3%** and neutral recall from **51.4% to 48.4%**. Agreement with 90 restaurant rating proxies improved from **61.1% to 64.4%**, while neutral proxy recall decreased from **43.3% to 40.0%**. The user's previously inspected 50 binary labels changed from **88% to 92%**, an exploratory check. These data do not establish reliable restaurant neutral or mixed-review performance. “The food was tasty” still receives a negative overall prediction; users should inspect the human-review queue and matched aspect evidence. The old short-review tuning comparison is documented in `SHORT_REVIEW_FT_EVALUATION.md`.

To reproduce the new architecture experiment, install `requirements.txt`, put the separate dataset bundle in `data/amazon_50k_multicategory/`, and run `python compare_submitted_models.py --splits-dir data/amazon_50k_multicategory --max-vocab 20000 --max-length 200 --epochs 5 --out artifacts/submitted_50k_comparison`. Merely viewing its saved comparison in the app does not require loading PyTorch models.
