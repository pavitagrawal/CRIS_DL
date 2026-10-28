# Retraining overall sentiment with beauty and phone reviews

## Why

The beauty-trained overall BiLSTM incorrectly called “The screen is bright and performance is fast” negative. We tested a candidate BiLSTM trained on two product categories to see whether overall sentiment improves beyond that one example. The aspect detector and its explicit cues are separate; they were not retrained here.

## Data and method

- Reuse the 9,000-review balanced All Beauty sample. Recreate the original seed-42 stratified split: 6,300 beauty training, 1,350 validation, 1,350 test reviews. The test reviews are excluded from candidate training.
- Stream the official McAuley Lab Amazon Reviews 2023 phone/accessory review file. From its first 100,000 source rows, recreate the earlier 1,500-review balanced phone test sample (500 per rating-derived class). No phone test review text enters candidate training or validation.
- From the **next 300,000 source rows**, sample 3,000 per class with seed 43. Split these 9,000 phone reviews into 7,200 train and 1,800 validation with a stratified seed-43 split.
- Combine 6,300 beauty + 7,200 phone reviews for candidate training; 1,350 beauty + 1,800 phone reviews for validation. Ratings 1–2 mean negative, 3 neutral, and 4–5 positive. Fit a new tokenizer only on training text. Use the same BiLSTM architecture and sequence length 120 with a 15,000-word vocabulary, batch size 128, at most five epochs, and early stopping on validation loss.
- Compare the original and candidate on **exactly the same beauty and phone validation/test samples**. Select the candidate because validation macro F1 improved in both categories. The app now uses the candidate weights and tokenizer. The original model and original four-model validation comparison are retained in `artifacts/beauty_baseline/`.

| Category and split | Reviews | Original accuracy | Candidate accuracy | Original macro F1 | Candidate macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beauty validation | 1,350 | 69.1% | 70.6% | 0.693 | 0.709 |
| Phone validation | 1,800 | 62.5% | 66.3% | 0.630 | 0.669 |
| Beauty test | 1,350 | 67.4% | 70.0% | 0.675 | 0.703 |
| Phone test | 1,500 | 63.4% | 65.9% | 0.639 | 0.665 |

Exact metrics and confusion matrices are in `artifacts/multidomain_comparison.json`. Reproduce the candidate training and comparison with `python train_multidomain.py` after installing dependencies and placing the separately supplied beauty training CSV at `data/amazon_beauty_balanced.csv`.

## What changed in the demonstration

The candidate calls “The screen is bright and performance is fast” **neutral** rather than negative. This still misses the clear positive meaning, and the aspect cues correctly show Screen positive and Performance positive. The candidate calls “The keyboard is comfortable but the battery drains quickly” neutral, which is plausible for that mixed review. We also tested an override that would set overall sentiment when multiple explicit aspect cues agreed; it slightly worsened beauty validation and test scores, so it is not deployed. We did not tune the deployed model around a single sentence.

We tested a narrower positive-only override for neutral model predictions without contrast words. It changed **zero** predictions in both validation sets. A broader variant also changed zero validation predictions, but incorrectly changed one neutral beauty test review to positive, lowering beauty test accuracy from 70.0% to about 69.9%. These splits do not support a claim that the override improves general performance. `evaluate_overall_rule.py` and `artifacts/overall_rule_evaluation.json` preserve the exploratory comparison; the override is **not used by the app**.

Instead, the app displays a separate **Aspect summary** beside the overall model label and adds it to the downloaded CSV. The bright-screen review shows **Overall sentiment (model): Neutral** and **Aspect summary: Positive** because Screen and Performance were each marked positive. A mixed camera/battery review shows **Aspect summary: Mixed**. This summary is derived from the aspect heuristic; it does not change the overall prediction or the measured BiLSTM accuracy.

## Limits

Both categories use star ratings as proxy text-sentiment labels. A three-star review can contain praise and complaints, so even correct reading of the text can disagree with its rating. The phone test was already inspected in earlier experiments; this is a repeated exploratory comparison, not a fresh independent benchmark. The source ranges have no exact normalized review-text overlap in the selected samples, but related products or near-duplicate reviews may still occur. We have measured only Beauty and Cell Phones/Accessories, so the result is not a promise about arbitrary uploads. A future independent category/product-held-out, human-checked test would strengthen the claim.
