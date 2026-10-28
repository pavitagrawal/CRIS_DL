# Short-review fine-tuning experiment

The 50,000-review BiLSTM missed several short product sentences in a separate UCI Amazon upload. We fine-tuned its existing PyTorch checkpoint with other UCI sentences and a replay sample from its original four-category training split. The app now offers the updated checkpoint as **Short-review tuned BiLSTM** and retains the original **New 50,000-review BiLSTM** in the sidebar. Aspect labels remain keyword and cue based.

## Data and separation

The UCI Sentiment Labelled Sentences Amazon subset contains 1,000 positive/negative product sentences. After case-insensitive text deduplication and excluding every sentence in the 200-review upload, 587 sentences went to training, 100 to validation, and 100 to a separate test set. Seed: 20261022, stratified by sentiment. The 50 sentences labeled by the user are among the excluded 200. No exact text from that upload entered fine-tuning, validation, or the fresh 100-sentence test. The 200 were inspected during development, so their result is exploratory even though they were excluded from training. UCI has no neutral reference labels.

We replayed 1,200 training reviews per class (3,600 total) from the original 50,000-review training split, and repeated the 587 UCI training sentences three times per epoch. The saved training tokenizer, vocabulary, 200-token length, and two-layer BiLSTM architecture were retained. We trained for three epochs on CPU using AdamW at 1e-4, batch size 64 and gradient clipping. Epoch 2 was selected by UCI validation accuracy, subject to four-category validation macro F1 dropping by at most 0.02 and neutral recall by at most 0.04 relative to the original checkpoint. Epoch 3 improved UCI validation accuracy but failed the neutral-recall guard and was not deployed. See `fine_tune_short_reviews.py` and the fixed split files in `data/uci_short_split/`.

| Evaluation set | Original accuracy | Tuned accuracy | Original macro F1 | Tuned macro F1 |
| --- | ---: | ---: | ---: | ---: |
| UCI validation (100) | 76.0% | 88.0% | 0.550 | 0.593 |
| Fresh UCI test (100) | 78.0% | 85.0% | 0.557 | 0.581 |
| Four-category validation (7,500) | 68.05% | 67.53% | 0.6803 | 0.6729 |
| Four-category test (7,500) | 67.71% | 67.59% | 0.6767 | 0.6726 |
| User-labeled UCI sample (50; exploratory) | 82.0% (41/50) | 88.0% (44/50) | 0.582 | 0.605 |

The updated model's four-category **neutral recall** decreased from 55.36% to 51.44% on the held-out test. The fresh UCI test is binary, while the model can still predict neutral; neutral outputs count as mistakes in accuracy. Its three-class macro F1 includes an absent neutral reference class and is therefore lower than a two-class score would be. The same issue applies to the user's 50 labels, which had no neutral answers. Neither test alone establishes accuracy on every category or aspect polarity.

The 50 labeled sentences were reviewed before this experiment and informed the decision to try short-review fine-tuning. Their 88% is an exploratory follow-up, not a new independent benchmark. The fresh 100-sentence UCI test is separate from training and validation but comes from the same UCI source and is small. The four-category test uses star-derived proxy labels. The original checkpoint remains available for users who prefer its stronger neutral recall and slightly higher four-category macro F1. Full confusion matrices and results are in `artifacts/short_review_candidate/evaluation.json`.

Among the earlier nine user-sample disagreements, the tuned checkpoint corrects three (the design-flaw complaint, “Very wind-resistant,” and the weak-signal complaint). Six remain wrong, including “Timely shipment and quality product,” which it still predicts negative. The model's output should be checked for consequential use; its numeric score is uncalibrated.
