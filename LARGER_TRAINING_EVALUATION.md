# Larger-data sentiment model

The app now uses a new BiLSTM trained from scratch on 43,497 reviews: the original 6,300 beauty and 7,200 phone training reviews, plus 29,997 additional reviews after exact-text deduplication. Additional reviews came from McAuley Lab's Amazon Reviews 2023 `All_Beauty` and `Cell_Phones_and_Accessories` files, sampled at 5,000 per class per category. The phone additions came from source rows 400,001–1,000,000, away from the prior phone validation and test source ranges. We excluded the original beauty sample and phone training/validation/test texts, then removed overlaps from the final training set. Ratings supplied the labels: 1–2 negative, 3 neutral, 4–5 positive. A three-star review is not necessarily neutral in its wording.

The tokenizer was fitted on training text only (15,000 word limit, 120 tokens per review). The model used batch size 128 and seed 77. We chose epoch 2 because it had the highest mean macro F1 on the separate beauty (1,350) and phone (1,800) validation sets among three epochs evaluated. No test examples were used to fit the weights or tokenizer. See `artifacts/larger_training_comparison.json` for epoch-by-epoch validation results and `artifacts/larger_training_evaluation.json` for test confusion matrices.

| Rating-labeled set | Reviews | Previous accuracy | New accuracy | Previous macro F1 | New macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beauty validation | 1,350 | 70.6% | 73.3% | 0.709 | 0.736 |
| Phone validation | 1,800 | 66.3% | 69.3% | 0.669 | 0.697 |
| Beauty test | 1,350 | 70.0% | 71.3% | 0.703 | 0.715 |
| Phone test | 1,500 | 65.9% | 69.1% | 0.665 | 0.695 |

The phone test had already been inspected in earlier work; treat these test comparisons as exploratory. An AI draft of sentiment labels on 50 beauty reviews agreed with the previous model on 39/50 and the new model on 38/50. Fourteen draft labels were marked ambiguous, and these AI labels are not human ground truth. Both models still call “The screen is bright and performance is fast” neutral; the mixed camera/battery example also remains neutral overall. These examples show why the rating-based gains should not be presented as a fix for every user-visible error. Before relying on a new domain's dashboard, collect a representative human-labeled evaluation sample in the app.

The previous deployed BiLSTM remains under `artifacts/multidomain_candidate/`. Model and tokenizer must be replaced together; using an old tokenizer with new weights produces incorrect input IDs. The aspect dictionary, clause-level heuristics, and human review queue are separate from this retraining and still have their documented limits.
