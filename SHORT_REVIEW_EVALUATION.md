# Improving short product review sentiment

The previous overall model misclassified “The mixer is noisy” as neutral. This iteration fine tuned the **BiLSTM itself**, while keeping the tokenizer and the aspect analyzer unchanged. The model now predicts **negative** for that exact sentence, without a rule overriding the overall label. It still misclassifies other examples, including “The mixer makes a very loud noise” and “The screen is bright and performance is fast” as neutral. Individual examples do not measure general quality.

## Labeled data and split

We used the Amazon sentence subset (1,000 human-selected, binary-labeled review sentences) from UCI Sentiment Labelled Sentences, DOI 10.24432/C57604. Identical text was grouped so duplicate sentences stayed in one split: 697 training, 150 validation, and 153 test sentences. No neutral reference labels exist; neutral predictions count as incorrect. The earlier deployed model was evaluated on **all 1,000** UCI sentences before this split, so the test comparison is exploratory despite keeping the 153 sentences out of this fine tuning run. This is not a fresh independent benchmark.

Starting with the appliance-extended BiLSTM, we replayed 6,300 beauty, 7,200 phone/accessory, and 7,115 appliance training reviews alongside the 697 UCI training sentences repeated eight times per epoch. The review-domain labels come from stars (1–2 negative, 3 neutral, 4–5 positive); UCI labels are human curated. We used the existing tokenizer, an Adam learning rate of 2e-5, batch size 128, and four epochs. Epoch 4 had the best mean validation macro F1 across UCI, beauty, phone, and appliance sets. We never fitted model weights on the UCI test split or category tests. The category tests were already inspected during previous development.

## Same-review comparison

| Evaluation set | Reviews | Before accuracy | Current accuracy | Before macro F1 | Current macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI held-out split (binary reference) | 153 | 79.74% | **84.97%** | 0.57112 | 0.58956 |
| Beauty (star proxy) | 1,350 | 71.56% | **71.70%** | 0.71680 | 0.71660 |
| Phone/accessories (star proxy) | 1,500 | **71.80%** | 71.07% | **0.72014** | 0.71055 |
| Appliances (star proxy) | 1,500 | 65.93% | **66.20%** | 0.66059 | 0.66147 |

The phone tradeoff is real in this sample. The UCI gain concerns short, selected product sentences and does not establish the result for arbitrary CSV uploads. In the UCI test split, 130/153 labels match exactly; 12 neutral predictions count as errors. The saved confusion matrices are in `artifacts/short_review_evaluation.json`, ordered negative, neutral, positive.

The first four-model RNN/LSTM/GRU/BiLSTM comparison is still the architecture experiment on beauty reviews. This is a later fine tuning of its chosen BiLSTM, not a new four-model comparison. Aspect labels still use keyword-matched clauses and narrow explicit cues, and have separate limitations.
