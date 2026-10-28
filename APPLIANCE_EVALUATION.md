# Appliance category extension (exploratory)

We took reviews from the official Amazon Reviews 2023 **Appliances** category. Overall sentiment reference labels are **star proxies**: 1–2 negative, 3 neutral, 4–5 positive. A star can disagree with the words in the review. Neither these ratings nor this experiment establishes aspect-level truth.

We sampled 1,500 evaluation reviews (500 per class) from the first 100,000 source records, excluding exact text from existing beauty and phone data. Separately, we sampled 9,000 reviews (3,000 per class) from source rows 100,001–300,000 and split by product ID into 7,115 training and 1,885 validation examples. The previous model and tokenizer were retained, and the model was fine tuned on those appliance training reviews plus 6,300 beauty and 7,200 phone training reviews. We chose epoch 3 from three epochs using the mean macro F1 on beauty, phone and appliance validation sets. No evaluation texts were fitted. Source ordering, possible shared products between the appliance test and training ranges, repeated development inspection, and star labels limit independence.

| Evaluation set | Reviews | Earlier accuracy | Current accuracy | Earlier macro F1 | Current macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beauty | 1,350 | 71.26% | 71.56% | 0.71485 | 0.71680 |
| Phone/accessories | 1,500 | 69.13% | 71.80% | 0.69510 | 0.72014 |
| Appliances | 1,500 | 65.73% | 65.93% | 0.66134 | 0.66059 |
| UCI Amazon sentences (binary reference) | 1,000 | 75.0% | 78.0% | — | — |

UCI has only positive and negative reference labels; the 149 neutral predictions are counted as incorrect. The model improvement on appliances is **small in accuracy and macro F1 slightly declined**. This is an exploratory extension to a third category, not evidence the system works on every possible CSV.

For “The mixer makes a very loud noise,” the current overall model still says **neutral**. We expanded the transparent Sound aspect detector to recognize noise, noisy and quiet; explicit sound cues mark “very loud noise” negative, and “no noise” or “quiet” positive. This makes the particular complaint visible at the aspect level without falsely claiming to have corrected the overall model. Other appliance aspects (for example fabric deterioration) still need coverage and human labeled evaluation.

To try the upload workflow, use `appliance_reviews_300_demo.csv` and select `review`. That file has no labels, so its dashboard reports predictions and **cannot display accuracy**. To estimate accuracy on your own domain, supply human labels and use the existing review queue / labeled sample evaluation workflow. See `artifacts/appliance_external_evaluation.json` for class confusion matrices (negative, neutral, positive order).

These appliance metrics compare the historical pre-short-review models. The currently deployed model was later fine tuned on short labeled UCI sentences; see `SHORT_REVIEW_EVALUATION.md` for its appliance test result.
