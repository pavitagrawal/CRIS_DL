# Final architecture and error analysis

## Experiment decision

The final four-model experiment selected **BiLSTM** by validation macro F1. Its held-out test accuracy was **67.71%** and test macro F1 was **0.6767**. All models used the same saved 35,000/7,500/7,500 split, training-only tokenization, and three rating-derived sentiment classes.

## Architecture comparison

| Model | Selected epoch | Validation macro F1 | Test accuracy | Test macro F1 | Train time | Prediction time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 5 | 0.6062 | 61.13% | 0.6115 | 14.3 s | 0.07 s |
| TextCNN | 5 | 0.6065 | 62.01% | 0.6157 | 215.3 s | 3.22 s |
| BiLSTM | 5 | 0.6803 | 67.71% | 0.6767 | 1096.5 s | 14.12 s |
| Transformer | 4 | 0.3942 | 39.57% | 0.3907 | 1042.4 s | 8.20 s |

## Selected model class analysis

| Class | Support | Correct | Errors | Precision | Recall | F1 | Share of model errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Negative | 2500 | 1794 | 706 | 0.6669 | 0.7176 | 0.6913 | 29.1% |
| Neutral | 2500 | 1384 | 1116 | 0.5767 | 0.5536 | 0.5649 | 46.1% |
| Positive | 2500 | 1900 | 600 | 0.7884 | 0.7600 | 0.7739 | 24.8% |

## Largest BiLSTM confusions

- Actual **neutral** predicted as **negative**: 744 reviews.
- Actual **negative** predicted as **neutral**: 568 reviews.
- Actual **positive** predicted as **neutral**: 448 reviews.

Neutral reviews have the lowest recall and account for most mistakes. The model most often maps neutral reviews to negative, followed by neutral reviews mapped to positive. This pattern supports the dashboard's human-review queue and the explicit warning that model scores are uncalibrated.

## Interpretation limits

The labels come from star ratings: 1-2 negative, 3 neutral, and 4-5 positive. A rating can disagree with the review wording. The experiment therefore measures reproducible agreement with rating proxies, not definitive human sentiment accuracy. The Transformer result is a compact model trained from scratch, not a pretrained language model result. No test metric was used to choose the winning architecture.

## Reproduction

Run `python prepare_multicategory_dataset.py` to recreate the documented sample, then run `python compare_submitted_models.py --splits-dir data/amazon_50k_multicategory --max-vocab 20000 --max-length 200 --epochs 5 --out artifacts/submitted_50k_comparison`. Finally run `python analyze_final_results.py` to rebuild this analysis.
