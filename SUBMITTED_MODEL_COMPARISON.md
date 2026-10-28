# Comparison aligned with the submitted project architectures

The synopsis and interim report specified **MLP, TextCNN, BiLSTM and an attention-based Transformer**. We inspected the team's GitHub repository `pavitagrawal/CRIS_DL` at commit `6221161` and use its four architecture classes from `models/`, with the padding corrections documented below. This new run evaluates those architectures together. It is a separate PyTorch experiment; the Streamlit app continues to serve the previously fine-tuned TensorFlow BiLSTM with its compatible tokenizer.

## Shared protocol

We used the available 9,000-review, class-balanced Amazon All Beauty sample and the same stratified 70/15/15 split (6,300 train, 1,350 validation, 1,350 test; seed 42) for all four models. Labels are **star-derived proxies** (1–2 negative, 3 neutral, 4–5 positive), not human judgments about every sentence. One Keras word tokenizer was fitted only on the training split; all four PyTorch models received the same integer sequences (15,000 vocabulary limit, 120 tokens, post padding). We trained from scratch on CPU for up to five epochs, batch size 128, embedding dimension 128. MLP, TextCNN, and BiLSTM used AdamW at 5e-4; the Transformer used AdamW at 1e-4. The epoch with the highest validation macro F1 was chosen per model; the winning architecture was chosen by that same validation metric. The test set was then scored. The comparison script, data, tokenizer, checkpoints, detailed metrics and confusion matrices are included.

We corrected padding handling in the team's MLP mean pooling, BiLSTM attention pooling, and Transformer encoder masking. The unmodified Transformer trial collapsed to one class; the correction prevented that failure but its configured from-scratch model still performed poorly. The TextCNN class was used as supplied. These changes are visible in the bundled model files. No pretrained Transformer weights were used.

| Model | Best epoch | Validation macro F1 | Test accuracy | Test macro F1 | Train time | Prediction time (1,350 reviews) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 5 | 0.5744 | 57.19% | 0.5730 | 2.4 s | 0.03 s |
| TextCNN | 4 | 0.5978 | 61.19% | 0.6119 | 26.1 s | 0.41 s |
| BiLSTM | 4 | **0.6875** | **66.30%** | **0.6673** | 126.8 s | 1.52 s |
| Transformer | 4 | 0.3409 | 38.22% | 0.3650 | 95.0 s | 0.85 s |

**BiLSTM was selected by validation macro F1.** Its test confusion matrix (true rows, predicted columns: negative, neutral, positive) is `[[307,117,26],[127,293,30],[41,114,295]]`. Full matrices and precision/recall are in `artifacts/submitted_comparison/results.json`; the summary is in `comparison.csv`. Timing reflects this local CPU run and includes all five training epochs, even when an earlier epoch was selected. It is not a hardware-independent complexity ranking.

This is **not** a reproduction of the interim report's claimed 50,000-review, four-category protocol or its MLP/TextCNN preliminary results. That dataset and trained checkpoints were not in the team repository we inspected. The interim report's 20,000-token/200-length setup and preprocessing also differ from this run. These test numbers should not be presented as the report's historical numbers. The poor Transformer result describes this configuration and training budget, not all attention-based Transformers. The deployed BiLSTM is a later, separately trained TensorFlow model; its reported results are not directly comparable to this architecture table.

To repeat the comparison with the included beauty CSV: install `requirements-comparison.txt` in a suitable Python environment, then run `python compare_submitted_models.py --data data/amazon_beauty_balanced.csv`. The app itself does not require PyTorch to show the saved comparison.
