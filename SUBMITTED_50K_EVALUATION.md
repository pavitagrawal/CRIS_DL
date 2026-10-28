# Fresh four-model experiment on 50,000 customer reviews

This experiment trains the four architectures named in the submitted project plan—MLP, TextCNN, BiLSTM and an attention-based Transformer—from scratch on the same newly sampled review dataset. It is a new experiment, **not a reproduction of the exact 50,000 rows or historical figures in the interim report**. The current Streamlit app serves a later fine-tuned PyTorch BiLSTM derived from the winning checkpoint.

## Dataset and split

The `prepare_multicategory_dataset.py` script samples 12,500 reviews each from Amazon Reviews 2023 All Beauty, Books, Electronics and Home & Kitchen. It streams the first 200,000 raw records of each category, takes a balanced sample of the three rating-derived classes, removes exact duplicate text ignoring case, and saves the source URLs and method in `data/amazon_50k_multicategory/provenance.json`. Ratings 1–2 become negative, 3 neutral, and 4–5 positive. These are **proxy labels**, so an ironic or mixed review may disagree with its star rating.

The fixed split has 35,000 training, 7,500 validation and 7,500 test reviews, stratified by category and class. No exact review text crosses the three splits. The split is at review level, so multiple reviews of one product can appear in different splits. The underlying dataset is a separate download from the app bundle.

## Shared training protocol

The team model classes came from `pavitagrawal/CRIS_DL` commit `6221161`, with padding-mask fixes to MLP, BiLSTM and Transformer. One word tokenizer was fitted only on training reviews; all models received sequences of at most 200 tokens with a 20,000-word vocabulary limit, post padding and the same class mapping. Each uses 128-dimensional embeddings and five training epochs with batch size 128, AdamW and seed 42. Learning rate was 5e-4 for MLP, TextCNN and BiLSTM, and 1e-4 for Transformer. No pretrained Transformer weights were used. Each model's saved epoch and the overall winning architecture are selected by **validation macro F1**; the held-out test set is evaluated after selection. See `compare_submitted_models.py` for the exact procedure.

## Measured results

The table below is populated from the saved run. Times are wall-clock measurements on this CPU environment and cover all five training epochs, even if the best checkpoint came from an earlier epoch.

| Model | Best epoch | Validation macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 5 | 0.6062 | 61.13% | 0.6117 | 0.6113 | 0.6115 | 0.2 min |
| TextCNN | 5 | 0.6065 | 62.01% | 0.6168 | 0.6201 | 0.6157 | 3.6 min |
| BiLSTM | 5 | 0.6803 | 67.71% | 0.6773 | 0.6771 | 0.6767 | 18.3 min |
| Transformer | 4 | 0.3942 | 39.57% | 0.3970 | 0.3957 | 0.3907 | 17.4 min |

**Selected architecture: BiLSTM** by validation macro F1 (0.6803); its held-out test accuracy was 67.71% and macro F1 0.6767. The app loads the later fine-tuned BiLSTM with its matching tokenizer; PyTorch must be installed to use it for predictions.

Confusion matrices use true classes as rows and predictions as columns, ordered **negative, neutral, positive**.

- **MLP test:** `[[1563, 678, 259], [765, 1278, 457], [224, 532, 1744]]`
- **TextCNN test:** `[[1733, 475, 292], [881, 1138, 481], [281, 439, 1780]]`
- **BiLSTM test:** `[[1794, 568, 138], [744, 1384, 372], [152, 448, 1900]]`
- **Transformer test:** `[[701, 1011, 788], [596, 1242, 662], [480, 995, 1025]]`

Full precision, recall, F1, validation and test confusion matrices, selected epochs and timings are saved under `artifacts/submitted_50k_comparison/`. The result measures **overall sentiment against rating-derived labels**; it does not measure aspect detection or prove performance on arbitrary new CSV categories. Different samples and protocols prevent direct score comparisons with the interim report or the earlier 9,000-review beauty experiment.

To reproduce the run, install `requirements-comparison.txt`, extract the separate dataset archive in `data/amazon_50k_multicategory/`, then run:

```bash
python compare_submitted_models.py --splits-dir data/amazon_50k_multicategory --max-vocab 20000 --max-length 200 --epochs 5 --out artifacts/submitted_50k_comparison
```
