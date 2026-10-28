# Phase 3 completion audit

This audit maps the remaining-work plan in the Phase 2 interim report to the final project bundle. It distinguishes completed implementation and measured evidence from claims that still require caution.

## Completion decision

The planned software and experimental deliverables are present: one shared preprocessing protocol, four implemented architectures, a fixed four-category comparison, validation-based model selection, confusion matrices, class-level error analysis, aspect extraction and polarity checks, a Streamlit application, a reproducible data builder, a demo workflow, and formal final-report and presentation files.

The project should be presented as complete for the course scope, with two explicit qualifications. The 50,000-row dataset is reproducibly generated but excluded from Git because of its size, and the reported sentiment labels are derived from ratings rather than independent three-class human annotations. The aspect component is a validated keyword-and-cue application layer, not a fully trained end-to-end ABSA model.

## Roadmap traceability

| Interim-report commitment | Final evidence | Status |
| --- | --- | --- |
| Implement BiLSTM under the common protocol | `models/model_bilstm.py`, `models/train_bilstm.py`, and the saved four-model results | Complete |
| Implement attention-based Transformer | `models/model_transformer.py`, `models/train_transformer.py`, and the saved four-model results | Complete |
| Keep a common dataset split and preprocessing protocol | `prepare_multicategory_dataset.py`, `utils/preprocessing.py`, and saved metadata for the 35,000/7,500/7,500 split | Complete |
| Tune and select models without test leakage | Per-epoch validation macro F1 selection in `compare_submitted_models.py`; fixed architecture settings documented in metadata | Complete for the declared compact search; not an exhaustive hyperparameter sweep |
| Compare Accuracy, macro Precision, Recall, F1, training time, and prediction time | `artifacts/submitted_50k_comparison/results.json` and `comparison.csv` | Complete |
| Add confusion-matrix and error analysis | Saved confusion matrices plus `FINAL_ARCHITECTURE_ANALYSIS.md` and `artifacts/final_class_metrics.csv` | Complete |
| Refine aspect identification and application layer | Product and restaurant profiles, clause evidence, conservative cues, SemEval detection/polarity evaluation, and human-review warnings | Complete within the documented heuristic scope |
| Finish dashboard and demo | `app.py`, included runtime checkpoint, CSV workflow, charts, review queue, human labels, downloads, and `demo.html` | Complete |
| Final report and presentation | `docs/final/Customer_Review_Intelligence_Final_Report.docx`, matching PDF, and final PPTX | Complete |
| Preserve version-control evidence | Local branch retains the original repository history and all work remains uncommitted for team-controlled commits | Complete locally |

## Final experiment decision

The final four-model comparison selected BiLSTM by validation macro F1. On the held-out 7,500-review test split it reached 67.71% accuracy and 0.6767 macro F1. Neutral reviews were the weakest class: recall was 55.36%, and 1,116 of 2,500 neutral examples were misclassified. The compact Transformer trained from scratch performed poorly and should not be described as comparable to a pretrained Transformer such as BERT.

## Risk controls retained from the interim report

- Class imbalance and class-specific weakness: the saved reports include macro metrics, class metrics, and confusion matrices.
- Long-review truncation: every architecture uses the same documented 200-token limit in the final comparison.
- Transformer cost: the experiment uses a compact two-layer encoder, controlled sequence length, gradient clipping, and validation-based epoch selection.
- Aspect-label availability: the application separates overall sentiment from aspect evidence and reports the heuristic method used for each aspect label.
- Fair comparison: the four planned architectures share one split, tokenizer, label mapping, vocabulary limit, sequence length, and evaluation code.

## Reproduction and verification

1. Run `scripts/bootstrap.ps1` in PowerShell.
2. Run `scripts/reproduce_phase3.ps1` to rebuild the dataset, train the four architectures, and regenerate the analysis.
3. Run `scripts/check.ps1` for the test suite.
4. Run `python scripts/validate_project.py` for dependency-light bundle checks.

The full training run is intentionally not repeated during ordinary application setup. The included checkpoint allows the Streamlit demonstration to run without retraining.
