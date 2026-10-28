# Step 3: Aspect analysis baseline and first trained prototype

The existing app detects a short list of aspect keywords and classifies the matched clause with the **overall review sentiment model**. This is a baseline, not a trained aspect-based method.

## Human-labeled source

SemEval 2014 Task 4 laptop reviews have human-labeled aspect terms and polarities. The [official task description](https://alt.qcri.org/semeval2014/task4/index.php) defines extraction and aspect polarity as separate tasks and includes a fourth `conflict` label. We used `Laptop_Train_v2.xml` from a [Zenodo copy of the task archive](https://zenodo.org/records/14655773) for this exploratory experiment. The XML has 3,045 sentences and 2,358 marked aspect terms. We excluded 45 `conflict` terms because our current app uses only negative, neutral and positive.

## Baseline check

The keyword dictionary recognized 587 of the remaining 2,313 labeled term instances (25.4%). On those 587, the current clause-based method achieved 54.5% accuracy and 0.479 macro F1. This initial check used the entire annotated training file, but neither app model had trained on this laptop corpus.

## Held-out experiment

We split **sentences** into 70% train, 15% validation and 15% test using seed 42 before expanding them into term examples. This prevents the same sentence's aspects from crossing splits. `train_aspect.py` trains a target-marked BiLSTM: it sees the review sentence and the highlighted aspect term and predicts its polarity. Vocabulary fitting uses only the training split. This prototype does not extract aspect terms itself.

| Measure | Clause baseline | Target-marked BiLSTM |
| --- | ---: | ---: |
| Validation macro F1, dictionary-covered terms (78) | 0.555 | 0.450 |
| Validation accuracy, dictionary-covered terms (78) | 62.8% | 64.1% |
| Test macro F1, dictionary-covered terms (86) | 0.511 | 0.506 |
| Test accuracy, dictionary-covered terms (86) | 61.6% | 73.3% |

The trained model's macro F1 is lower on the validation subset, which is the relevant selection measure for the app. Its higher test accuracy does not justify switching after viewing the test results. It achieved 0.458 validation macro F1 on **all** 368 gold aspect terms and 0.550 test macro F1 on **all** 332 gold terms, where the target was supplied. An additional TF-IDF/logistic regression check reached 0.470 validation macro F1 on covered terms, also below the current baseline's 0.555. These covered subsets are small; exact scores can change substantially with a different split.

**Decision:** keep the target-marked BiLSTM out of the app. It remains in the local training workspace for experimentation and is not included in the app-ready bundle. `train_aspect.py` reproduces it, downloading the archive if `--archive` is omitted.

## Narrow correction for explicit battery language

`aspect_rules.py` adds a small set of transparent cues for clear battery-life statements. The app now labels “battery drains quickly” negative and “battery lasts all day” positive, and marks those results as `explicit phrase` without inventing a model score. Unmatched or ambiguous clauses still use the old clause model. On the 78 dictionary-covered validation terms, only four outputs were affected and aggregate accuracy (62.8%) and macro F1 (0.555) stayed the same. On the already-inspected 86 covered test terms, exploratory macro F1 changed from 0.511 to 0.530; this is **not** a fresh independent confirmation of improvement. The fix addresses a visible error but does not validate aspect sentiment generally.

## Follow-up: expanded aspect extraction

We added specific aspect groups and variants found in the **training portion** of the same laptop corpus, such as Keyboard, Software, Storage, Warranty, Touchpad, and Connectivity. The app still uses keywords for detection and the same saved overall sentiment model for clause polarity; no aspect model was silently substituted. `evaluate_aspect_extraction.py` compares the frozen original dictionary against the expanded one using the existing sentence-level split. A detected keyword must overlap an annotated aspect-term span. Each detected mention and gold term can be matched at most once. Precision counts matched detected mentions; recall counts matched gold terms. The SemEval `conflict` polarity terms are excluded, as in the earlier experiment.

| Split and method | Gold terms found | Mention precision | Gold-term recall | Detection F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation: original | 78 / 368 | 82.1% | 21.2% | 0.337 |
| Validation: expanded | 206 / 368 | 84.4% | 56.0% | 0.673 |
| Held-out test: original | 86 / 332 | 84.3% | 25.9% | 0.396 |
| Held-out test: expanded | 198 / 332 | 89.6% | 59.6% | 0.716 |

For the same annotated terms, we also locate the containing clause, run the **unchanged** beauty-trained sentiment model, and apply the already documented battery phrase rule. Some spans lack a containing clause after splitting, so covered totals can be slightly smaller than extraction matches. These polarity figures are exploratory because the laptop test was used earlier while inspecting the prototype; they are not fresh independent confirmation.

| Split and method | Covered with clause | Correct polarity on covered | Correct detection and polarity / all gold terms |
| --- | ---: | ---: | ---: |
| Validation: original | 76 | 64.5% | 13.3% |
| Validation: expanded | 203 | 56.2% | 31.0% |
| Test: original | 86 | 64.0% | 16.6% |
| Test: expanded | 198 | 60.6% | 36.1% |

**Decision:** ship the expanded dictionary. It recognizes many more annotated terms and does not lower mention precision on this corpus, but the newly covered terms are harder for the clause sentiment model: covered-set polarity accuracy drops. The last column counts a gold term as correct only if its aspect was found, a containing clause was located, and its sentiment matched the human label. It is not an accuracy measure for an arbitrary uploaded CSV.

This evaluation matches term spans, **not** the semantic correctness of the chosen aspect group. It uses one laptop corpus, while a phone or beauty upload may have different terms. The clause model can give two aspects in one clause the same polarity even when the opinions differ. A future step is a larger human-labeled cross-category set and a target-aware polarity method that wins on untouched validation data. The beauty-trained model's 67.4% overall review test accuracy is unrelated to these aspect figures. Machine-readable counts are in `artifacts/aspect_extraction_evaluation.json`.

## Follow-up: explicit sentiment cues

The user-visible example “The keyboard is comfortable but the battery drains quickly” revealed a specific polarity error: the clause model called Keyboard neutral. We added a conservative set of clear positive and negative words with simple negation handling in `aspect_polarity.py`. It applies only when a clause names exactly one known aspect; mixed or unsupported wording falls back to the saved clause model. The previous battery-life phrase rule remains first for Battery. An explicit cue has **no model score** and is identified as such in the app.

| Split, expanded detector | Previous polarity accuracy | With cues | Previous macro F1 | With cues |
| --- | ---: | ---: | ---: | ---: |
| Validation, 203 covered terms | 56.2% | 58.1% | 0.534 | 0.541 |
| Previously inspected test, 198 covered terms | 60.6% | 62.1% | 0.548 | 0.559 |

The rule was used on 44 validation terms and agreed with 39 human labels; on the earlier inspected test, it was used on 37 and agreed with 34. On the keyboard example it now gives **Keyboard positive**, while the battery phrase gives **Battery negative**. This is a small, exploratory gain on one laptop dataset, not evidence of reliable sentiment across product categories. The test was previously inspected while improving the app, so its result is not an untouched final confirmation. The overall BiLSTM was not retrained or changed.

## CSV demonstration corrections

The four-row `sample_upload.csv` exposed two further clause errors. We now split an `and` phrase only when each side names one different aspect and each has a clear explicit sentiment cue. Thus “The screen is bright and performance is fast” yields **Screen positive** and **Performance positive**. We also recognize the specific negative cue “unhelpful,” so “customer support was unhelpful” yields **Customer support negative**. More permissive splitting reduced validation polarity accuracy, so the shipped split is intentionally narrow. With this narrow split, expanded-dictionary cue polarity remained **58.1% (0.541 macro F1)** on 203 covered validation terms. On the previously inspected 198-term test subset it was **62.1% (0.559 macro F1)**. The exact count of cue-triggered test terms changed from the preceding run because of the new splitting; see the machine-readable result.

The overall review BiLSTM still predicts **negative** for the screen/performance sample sentence. Overall review predictions and aspect labels are separate outputs, so this can happen; it is an error in the overall prediction for that example. The app now calls out that possibility. The dashboard also lists all aspects tied for most positive/negative mentions and no longer displays stray `None` values below those summaries.
