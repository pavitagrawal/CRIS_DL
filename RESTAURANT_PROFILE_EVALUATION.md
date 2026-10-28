# Restaurant aspect profile

The app now has a **Review category for aspects** menu with `Products and electronics` (the prior behavior) and `Restaurants`. The latter uses restaurant-specific labels Food, Service, Atmosphere, Price, Drinks, and Delivery. The menu affects keyword detection, clause splitting, and explicit cue attribution; it does **not** retrain or change the overall sentiment model. The category is included in the CSV analysis signature so changing it requires a new analysis and cannot silently reuse old results.

The profile vocabulary was chosen from frequent annotated terms in the SemEval 2014 restaurant **training** file. Results on the 800-sentence restaurant Phase B file are exploratory because the previous evaluation already examined that file before this change. Detection gives credit for term-span overlap and does not verify whether a semantic category is correct.

| Restaurant Phase B detection | Original product profile | Restaurant profile |
| --- | ---: | ---: |
| Annotated terms found | 110 / 1,134 | 700 / 1,134 |
| Mention precision | 66.3% | 94.1% |
| Gold-term recall | 9.7% | 61.7% |
| Detection F1 | 16.9% | 74.5% |

On the restaurant **training** file, the new profile covered 2,227 / 3,602 gold polarity terms with a matched keyword and clause; 1,342 were correctly labeled. This is 60.3% accuracy on covered terms, macro F1 0.479, and 37.3% correctly found and labeled among all gold terms. The older product vocabulary covered 344 terms, with 244 correctly labeled (70.9% on its narrow covered subset; 6.8% of all gold terms). The new profile detects much more restaurant feedback, but polarity on the new coverage is weaker. These training-file polarity figures are development diagnostics, not a final untouched benchmark. The Phase B archive has no polarity answers.

Try the restaurant profile with `The food is excellent but the service is bad.` It should show Food positive and Service negative. For uploads, choose the profile before pressing **Analyze dataset**. The app still cannot infer or validate a completely arbitrary new category; each additional category needs vocabulary and separate labeled evaluation. See `artifacts/public_aspect_evaluation.json` and `evaluate_aspects_public.py` for reproducible counts.
