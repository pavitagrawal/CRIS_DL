# Restaurant category and cue improvement

The review `The food was delicious` exposed two limitations. With **Products and electronics** selected, no Food aspect existed, so the app showed “No aspects found.” With **Restaurants** selected, the default model still predicted Negative for the whole review and its Food clause.

The single-review screen now suggests **Restaurants** when it finds distinctive restaurant terms and no terms from the selected product profile. It leaves category selection to the user. For a clause naming exactly one recognized aspect, `delicious` and `tasty` are explicit positive cues; `not delicious` is negative. The cue does not override the overall model output. If the aspect and overall labels disagree, the app flags the review for human review.

On the existing annotated SemEval restaurant aspect set, the already developed keyword profile covered 2,227 gold aspect terms. Correct polarity among those covered terms rose from **1,342/2,227 (60.26%)** to **1,359/2,227 (61.02%)**; covered macro F1 rose from 0.4787 to 0.4835. These are **exploratory development-set** results: the restaurant aspect profile had already been developed with this source. The unchanged overall sentiment model was not retrained, and these figures do not measure overall accuracy or prove broader restaurant generalization. Full updated counts are in `artifacts/public_aspect_evaluation.json`.

Example after choosing **Restaurants**:

| Review | Overall model | Food aspect | Interpretation |
|---|---|---|---|
| The food was delicious | Negative | Positive, explicit cue | Visible disagreement; human review warranted |

The safest next research step, if more time is available, is an unseen human-labeled restaurant test with genuine Neutral and mixed reviews. Do not treat a three-star rating as a human Neutral judgment.

The single-review app also offers a side-by-side comparison of the two saved restaurant-capable models. It makes their disagreement visible while keeping the better measured short-review model as the default. This comparison does not blend model answers or override the chosen prediction.
