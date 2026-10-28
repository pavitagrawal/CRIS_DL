"""Streamlit dashboard for a trained customer review sentiment model."""

from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

from review_intelligence import ASPECT_PROFILES, LABELS, SubmittedBiLSTMPredictor, suggested_aspect_profile
from review_queue import apply_human_labels, review_reasons
from human_evaluation import draw_sample, evaluate_sample, labeled_sample


st.set_page_config(page_title="Customer Review Intelligence", page_icon="💬", layout="wide")
st.title("Customer Review Intelligence")
st.caption("Overall sentiment from the trained model, with a narrow explicit-cue correction for short restaurant reviews; aspects use keyword-matched clauses and cues.")


@st.cache_resource
def predictor():
    return SubmittedBiLSTMPredictor("artifacts/short_review_candidate")


if not Path("artifacts/short_review_candidate/bilstm.pt").exists():
    st.info("No trained model found. See README.md for the saved checkpoint.")
    st.stop()

if importlib.util.find_spec("torch") is None:
    st.error("PyTorch is required. Install the packages in requirements.txt in this environment.")
    st.stop()
aspect_profile = st.sidebar.selectbox("Review category for aspects", list(ASPECT_PROFILES))
st.sidebar.caption("The category changes the aspect names used in the analysis.")
st.caption("Overall sentiment uses the saved BiLSTM; a clear single-aspect restaurant phrase can correct its label. Aspect labels use keyword and explicit-cue rules.")
st.caption(f"Aspect category: {aspect_profile}. Other categories need their own vocabulary and validation.")
if aspect_profile == "Restaurants":
    st.caption("Restaurant results should be checked, particularly mixed and neutral reviews.")

try:
    engine = predictor()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

tab_single, tab_dataset, tab_models = st.tabs(["Single review", "Upload CSV", "Model comparison"])

with tab_single:
    review = st.text_area("Enter a customer review", placeholder="The camera is excellent but the battery drains quickly.")
    if st.button("Analyze review", type="primary"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            result = engine.analyze([review], aspect_profile=aspect_profile)[0]
            overall_col, aspect_col = st.columns(2)
            overall_col.metric("Overall sentiment", result["sentiment"].title())
            aspect_col.metric("Aspect summary", result["aspect_summary"])
            if result["sentiment_method"] != "model":
                st.info(f"A clear restaurant phrase changed the model's {result['model_sentiment']} "
                        f"label to {result['sentiment']}. Check the review context.")
            st.write(f"Model score for its {result['model_sentiment']} class: {result['confidence']:.1%} (uncalibrated)")
            reasons = review_reasons(result, review)
            if reasons:
                st.warning("Needs human review: " + ", ".join(reasons) + ". The model label may be wrong.")
            st.caption("Aspect summary combines detected aspect labels. It can disagree with the overall model and is not a second trained classifier.")
            if result["aspects"]:
                st.subheader("Aspect mentions")
                st.dataframe(pd.DataFrame(result["aspects"]).rename(columns={
                    "aspect": "Aspect", "sentiment": "Clause sentiment",
                    "confidence": "Model score (if used)", "method": "Method",
                    "evidence": "Matched clause"}), hide_index=True)
            else:
                st.info("No aspect keywords were found in this review.")
                suggestion = suggested_aspect_profile(review, aspect_profile)
                if suggestion:
                    st.info(f"This review mentions terms recognized by {suggestion}. "
                            "Choose that category in the sidebar and analyze again to see those aspects. "
                            "The category choice does not change the selected overall sentiment model.")

with tab_dataset:
    uploaded = st.file_uploader("Upload a UTF-8 CSV containing a review column", type="csv")
    if uploaded is not None:
        try:
            uploaded_bytes = uploaded.getvalue()
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
        if df.empty or not len(df.columns):
            st.warning("The CSV is empty.")
            st.stop()
        columns = list(df.columns)
        preferred = ("review", "review_text", "text", "customer_comment", "comment")
        normalized = [str(name).strip().lower() for name in columns]
        suggested = next((normalized.index(name) for name in preferred if name in normalized), None)
        if suggested is None:
            suggested = next((i for i, name in enumerate(normalized)
                              if any(word in name for word in ("review", "text", "comment"))
                              and not name.endswith(("_id", "_key", "_number"))), 0)
        column = st.selectbox("Which column contains reviews?", columns, index=suggested)
        signature = hashlib.sha256(uploaded_bytes + str(column).encode("utf-8") +
                                   aspect_profile.encode("utf-8")).hexdigest()
        if st.button("Analyze dataset", type="primary"):
            valid = df[column].notna() & df[column].astype(str).str.strip().ne("")
            if not valid.any():
                st.warning("No nonempty reviews in that column.")
            else:
                with st.spinner("Analyzing reviews and aspect mentions..."):
                    rows = df.loc[valid].copy()
                    predictions = engine.analyze(rows[column].astype(str).tolist(), aspect_profile=aspect_profile)
                rows["predicted_sentiment"] = [r["sentiment"] for r in predictions]
                rows["sentiment_method"] = [r["sentiment_method"] for r in predictions]
                rows["model_sentiment"] = [r["model_sentiment"] for r in predictions]
                rows["model_score"] = [round(r["confidence"], 4) for r in predictions]
                rows["aspect_summary"] = [r["aspect_summary"] for r in predictions]
                rows["aspect_mentions"] = ["; ".join(f"{a['aspect']}: {a['sentiment']}"
                                                     for a in r["aspects"]) for r in predictions]
                rows.insert(0, "review_row", range(1, len(rows) + 1))
                reasons = [review_reasons(r, text) for r, text in zip(
                    predictions, rows[column].astype(str))]
                rows["review_status"] = ["Needs review" if reason else "Model result"
                                         for reason in reasons]
                rows["review_reason"] = ["; ".join(reason) for reason in reasons]
                run = st.session_state.get("analysis_run", 0) + 1
                st.session_state["analysis_run"] = run
                st.session_state["analysis"] = {"signature": signature, "rows": rows,
                                                "predictions": predictions, "column": column,
                                                "source_count": len(df), "corrections": {},
                                                "evaluation_labels": {}, "run": run,
                                                "sample": draw_sample(rows, int(signature[:8], 16))}
        analysis = st.session_state.get("analysis")
        if analysis is not None and analysis["signature"] == signature:
            rows = analysis["rows"]
            predictions = analysis["predictions"]
            st.metric("Reviews analyzed", f"{len(rows):,}")
            if len(rows) < analysis["source_count"]:
                st.caption(f"Skipped {analysis['source_count']-len(rows)} empty or missing reviews.")
            counts = rows["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
            cols = st.columns(3)
            for panel, label in zip(cols, LABELS):
                panel.metric(label.title(), f"{counts[label] / len(rows):.1%}")
            st.caption("Overall sentiment comes from the saved review model, with a narrow explicit restaurant cue when applicable. The model_sentiment and sentiment_method columns reveal any correction; model_score always refers to the original model label. Mixed or unfamiliar reviews may still be wrong.")
            st.bar_chart(counts.rename("Reviews"))
            aspect_counts = defaultdict(Counter)
            for result in predictions:
                for item in result["aspects"]:
                    aspect_counts[item["aspect"]][item["sentiment"]] += 1
            if aspect_counts:
                table = pd.DataFrame.from_dict(aspect_counts, orient="index").fillna(0).astype(int)
                table = table.reindex(columns=LABELS, fill_value=0)
                table.index.name = "Aspect"
                st.subheader("Sentiment by aspect mention")
                st.caption("Counts are mentions, so one review may contribute to multiple aspects. Clauses are classified by the overall sentiment model; inspect evidence before drawing conclusions.")
                st.dataframe(table)
                st.bar_chart(table)
                positive = table["positive"]
                negative = table["negative"]
                if positive.max():
                    leaders = ", ".join(positive[positive == positive.max()].index)
                    st.write(f"Most positive mentions: **{leaders}** ({positive.max()} each)")
                if negative.max():
                    leaders = ", ".join(negative[negative == negative.max()].index)
                    st.write(f"Most negative mentions: **{leaders}** ({negative.max()} each)")
            else:
                st.info("No keywords from the selected aspect category were found in these reviews.")
                checked = rows[column].astype(str).head(50)
                hints = Counter(suggested_aspect_profile(text, aspect_profile) for text in checked)
                alternative = next((name for name in ASPECT_PROFILES if name != aspect_profile), None)
                if alternative and hints[alternative] >= max(2, len(checked) // 5):
                    st.info(f"Several sampled reviews contain {alternative} terms. "
                            "Choose that aspect category in the sidebar and analyze the CSV again. "
                            "The category does not change the overall sentiment model.")
            with st.expander("Evaluate model with your own labels"):
                sample = analysis["sample"]
                st.write(f"A fixed random sample of **{len(sample)}** reviews from all {len(rows)} analyzed reviews, "
                         "including flagged and unflagged examples. Read each review and select its sentiment. "
                         "Model answers are hidden here while you label.")
                if len(sample) < 30:
                    st.warning("This upload has fewer than 30 reviews. You can try the evaluation flow, "
                               "but the result is too small to judge model reliability.")
                st.caption("Judge the overall meaning of the text. For mixed praise and criticism, "
                           "choose neutral if neither side dominates. Do not use a star rating as the answer.")
                label_table = pd.DataFrame({
                    "review_row": sample["review_row"].tolist(),
                    "review": sample[column].astype(str).tolist(),
                    "human_sentiment": [analysis["evaluation_labels"].get(int(row), "")
                                        for row in sample["review_row"]],
                })
                edited_sample = st.data_editor(label_table, hide_index=True,
                    key=f"evaluation_{signature}_{analysis['run']}",
                    disabled=["review_row", "review"],
                    column_config={"human_sentiment": st.column_config.SelectboxColumn(
                        "Your sentiment label", options=["", *LABELS], required=False)},
                    use_container_width=True)
                for row_number, label in zip(edited_sample["review_row"], edited_sample["human_sentiment"]):
                    if pd.notna(label) and label in LABELS:
                        analysis["evaluation_labels"][int(row_number)] = label
                    else:
                        analysis["evaluation_labels"].pop(int(row_number), None)
                st.write(f"**{len(analysis['evaluation_labels'])} of {len(sample)}** sample reviews labeled.")
                scored = labeled_sample(sample, analysis["evaluation_labels"])
                st.download_button("Download evaluation labels", scored.to_csv(index=False).encode("utf-8"),
                                   file_name="human_evaluation_sample.csv", mime="text/csv")
                if scored["human_sentiment"].ne("").all():
                    metrics = evaluate_sample(scored)
                    st.metric("Accuracy on this human-labeled sample", f"{metrics['accuracy']:.1%}")
                    st.write(f"Macro F1 across negative, neutral, and positive: **{metrics['macro_f1']:.3f}**")
                    st.write("Confusion matrix: human labels in rows, model predictions in columns")
                    st.dataframe(pd.DataFrame(metrics["confusion"], index=LABELS, columns=LABELS))
                    st.write(f"The review queue flagged **{metrics['flagged']} of {metrics['count']}** sampled reviews. "
                             f"Of **{metrics['errors']}** model errors, **{metrics['flagged_errors']}** were flagged "
                             f"and **{metrics['unflagged_errors']}** were not.")
                    if len(sample) < 30:
                        st.caption("This tiny demonstration sample cannot establish model accuracy or queue performance.")
                    else:
                        st.caption("Results describe only this random sample and your labels. Recheck ambiguous labels; "
                                   "do not retrain on this evaluation sample if you want an independent test later.")
            st.subheader("Human review queue")
            flagged = rows.loc[rows["review_status"] == "Needs review"]
            st.write(f"**{len(flagged):,}** of {len(rows):,} reviews flagged by provisional rules: "
                     "model score below 60%, mixed aspect signals, overall/aspect disagreement, "
                     "or a neutral prediction alongside an explicit complaint phrase.")
            st.caption("The model score is not a calibrated chance of being correct. Unflagged reviews can also be wrong. "
                       "The queue has not been independently validated as an error detector.")
            if len(flagged):
                page_size = 100
                pages = (len(flagged) + page_size - 1) // page_size
                page = st.selectbox("Review queue page", range(1, pages + 1),
                                    format_func=lambda i: f"Page {i} of {pages}")
                subset = flagged.iloc[(page - 1) * page_size: page * page_size]
                editor = pd.DataFrame({
                    "review_row": subset["review_row"].tolist(),
                    "review": subset[analysis["column"]].astype(str).tolist(),
                    "predicted_sentiment": subset["predicted_sentiment"].tolist(),
                    "aspect_summary": subset["aspect_summary"].tolist(),
                    "review_reason": subset["review_reason"].tolist(),
                    "human_sentiment": [analysis["corrections"].get(int(row), "")
                                        for row in subset["review_row"]],
                })
                edited = st.data_editor(editor, hide_index=True,
                    key=f"review_{signature}_{analysis['run']}_{page}",
                    disabled=["review_row", "review", "predicted_sentiment", "aspect_summary", "review_reason"],
                    column_config={"human_sentiment": st.column_config.SelectboxColumn(
                        "Your sentiment label", options=["", *LABELS], required=False)},
                    use_container_width=True)
                for row_number, label in zip(edited["review_row"], edited["human_sentiment"]):
                    if pd.notna(label) and label in LABELS:
                        analysis["corrections"][int(row_number)] = label
                    else:
                        analysis["corrections"].pop(int(row_number), None)
                st.caption(f"{len(analysis['corrections']):,} human labels recorded in this browser session. "
                           "Download the file to keep them.")
            st.subheader("Review level results")
            final_rows = apply_human_labels(rows, analysis["corrections"])
            st.dataframe(final_rows.head(1000), hide_index=True)
            st.download_button("Download predictions and human corrections",
                               final_rows.to_csv(index=False).encode("utf-8"),
                               file_name="review_predictions_with_corrections.csv", mime="text/csv")

with tab_models:
    comparison_path = Path("artifacts/beauty_baseline/model_comparison.csv")
    if not comparison_path.exists():
        comparison_path = Path("artifacts/model_comparison.csv")
    metadata_path = Path("artifacts/metadata.json")
    if comparison_path.exists() and metadata_path.exists():
        import json
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        short_path = Path("artifacts/short_review_candidate/evaluation.json")
        current_path = Path("artifacts/final_model_evaluation.json")
        if current_path.exists():
            current = json.loads(current_path.read_text(encoding="utf-8"))
            st.subheader("Current model evaluation")
            st.dataframe(pd.DataFrame(current["checks"]), hide_index=True)
            st.caption(current["limitations"])
        if short_path.exists():
            st.subheader("Short-review fine-tuning comparison")
            short = json.loads(short_path.read_text(encoding="utf-8"))
            if "candidate" in short:
                table = []
                for name, label in (("short_test", "Fresh 100 short reviews"),
                                    ("four_test", "7,500 four-category reviews"),
                                    ("human_50", "Your 50 labeled reviews")):
                    table.append({"Evaluation set": label,
                                  "Original accuracy": short["baseline"][name]["accuracy"],
                                  "Tuned accuracy": short["candidate"][name]["accuracy"],
                                  "Original macro F1": short["baseline"][name]["macro_f1"],
                                  "Tuned macro F1": short["candidate"][name]["macro_f1"]})
                st.dataframe(pd.DataFrame(table), hide_index=True)
                st.caption("Short-review test labels are positive/negative only. Your 50 were inspected "
                           "during development, so that row is exploratory. Neutral recall on the "
                           "four-category test decreased; see SHORT_REVIEW_FT_EVALUATION.md.")
        fifty_path = Path("artifacts/submitted_50k_comparison/comparison.csv")
        if fifty_path.exists():
            st.subheader("Submitted architectures: 50,000 reviews across four categories")
            st.write("MLP, TextCNN, BiLSTM and attention-based Transformer use the same saved "
                     "35,000/7,500/7,500 train, validation and test split. Each model's checkpoint "
                     "is selected using validation macro F1; the test columns are held-out results.")
            st.dataframe(pd.read_csv(fifty_path), hide_index=True)
            fifty_meta = json.loads(Path("artifacts/submitted_50k_comparison/metadata.json").read_text(encoding="utf-8"))
            st.caption(f"Selected architecture: {fifty_meta['winner']}. Labels come from star ratings, "
                       "which can disagree with review text. This is a new sample, not the exact "
                       "50,000 reviews from the interim report. Select the newly trained PyTorch "
                       "BiLSTM checkpoint for predictions. See SUBMITTED_50K_EVALUATION.md.")
        submitted_path = Path("artifacts/submitted_comparison/comparison.csv")
        if submitted_path.exists():
            st.subheader("Preliminary submitted architectures: 9,000 beauty reviews")
            st.write("MLP, TextCNN, BiLSTM and attention-based Transformer were trained on the same "
                     "9,000-review beauty sample. Each model's epoch was selected by validation macro F1.")
            st.dataframe(pd.read_csv(submitted_path), hide_index=True)
            submitted_meta = json.loads(Path("artifacts/submitted_comparison/metadata.json").read_text(encoding="utf-8"))
            st.caption(f"Winner by validation macro F1: {submitted_meta['winner']}. "
                       "This is an earlier PyTorch architecture experiment. The app can serve the "
                       "new 50,000-review PyTorch BiLSTM or its previous TensorFlow BiLSTM. "
                       "See SUBMITTED_MODEL_COMPARISON.md for dataset and limitations.")
        st.write("**Earlier exploratory RNN/LSTM/GRU/BiLSTM comparison** (different architectures)")
        st.dataframe(pd.read_csv(comparison_path), hide_index=True)
        st.write(f"The original comparison selected **{info['best_model']}**. The app uses a BiLSTM subsequently trained on beauty, phone and appliance reviews.")
        expanded_path = Path("artifacts/multidomain_comparison.json")
        if expanded_path.exists():
            study = json.loads(expanded_path.read_text(encoding="utf-8"))
            st.write("**Earlier retraining versus original BiLSTM** (same reviews in each row)")
            rows = []
            for name in ("beauty_validation", "phone_validation", "beauty_test", "phone_test"):
                for model_name, key in (("Original beauty BiLSTM", "current"),
                                        ("Earlier beauty + phone BiLSTM", "candidate")):
                    metrics = study[name][key]
                    rows.append({"Dataset": name.replace("_", " ").title(), "Reviews": study[name]["count"],
                                 "Model": model_name, "Accuracy": metrics["accuracy"],
                                 "Macro F1": metrics["f1_macro"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Sentiment labels come from star ratings. The phone test was previously inspected, so its result is exploratory. See MULTIDOMAIN_EVALUATION.md for limits.")
        larger_path = Path("artifacts/larger_training_evaluation.json")
        if larger_path.exists():
            study = json.loads(larger_path.read_text(encoding="utf-8"))
            st.write("**Earlier larger-data model versus previous model** (same reviews in each row)")
            rows = []
            for name in ("beauty_test", "phone_test"):
                for model_name, key in (("Previous BiLSTM", "old"), ("Larger-data BiLSTM", "candidate")):
                    metrics = study[name][key]
                    rows.append({"Dataset": name.replace("_", " ").title(), "Reviews": study[name]["count"],
                                 "Model": model_name, "Accuracy": metrics["accuracy"],
                                 "Macro F1": metrics["macro_f1"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Labels come from star ratings. The phone test was previously inspected. The 50 AI draft labels went from 78% to 76% agreement, and the bright-screen example remains neutral. See LARGER_TRAINING_EVALUATION.md.")
            for name in ("beauty_test", "phone_test"):
                st.write(f"Earlier larger-data model {name.replace('_', ' ')} confusion matrix (true rows, predicted columns)")
                st.dataframe(pd.DataFrame(study[name]["candidate"]["confusion"], index=info["labels"],
                                          columns=info["labels"]))
        appliance_path = Path("artifacts/appliance_external_evaluation.json")
        if appliance_path.exists():
            study = json.loads(appliance_path.read_text(encoding="utf-8"))
            st.write("**Historical appliance fine tuning versus beauty + phone model** (same reviews in each row)")
            rows = []
            for name in ("beauty_test", "phone_test", "appliance_test"):
                for model_name, key in (("Earlier beauty + phone", "old"), ("Appliance extension", "candidate")):
                    metrics = study[name][key]
                    rows.append({"Dataset": name.replace("_", " ").title(), "Reviews": study[name]["n"],
                                 "Model": model_name, "Accuracy": metrics["accuracy"],
                                 "Macro F1": metrics["macro_f1"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Stars supply proxy sentiment labels. These test sets were inspected during development, so comparisons are exploratory; see APPLIANCE_EVALUATION.md.")
        short_path = Path("artifacts/short_review_evaluation.json")
        if short_path.exists():
            study = json.loads(short_path.read_text(encoding="utf-8"))
            st.write("**Current short-review model versus appliance extension** (same reviews in each row)")
            rows = []
            for name in ("uci_heldout", "beauty_test", "phone_test", "appliance_test"):
                for model_name, key in (("Appliance extension", "old"), ("Current short-review model", "candidate")):
                    metrics = study[name][key]
                    rows.append({"Dataset": name.replace("_", " ").title(), "Reviews": study[name]["n"],
                                 "Model": model_name, "Accuracy": metrics["accuracy"], "Macro F1": metrics["macro_f1"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("UCI test sentences were kept out of this fine tuning run, but an earlier project version had evaluated all 1,000. Beauty, phone and appliance tests were also inspected. These comparisons are exploratory; see SHORT_REVIEW_EVALUATION.md.")
        else:
            st.write("Held-out test metrics for selected model:", info["test_metrics"])
            matrix_path = Path("artifacts/test_confusion_matrix.csv")
            if matrix_path.exists():
                st.write("Confusion matrix (rows = true class, columns = predicted class)")
                st.dataframe(pd.read_csv(matrix_path, index_col=0))
        external_path = Path("artifacts/uci_external_evaluation.json")
        if external_path.exists():
            external = json.loads(external_path.read_text(encoding="utf-8"))
            st.subheader("UCI Amazon review sentences: held-out split")
            st.metric("Accuracy on UCI test split", f"{external['correct'] / external['count']:.1%}")
            st.write(f"{external['correct']} correct out of {external['count']} reviews. "
                     "UCI labels are positive or negative only; neutral model predictions count as incorrect.")
            st.dataframe(pd.DataFrame(external["confusion"],
                                      index=["Actual negative", "Actual positive"],
                                      columns=["Predicted negative", "Predicted neutral", "Predicted positive"]))
            st.caption("The other UCI sentences were used for training (697) and validation (150). "
                       "This split tests overall sentiment, not aspect detection. The old model had already been evaluated on all 1,000 sentences, so treat the comparison as exploratory. "
                       "Source: UCI Sentiment Labelled Sentences (Kotzias et al., 2015), DOI 10.24432/C57604. See SHORT_REVIEW_EVALUATION.md.")
    else:
        st.info("Training comparison files are missing.")
