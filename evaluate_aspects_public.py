"""Evaluate the shipped aspect detector and clause polarity on SemEval annotations.

Usage: python evaluate_aspects_public.py --archive semeval14.zip
The Phase B files contain term spans but no polarity; restaurant training has
polarity labels and is used only for cross-domain evaluation here.
"""
import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from evaluate_aspect_extraction import evaluate, evaluate_polarity
from review_intelligence import RESTAURANT_ASPECT_KEYWORDS, SubmittedBiLSTMPredictor


def load_xml(archive, name, polarity=False):
    with zipfile.ZipFile(archive) as handle:
        root = ET.fromstring(handle.read(name))
    sentences = []
    for item in root.findall(".//sentence"):
        terms = []
        for term in item.findall(".//aspectTerm"):
            label = term.get("polarity")
            if polarity and label not in {"negative", "neutral", "positive"}:
                continue
            start, end = int(term.get("from")), int(term.get("to"))
            terms.append({"target": term.get("term"), "start": start, "end": end, "label": label})
        sentences.append({"text": item.findtext("text") or "", "terms": terms})
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="artifacts/public_aspect_evaluation.json")
    args = parser.parse_args()
    datasets = {
        "laptop_phase_b": load_xml(args.archive, "Laptops_Test_Data_phaseB.xml"),
        "restaurant_phase_b": load_xml(args.archive, "Restaurants_Test_Data_phaseB.xml"),
    }
    result = {"source": "SemEval 2014 Task 4 archive", "detection": {}}
    for name, sentences in datasets.items():
        result["detection"][name] = {"sentences": len(sentences), **evaluate(sentences, None)}
    result["restaurant_profile_detection"] = evaluate(datasets["restaurant_phase_b"], RESTAURANT_ASPECT_KEYWORDS)
    restaurant = load_xml(args.archive, "Restaurants_Train_v2.xml", polarity=True)
    predictor = SubmittedBiLSTMPredictor("artifacts/short_review_candidate")
    result["polarity"] = {"dataset": "restaurant training file; no restaurant data used to train or select app aspect rules",
                          "sentences": len(restaurant), **evaluate_polarity(restaurant, predictor, None, True)}
    result["restaurant_profile_polarity_development"] = evaluate_polarity(restaurant, predictor, RESTAURANT_ASPECT_KEYWORDS, True)
    result["limits"] = "Restaurant profile keywords were chosen using restaurant training term frequencies. Phase B detection was inspected before this profile was created; its comparison is exploratory. Term-span overlap does not check semantic group correctness. Restaurant polarity is scored only on gold terms with a matching keyword and located clause. Phase B has no polarity labels. No neural model was retrained."
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
