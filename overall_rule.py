"""Narrow agreement rule for two clearly positive aspect clauses."""

import re


def agreed_positive_aspects(review, model_label, aspects):
    """Return positive only for two independent, explicitly positive clauses."""
    from review_intelligence import aspect_clauses

    if model_label != "neutral":
        return None
    if re.search(r"\b(?:but|however|although|whereas|though|yet|despite|not|never|without|no)\b",
                 review, re.I):
        return None
    clauses = aspect_clauses(review)
    if len(clauses) < 2 or len(aspects) < 2:
        return None
    if len({item["aspect"] for item in aspects}) < 2:
        return None
    if all(item["sentiment"] == "positive" and
           item["method"] in ("explicit cue", "explicit phrase") for item in aspects):
        return "positive"
    return None
