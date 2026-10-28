"""Conservative explicit sentiment cues for a single aspect in one clause."""

import re

from aspect_rules import explicit_battery_polarity


POSITIVE = r"great|excellent|amazing|fantastic|perfect|wonderful|comfortable|reliable|smooth|fast|bright|sharp|clear|good|love|loved|easy|delicious|tasty"
NEGATIVE = r"terrible|awful|horrible|poor|bad|broken|defective|uncomfortable|unhelpful|slow|laggy|blurry|dim|disappointing|disappointed|hate|hated|difficult"
NEGATION = r"not|never|no|isn't|wasn't|aren't|doesn't|don't|didn't|can't|cannot|won't"


def explicit_aspect_polarity(clause, aspect, keywords=None):
    """Use a clear cue only if the clause mentions one distinct aspect."""
    from review_intelligence import aspects_in
    if aspect == "Battery":
        battery = explicit_battery_polarity(clause)
        if battery:
            return battery
    if aspects_in(clause, keywords) != [aspect]:
        return None
    text = " ".join(clause.lower().split())
    if aspect == "Sound":
        quiet = bool(re.search(r"\b(?:quiet|silent|no noise|little noise|minimal noise|low noise|not noisy|not too noisy)\b", text))
        complaint_text = re.sub(r"\b(?:not(?: too)? noisy|no noise|little noise|minimal noise|low noise)\b", "", text)
        noisy = bool(re.search(r"\b(?:noisy|loud noise|excessive noise|annoying noise|heavy noise|big noise|a lot of noise)\b", complaint_text))
        # A clause with both cues needs context; keep the model's answer.
        if quiet != noisy:
            return "positive" if quiet else "negative"
    # Negated praise is negative; negated criticism is nuanced, so abstain.
    if re.search(rf"\b(?:{NEGATION})\s+(?:very\s+|really\s+)?(?:{NEGATIVE})\b", text):
        return None
    negated_positive = re.search(rf"\b(?:{NEGATION})\s+(?:very\s+|really\s+)?(?:{POSITIVE})\b", text)
    positive = re.search(rf"\b(?:{POSITIVE})\b", text)
    negative = re.search(rf"\b(?:{NEGATIVE})\b", text)
    if negated_positive:
        return "negative" if not negative else None
    if bool(positive) == bool(negative):
        return None
    return "positive" if positive else "negative"
