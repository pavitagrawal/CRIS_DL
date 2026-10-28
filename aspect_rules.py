"""Conservative, explicit battery-life phrases for the aspect baseline."""

import re


NEGATIVE_BATTERY = [
    r"\b(?:battery|charge)\b.{0,35}\b(?:drains?|draining|dies|dead|poor|terrible|awful|bad|short)\b",
    r"\b(?:drains?|draining|dies|dead|poor|terrible|awful|bad|short)\b.{0,35}\b(?:battery|charge)\b",
    r"\b(?:battery|charge)\b.{0,30}\b(?:does\s*not|doesn't|won't|cannot|can't|never)\s+last\b",
    r"\b(?:battery|charge)\b.{0,30}\b(?:does\s*not|doesn't|won't|cannot|can't|never)\s+(?:hold|keep)\b",
    r"\b(?:short|low)\s+battery\s+life\b",
    r"\b(?:not|never)\s+charging\b",
]
POSITIVE_BATTERY = [
    r"\b(?:battery|charge)\b.{0,30}\b(?:lasts?|lasting)\s+(?:all\s+day|a\s+long\s+time|long)\b",
    r"\b(?:long|great|excellent|amazing|good)\s+battery\s+life\b",
    r"\b(?:battery|charge)\b.{0,30}\b(?:great|excellent|amazing|good)\b",
]


def explicit_battery_polarity(clause):
    """Return a label only for an unambiguous battery-life cue; otherwise None."""
    text = " ".join(clause.lower().split())
    if not re.search(r"\b(?:battery|charge|charging)\b", text):
        return None
    # These negated complaints need wider context; leave them to the model.
    if re.search(r"\b(?:not|doesn't|does\s+not|isn't|wasn't)\s+(?:bad|poor|terrible|drain|drains|die|dies)\b", text):
        return None
    if any(re.search(pattern, text) for pattern in NEGATIVE_BATTERY):
        return "negative"
    if any(re.search(pattern, text) for pattern in POSITIVE_BATTERY):
        # Avoid reading "not good battery life" or "not long battery life" as praise.
        if re.search(r"\b(?:not|never|isn't|wasn't|no)\b.{0,15}\b(?:good|great|excellent|amazing|long|lasts?)\b", text):
            return "negative"
        return "positive"
    return None
