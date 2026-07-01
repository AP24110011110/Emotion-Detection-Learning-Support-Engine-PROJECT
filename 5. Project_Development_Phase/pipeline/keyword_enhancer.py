import numpy as np
from typing import Dict, List

from utils.constants import EMOTION_LABELS, KEYWORD_BOOSTS
from utils.helpers import format_confidence, normalize_text, softmax


def extract_emotion_keywords(text: str) -> List[tuple[str, float]]:
    """Identify weighted emotion keywords in the input text."""
    normalized_text = normalize_text(text)
    tokens = normalized_text.split()
    matches: List[tuple[str, float]] = []
    for token in tokens:
        if token in KEYWORD_BOOSTS:
            matches.append((KEYWORD_BOOSTS[token][0], KEYWORD_BOOSTS[token][1]))
    return matches


def boost_probabilities(
    probabilities: Dict[str, float],
    text: str,
    boost_amount: float = 0.08,
) -> Dict[str, float]:
    """Adjust emotion probabilities based on weighted keyword signals while preserving a valid distribution."""
    if not probabilities:
        return probabilities

    boosted = probabilities.copy()
    keyword_matches = extract_emotion_keywords(text)
    if not keyword_matches:
        return probabilities

    for label, weight in keyword_matches:
        if label in boosted:
            boosted[label] = min(0.999, boosted.get(label, 0.0) + boost_amount * weight)

    logits = np.array([boosted[label] for label in EMOTION_LABELS], dtype=np.float32)
    normalized = softmax(logits)
    return {label: float(normalized[idx]) for idx, label in enumerate(EMOTION_LABELS)}


def enhance_prediction(
    probabilities: Dict[str, float],
    text: str,
) -> Dict[str, float]:
    """Return keyword-enhanced probabilities for a given text input."""
    enhanced = boost_probabilities(probabilities, text)
    return {label: format_confidence(enhanced.get(label, 0.0)) for label in EMOTION_LABELS}
