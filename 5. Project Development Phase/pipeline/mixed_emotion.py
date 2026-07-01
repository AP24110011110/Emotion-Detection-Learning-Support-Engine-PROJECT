from typing import Dict, List, Optional, Tuple

from pipeline.keyword_enhancer import extract_emotion_keywords
from utils.constants import EMOTION_LABELS, MIXED_EMOTION_THRESHOLD
from utils.helpers import format_confidence


def _has_semantic_support(label: str, text: Optional[str]) -> bool:
    """Return True when the candidate emotion appears to be supported by the input text."""
    if not text:
        return True

    supported_labels = {match_label for match_label, _weight in extract_emotion_keywords(text)}
    return label in supported_labels


def detect_mixed_emotions(
    probabilities: Dict[str, float],
    threshold: float = MIXED_EMOTION_THRESHOLD,
    text: Optional[str] = None,
    max_secondary_emotions: int = 2,
) -> List[str]:
    """Return secondary emotion labels only when the evidence is meaningful and text-supported."""
    if not probabilities:
        return []

    sorted_labels = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if not sorted_labels:
        return []

    _, top_score = sorted_labels[0]
    mixed: List[str] = []

    for label, score in sorted_labels[1:]:
        if score < max(threshold, 0.12):
            continue
        if score < max(0.12, 0.35 * top_score):
            continue

        primary_gap = top_score - score
        if top_score >= 0.70 and primary_gap >= 0.25:
            continue
        if top_score >= 0.55 and primary_gap >= 0.20:
            continue
        if top_score >= 0.40 and primary_gap >= 0.15:
            continue
        if not _has_semantic_support(label, text):
            continue

        mixed.append(label)
        if len(mixed) >= max_secondary_emotions:
            break

    return mixed


def aggregate_probabilities(
    bilstm_probs: Dict[str, float],
    bert_probs: Dict[str, float],
) -> Dict[str, float]:
    """Average the probability distributions from both models into a single distribution."""
    if set(bilstm_probs.keys()) != set(bert_probs.keys()):
        raise ValueError("Probability dictionaries must share the same emotion labels.")

    return {
        label: format_confidence((bilstm_probs[label] + bert_probs[label]) / 2.0)
        for label in EMOTION_LABELS
    }


def summarize_mixed_emotions(
    bilstm_probs: Dict[str, float],
    bert_probs: Dict[str, float],
    threshold: float = MIXED_EMOTION_THRESHOLD,
    text: Optional[str] = None,
) -> List[str]:
    """Combine mixed emotion candidates from both models."""
    mixed_bilstm = set(detect_mixed_emotions(bilstm_probs, threshold=threshold, text=text))
    mixed_bert = set(detect_mixed_emotions(bert_probs, threshold=threshold, text=text))
    return sorted(mixed_bilstm.union(mixed_bert))


def get_top_label_and_confidence(probabilities: Dict[str, float]) -> Tuple[str, float]:
    """Return the highest-probability label and its confidence score."""
    if not probabilities:
        raise ValueError("Probabilities cannot be empty.")

    top_label = max(probabilities, key=probabilities.get)
    return top_label, format_confidence(probabilities[top_label])
