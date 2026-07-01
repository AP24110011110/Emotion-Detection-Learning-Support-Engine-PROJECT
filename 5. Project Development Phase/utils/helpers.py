import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from utils.constants import EMOTION_LABELS, EMOTION_TO_INDEX, INDEX_TO_EMOTION


def ensure_path_exists(path: Path) -> Path:
    """Ensure the parent directory of a path exists before writing files."""
    path = path if isinstance(path, Path) else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    """Normalize raw text for tokenization and keyword extraction."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def get_timestamp() -> str:
    """Return an ISO-formatted UTC timestamp for logging records."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def format_confidence(confidence: float) -> float:
    """Clamp and round confidence values for presentation."""
    return round(float(max(0.0, min(1.0, confidence))), 4)


def validate_emotion_label(label: str) -> str:
    """Validate that a label exists in the emotion ontology."""
    if label not in EMOTION_LABELS:
        raise ValueError(f"Invalid emotion label: {label}")
    return label


def encode_labels(labels: Iterable[str]) -> np.ndarray:
    """Convert emotion labels to integer indices."""
    encoded = [EMOTION_TO_INDEX[validate_emotion_label(label)] for label in labels]
    return np.array(encoded, dtype=np.int32)


def decode_indices(indices: Iterable[int]) -> List[str]:
    """Convert numeric indices back to emotion labels."""
    labels: List[str] = []
    for idx in indices:
        if idx not in INDEX_TO_EMOTION:
            raise ValueError(f"Invalid emotion index: {idx}")
        labels.append(INDEX_TO_EMOTION[idx])
    return labels


def get_top_prediction(probabilities: np.ndarray) -> Tuple[str, float]:
    """Return the highest scoring emotion label and its confidence."""
    if probabilities.ndim != 1 or probabilities.size != len(EMOTION_LABELS):
        raise ValueError("Probabilities must be a 1D array with one score per emotion label.")

    max_index = int(np.argmax(probabilities))
    top_label = INDEX_TO_EMOTION[max_index]
    top_confidence = format_confidence(float(probabilities[max_index]))
    return top_label, top_confidence


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert raw model logits into a calibrated probability distribution."""
    if logits.ndim != 1:
        raise ValueError("Logits must be a 1D array.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    scaled_logits = np.asarray(logits, dtype=np.float64) / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)
