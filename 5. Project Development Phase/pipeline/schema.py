from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from utils.constants import EMOTION_LABELS


@dataclass
class ModelPrediction:
    """Represents a prediction from a single emotion model."""

    model_name: str
    top_label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_label not in EMOTION_LABELS:
            raise ValueError(f"Unknown emotion label: {self.top_label}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class PredictionResult:
    """Aggregate inference result returned by the prediction pipeline."""

    student_input: str
    selected_field: str
    bilstm_prediction: ModelPrediction
    bert_prediction: ModelPrediction
    mixed_emotions: List[str]
    final_prediction: str
    final_confidence: float
    gemini_response: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert the prediction result to a flattened dictionary for CSV logging."""
        return {
            "timestamp": self.timestamp,
            "student_input": self.student_input,
            "selected_field": self.selected_field,
            "bilstm_prediction": self.bilstm_prediction.top_label,
            "bilstm_confidence": f"{self.bilstm_prediction.confidence:.4f}",
            "bert_prediction": self.bert_prediction.top_label,
            "bert_confidence": f"{self.bert_prediction.confidence:.4f}",
            "mixed_emotions": ",".join(self.mixed_emotions),
            "final_prediction": self.final_prediction,
            "final_confidence": f"{self.final_confidence:.4f}",
            "gemini_response": self.gemini_response or "",
        }

    def summarize(self) -> Dict[str, object]:
        """Return a compact summary useful for UI display."""
        return {
            "final_prediction": self.final_prediction,
            "final_confidence": self.final_confidence,
            "mixed_emotions": self.mixed_emotions,
            "bilstm": {
                "label": self.bilstm_prediction.top_label,
                "confidence": self.bilstm_prediction.confidence,
            },
            "bert": {
                "label": self.bert_prediction.top_label,
                "confidence": self.bert_prediction.confidence,
            },
        }
