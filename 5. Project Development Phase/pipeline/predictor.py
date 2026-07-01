import logging
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
from tensorflow.keras.models import load_model

from pipeline.keyword_enhancer import enhance_prediction
from pipeline.mixed_emotion import get_top_label_and_confidence, summarize_mixed_emotions
from pipeline.preprocessing import (
    encode_for_bert,
    load_bilstm_tokenizer,
    load_bert_tokenizer,
    text_to_bilstm_sequences,
)
from pipeline.schema import ModelPrediction, PredictionResult
from utils.config import Config
from utils.constants import EMOTION_LABELS
from utils.helpers import format_confidence, softmax

logger = logging.getLogger(__name__)


def load_bert_model_class() -> Type:
    """Load a compatible TensorFlow-backed Transformers classification model class."""
    try:
        from transformers import TFAutoModelForSequenceClassification

        return TFAutoModelForSequenceClassification
    except Exception:
        try:
            from transformers import TFBertForSequenceClassification

            return TFBertForSequenceClassification
        except Exception as exc:
            raise ImportError(
                "A TensorFlow Transformers classification model class is not available. "
                "Install transformers with TensorFlow support or a compatible backend."
            ) from exc


class EmotionPredictor:
    """Predict emotion from text using BiLSTM and BERT models."""

    def __init__(
        self,
        config: Config,
        bilstm_tokenizer_path: Path,
        bert_tokenizer_path: Path,
        max_length: int = 128,
    ) -> None:
        self.config = config
        self.max_length = max_length
        self.bilstm_model = self._load_bilstm_model(config.bilstm_model_path)
        self.bilstm_tokenizer = load_bilstm_tokenizer(bilstm_tokenizer_path)
        self.bert_model = None
        self.bert_tokenizer = None
        self.bert_available = False

        try:
            self.bert_model = self._load_bert_model(config.bert_model_path)
            self.bert_tokenizer = load_bert_tokenizer(bert_tokenizer_path)
            self.bert_available = True
        except Exception as exc:
            logger.warning(
                "BERT model unavailable; falling back to BiLSTM-only inference. %s",
                exc,
            )

    def _load_bilstm_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"BiLSTM model not found at {model_path}")
        return load_model(model_path)

    def _load_bert_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"BERT model not found at {model_path}")
        model_class = load_bert_model_class()
        return model_class.from_pretrained(model_path)

    def _predict_bilstm(self, text: str) -> Dict[str, float]:
        sequences = text_to_bilstm_sequences([text], self.bilstm_tokenizer, max_length=self.max_length)
        probabilities = self.bilstm_model.predict(sequences, verbose=0)
        flattened = np.asarray(probabilities).reshape(-1, len(EMOTION_LABELS))[0]
        calibrated = softmax(np.asarray(flattened, dtype=np.float32))
        return {label: float(score) for label, score in zip(EMOTION_LABELS, calibrated)}

    def _predict_bert(self, text: str) -> Dict[str, float]:
        if not self.bert_available or self.bert_model is None or self.bert_tokenizer is None:
            return self._fallback_bert_probabilities(text)

        encoded_inputs = encode_for_bert([text], self.bert_tokenizer, max_length=self.max_length)
        predictions = self.bert_model.predict({k: np.asarray(v) for k, v in encoded_inputs.items()}, verbose=0)

        if hasattr(predictions, "logits"):
            logits = predictions.logits
        elif isinstance(predictions, tuple):
            logits = predictions[0]
        else:
            logits = predictions

        logits = np.asarray(logits).reshape(-1, len(EMOTION_LABELS))[0]
        probabilities = softmax(logits)
        return {label: float(score) for label, score in zip(EMOTION_LABELS, probabilities)}

    def _fallback_bert_probabilities(self, text: str) -> Dict[str, float]:
        try:
            probabilities = self._predict_bilstm(text)
            logger.info("Using BiLSTM output as a BERT fallback prediction.")
            return probabilities
        except Exception as exc:
            logger.warning(
                "BERT fallback failed because BiLSTM inference failed: %s. Returning uniform probabilities.",
                exc,
            )
            return {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}

    def _build_model_prediction(self, model_name: str, probabilities: Dict[str, float]) -> ModelPrediction:
        top_label, confidence = get_top_label_and_confidence(probabilities)
        return ModelPrediction(
            model_name=model_name,
            top_label=top_label,
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict(self, student_input: str, selected_field: str) -> PredictionResult:
        """Run both emotion classifiers, apply enhancements, and return a unified result."""
        bilstm_probs = self._predict_bilstm(student_input)
        bert_probs = self._predict_bert(student_input)

        enhanced_bilstm_probs = enhance_prediction(bilstm_probs, student_input)
        enhanced_bert_probs = enhance_prediction(bert_probs, student_input)

        bilstm_prediction = self._build_model_prediction("BiLSTM", enhanced_bilstm_probs)
        bert_prediction = self._build_model_prediction(
            "BERT" if self.bert_available else "BERT (fallback)",
            enhanced_bert_probs,
        )

        combined_probs = {
            label: (enhanced_bilstm_probs.get(label, 0.0) + enhanced_bert_probs.get(label, 0.0)) / 2.0
            for label in EMOTION_LABELS
        }
        final_probs = {
            label: format_confidence((combined_probs.get(label, 0.0) * 0.7) + (enhanced_bilstm_probs.get(label, 0.0) * 0.15) + (enhanced_bert_probs.get(label, 0.0) * 0.15))
            for label in EMOTION_LABELS
        }
        final_label, final_confidence = get_top_label_and_confidence(final_probs)

        mixed_emotions = summarize_mixed_emotions(
            enhanced_bilstm_probs,
            enhanced_bert_probs,
            text=student_input,
        )

        return PredictionResult(
            student_input=student_input,
            selected_field=selected_field,
            bilstm_prediction=bilstm_prediction,
            bert_prediction=bert_prediction,
            mixed_emotions=mixed_emotions,
            final_prediction=final_label,
            final_confidence=final_confidence,
            gemini_response=None,
        )


def build_predictor(
    config: Optional[Config] = None,
    bilstm_tokenizer_path: Optional[Path] = None,
    bert_tokenizer_path: Optional[Path] = None,
    max_length: int = 128,
) -> EmotionPredictor:
    """Construct an emotion predictor using configured paths."""
    if config is None:
        config = Config.load()
    if bilstm_tokenizer_path is None:
        bilstm_tokenizer_path = config.tokenizer_path / "bilstm_tokenizer.json"
    if bert_tokenizer_path is None:
        bert_tokenizer_path = config.tokenizer_path / "bert_tokenizer"

    return EmotionPredictor(
        config=config,
        bilstm_tokenizer_path=bilstm_tokenizer_path,
        bert_tokenizer_path=bert_tokenizer_path,
        max_length=max_length,
    )
