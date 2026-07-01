from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import PreTrainedTokenizerBase

from utils.constants import DEFAULT_MAX_SEQUENCE_LENGTH
from utils.helpers import normalize_text


@dataclass
class BiLSTMPreprocessor:
    """Prepare text sequences and manage tokenizer artifacts for BiLSTM training."""

    num_words: int = 10000
    oov_token: str = "<OOV>"
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    tokenizer: Optional[Tokenizer] = None

    def build_tokenizer(self, texts: Iterable[str]) -> Tokenizer:
        """Fit a Keras tokenizer on normalized training text."""
        tokenizer = Tokenizer(num_words=self.num_words, oov_token=self.oov_token)
        normalized_texts = [normalize_text(text) for text in texts]
        tokenizer.fit_on_texts(normalized_texts)
        self.tokenizer = tokenizer
        return tokenizer

    def texts_to_padded_sequences(self, texts: Iterable[str]) -> np.ndarray:
        """Convert raw text into padded integer sequences for BiLSTM input."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call build_tokenizer first.")

        normalized_texts = [normalize_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(normalized_texts)
        return pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")

    def save_tokenizer(self, output_path: Path) -> None:
        """Persist the fitted tokenizer to disk."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available to save.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_json = self.tokenizer.to_json()
        output_path.write_text(tokenizer_json, encoding="utf-8")

    def load_tokenizer(self, input_path: Path) -> Tokenizer:
        """Load a saved tokenizer from disk."""
        if not input_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {input_path}")
        tokenizer_json = input_path.read_text(encoding="utf-8")
        tokenizer = Tokenizer.from_json(tokenizer_json)
        self.tokenizer = tokenizer
        return tokenizer


def build_bert_encoding(
    texts: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> Dict[str, np.ndarray]:
    """Encode text with a Hugging Face tokenizer for BERT fine-tuning."""
    normalized_texts = [normalize_text(text) for text in texts]
    encoded = tokenizer(
        normalized_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        **({"token_type_ids": encoded["token_type_ids"]} if "token_type_ids" in encoded else {}),
    }


def prepare_labels(labels: Iterable[int], num_classes: int) -> np.ndarray:
    """Return integer labels as categorical values suitable for Keras training."""
    labels_array = np.array(list(labels), dtype=np.int32)
    if labels_array.ndim != 1:
        raise ValueError("Labels must be a 1D iterable of integers.")
    if np.any(labels_array < 0) or np.any(labels_array >= num_classes):
        raise ValueError("Labels contain values outside the allowed class range.")
    return labels_array


def get_padding_length(tokenizer: PreTrainedTokenizerBase, default: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> int:
    """Return a safe sequence length to use during BERT encoding."""
    if tokenizer.model_max_length and tokenizer.model_max_length > 0:
        return min(int(tokenizer.model_max_length), default)
    return default
