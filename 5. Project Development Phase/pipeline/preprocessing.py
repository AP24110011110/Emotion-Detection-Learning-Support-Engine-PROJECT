from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from training.preprocess import get_padding_length
from utils.constants import DEFAULT_MAX_SEQUENCE_LENGTH
from utils.helpers import normalize_text


def load_bilstm_tokenizer(tokenizer_path: Path):
    """Load a saved Keras tokenizer for BiLSTM inference."""
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"BiLSTM tokenizer not found at {tokenizer_path}")

    tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
    return tokenizer_from_json(tokenizer_json)


def text_to_bilstm_sequences(
    texts: Iterable[str],
    tokenizer,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Convert raw texts to padded integer sequences for BiLSTM inference."""
    normalized_texts = [normalize_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(normalized_texts)
    return pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")


def load_bert_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerBase:
    """Load a saved Hugging Face tokenizer for BERT inference."""
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"BERT tokenizer directory not found at {tokenizer_dir}")
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def encode_for_bert(
    texts: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> Dict[str, np.ndarray]:
    """Encode text for BERT inference with padding and attention masks."""
    normalized_texts = [normalize_text(text) for text in texts]
    effective_length = get_padding_length(tokenizer, default=max_length)
    encoded = tokenizer(
        normalized_texts,
        padding="max_length",
        truncation=True,
        max_length=effective_length,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        **({"token_type_ids": encoded["token_type_ids"]} if "token_type_ids" in encoded else {}),
    }
