import logging
import sys
from pathlib import Path
from typing import Dict, Type

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer

from training.dataset import EmotionDataset
from training.preprocess import build_bert_encoding, get_padding_length, prepare_labels
from utils.config import Config
from utils.constants import DEFAULT_BERT_MODEL_NAME, EMOTION_LABELS

logger = logging.getLogger(__name__)


def load_bert_model_class() -> Type:
    """Load a compatible Transformers classification model class when the backend is available."""
    try:
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification
    except Exception as exc:
        raise ImportError("A compatible Transformers classification model is not available in this environment.") from exc


def build_bert_model(model_name: str, num_labels: int):
    """Load a pre-trained BERT model configured for sequence classification."""
    model_class = load_bert_model_class()
    return model_class.from_pretrained(model_name, num_labels=num_labels)


def build_tf_dataset(encoded_inputs: Dict[str, tf.Tensor], labels: tf.Tensor, batch_size: int) -> tf.data.Dataset:
    """Create a TensorFlow dataset from encoded BERT inputs and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((encoded_inputs, labels))
    return dataset.shuffle(buffer_size=2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalance-aware fine-tuning."""
    counts = np.bincount(labels)
    max_count = counts.max()
    return {idx: float(max_count / (count + 1e-6)) for idx, count in enumerate(counts)}


def train_bert(
    dataset_path: Path,
    tokenizer_path: Path,
    model_path: Path,
    model_name: str = DEFAULT_BERT_MODEL_NAME,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 4,
) -> Dict[str, Path]:
    """Fine-tune a BERT-based emotion classifier and save the model artifacts."""
    tf.keras.utils.set_random_seed(42)
    dataset = EmotionDataset.from_csv(dataset_path)
    splits = dataset.train_test_split(test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    effective_length = get_padding_length(tokenizer, default=max_length)
    tokenizer.save_pretrained(tokenizer_path)

    train_encodings = build_bert_encoding(splits["X_train"], tokenizer, max_length=effective_length)
    test_encodings = build_bert_encoding(splits["X_test"], tokenizer, max_length=effective_length)

    y_train = prepare_labels(splits["y_train"], num_classes=len(EMOTION_LABELS))
    y_test = prepare_labels(splits["y_test"], num_classes=len(EMOTION_LABELS))

    train_dataset = build_tf_dataset(
        {k: tf.convert_to_tensor(v) for k, v in train_encodings.items()},
        tf.convert_to_tensor(y_train),
        batch_size=batch_size,
    )
    validation_dataset = build_tf_dataset(
        {k: tf.convert_to_tensor(v) for k, v in test_encodings.items()},
        tf.convert_to_tensor(y_test),
        batch_size=batch_size,
    )

    try:
        import torch
    except Exception as exc:
        raise ImportError("PyTorch is required for BERT fine-tuning in the current environment.") from exc

    model = build_bert_model(model_name=model_name, num_labels=len(EMOTION_LABELS))
    optimizer = Adam(learning_rate=2e-5)
    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_path / "checkpoint"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1),
    ]

    class_weights = build_class_weights(y_train)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        class_weight=class_weights,
    )

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    config = Config.load()
    raw_data_path = Path("data/raw/emotion_data.csv")
    train_bert(
        dataset_path=raw_data_path,
        tokenizer_path=config.tokenizer_path / "bert_tokenizer",
        model_path=config.bert_model_path,
        model_name=DEFAULT_BERT_MODEL_NAME,
        max_length=128,
        batch_size=16,
        epochs=4,
    )
