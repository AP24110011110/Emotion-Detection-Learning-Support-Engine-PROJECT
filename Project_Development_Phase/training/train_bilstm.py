import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (  # noqa: F401
    Bidirectional,
    Dense,
    Embedding,
    GlobalMaxPool1D,
    LSTM,
    SpatialDropout1D,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from training.dataset import EmotionDataset
from training.preprocess import BiLSTMPreprocessor, prepare_labels
from utils.config import Config
from utils.constants import EMOTION_LABELS

logger = logging.getLogger(__name__)


def build_bilstm_model(
    vocabulary_size: int,
    embedding_dim: int = 128,
    max_length: int = 128,
    dropout_rate: float = 0.35,
    num_classes: int = len(EMOTION_LABELS),
) -> Sequential:
    """Create a robust BiLSTM classifier with regularization and dropout."""
    model = Sequential(
        [
            Embedding(
                input_dim=vocabulary_size,
                output_dim=embedding_dim,
                input_length=max_length,
                embeddings_regularizer=l2(1e-5),
            ),
            SpatialDropout1D(rate=dropout_rate),
            Bidirectional(LSTM(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.2)),
            GlobalMaxPool1D(),
            Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train_bilstm(
    dataset_path: Path,
    tokenizer_path: Path,
    model_path: Path,
    num_words: int = 10000,
    max_length: int = 128,
    batch_size: int = 32,
    epochs: int = 18,
) -> Dict[str, Path]:
    """Train the BiLSTM model and persist tokenizer and weights."""
    tf.keras.utils.set_random_seed(42)
    dataset = EmotionDataset.from_csv(dataset_path)
    splits = dataset.train_test_split(test_size=0.2, random_state=42)

    preprocessor = BiLSTMPreprocessor(num_words=num_words, max_length=max_length)
    preprocessor.build_tokenizer(splits["X_train"])
    X_train = preprocessor.texts_to_padded_sequences(splits["X_train"])
    X_test = preprocessor.texts_to_padded_sequences(splits["X_test"])

    y_train = prepare_labels(splits["y_train"], num_classes=len(EMOTION_LABELS))
    y_test = prepare_labels(splits["y_test"], num_classes=len(EMOTION_LABELS))

    class_weights = dict(enumerate(np.bincount(y_train).max() / (np.bincount(y_train) + 1e-6)))

    model = build_bilstm_model(
        vocabulary_size=min(num_words, len(preprocessor.tokenizer.word_index) + 1),
        max_length=max_length,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_path.with_suffix(".checkpoint.h5")
    def scheduler(epoch: int, lr: float) -> float:
        if epoch > 3:
            return lr * 0.8
        return lr

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        LearningRateScheduler(scheduler),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        class_weight=class_weights,
    )

    model.save(model_path)
    preprocessor.save_tokenizer(tokenizer_path)

    return {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    config = Config.load()
    raw_data_path = Path("data/raw/emotion_data.csv")
    train_bilstm(
        dataset_path=raw_data_path,
        tokenizer_path=config.tokenizer_path / "bilstm_tokenizer.json",
        model_path=config.bilstm_model_path,
        num_words=10000,
        max_length=128,
        batch_size=32,
        epochs=10,
    )
