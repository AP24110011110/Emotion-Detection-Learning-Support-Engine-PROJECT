from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.constants import EMOTION_LABELS
from utils.helpers import encode_labels, normalize_text


class EmotionDataset:
    """Load and prepare text-based emotion labels for model training."""

    REQUIRED_COLUMNS = ["text", "emotion"]

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
        self._validate_dataframe()

    @classmethod
    def from_csv(cls, csv_path: Path, encoding: str = "utf-8") -> "EmotionDataset":
        """Load a labeled dataset from a CSV file."""
        dataframe = pd.read_csv(csv_path, encoding=encoding)
        return cls(dataframe)

    def _validate_dataframe(self) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.dataframe.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        invalid_labels = set(self.dataframe["emotion"]).difference(set(EMOTION_LABELS))
        if invalid_labels:
            raise ValueError(f"Dataset contains unknown emotion labels: {sorted(invalid_labels)}")

    def clean_text(self) -> pd.Series:
        """Normalize the text field for training and tokenization."""
        return self.dataframe["text"].astype(str).apply(normalize_text)

    def labels(self) -> pd.Series:
        """Return the raw emotion labels."""
        return self.dataframe["emotion"].astype(str)

    def encoded_labels(self) -> pd.Series:
        """Return the integer-encoded emotion labels."""
        return pd.Series(encode_labels(self.labels()), index=self.dataframe.index)

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Dict[str, pd.Series]:
        """Split the dataset into training and evaluation subsets."""
        features = self.clean_text()
        labels = self.encoded_labels()
        split_args = {
            "test_size": test_size,
            "random_state": random_state,
        }
        if stratify:
            split_args["stratify"] = labels

        X_train, X_test, y_train, y_test = train_test_split(features, labels, **split_args)
        return {
            "X_train": X_train.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
        }

    def summary(self) -> Dict[str, int]:
        """Return a simple class distribution summary."""
        return self.labels().value_counts().to_dict()


def load_dataset_from_raw(raw_dir: Path, filename: str = "emotion_data.csv") -> EmotionDataset:
    """Convenience helper for loading raw dataset files."""
    csv_path = raw_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset CSV at {csv_path}")
    return EmotionDataset.from_csv(csv_path)
