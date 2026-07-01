import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.constants import CSV_COLUMNS
from utils.helpers import ensure_path_exists


logger = logging.getLogger(__name__)


class CsvLogger:
    """Manage CSV persistence for prediction records."""

    def __init__(self, csv_path: Path, columns: Optional[List[str]] = None) -> None:
        self.csv_path = csv_path
        self.columns = columns or CSV_COLUMNS
        ensure_path_exists(self.csv_path)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        if not self.csv_path.exists():
            logger.info("CSV log not found, creating new file at %s", self.csv_path)
            self._create_csv()
            return

        try:
            with self.csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)
                header = next(reader, [])
            if header != self.columns:
                logger.warning(
                    "CSV header mismatch detected. Expected %s but got %s. Recreating file.",
                    self.columns,
                    header,
                )
                self._create_csv()
        except Exception as exc:
            logger.exception("Failed to validate CSV header, recreating log file.")
            self._create_csv()

    def _create_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.columns)
                writer.writeheader()
        except Exception as exc:
            logger.exception("Failed to create CSV log file at %s", self.csv_path)
            raise RuntimeError(f"Could not create CSV log at {self.csv_path}") from exc

    def append_record(self, record: Dict[str, Any]) -> None:
        """Append a new prediction record to the CSV log."""
        try:
            row = {column: record.get(column, "") for column in self.columns}
            with self.csv_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.columns)
                writer.writerow(row)
        except Exception as exc:
            logger.exception("Failed to append record to CSV log.")
            raise RuntimeError("Failed to write prediction record to CSV.") from exc

    def load_history(self) -> pd.DataFrame:
        """Load the full prediction history as a pandas DataFrame."""
        if not self.csv_path.exists():
            self._create_csv()
            return pd.DataFrame(columns=self.columns)

        try:
            dataframe = pd.read_csv(self.csv_path, dtype=str)
            missing_columns = [col for col in self.columns if col not in dataframe.columns]
            if missing_columns:
                logger.warning(
                    "CSV history is missing columns %s. Adding missing columns with empty values.",
                    missing_columns,
                )
                for column in missing_columns:
                    dataframe[column] = ""
            return dataframe[self.columns]
        except Exception as exc:
            logger.exception("Failed to load CSV history from %s", self.csv_path)
            raise RuntimeError("Failed to load session history from CSV.") from exc

    def validate_record(self, record: Dict[str, Any]) -> None:
        """Validate that the record contains the expected CSV fields."""
        missing_keys = [col for col in self.columns if col not in record]
        if missing_keys:
            raise ValueError(f"Prediction record is missing required fields: {missing_keys}")
