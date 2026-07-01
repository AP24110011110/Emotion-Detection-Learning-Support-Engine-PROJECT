import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from the environment."""

    gemini_api_key: str
    csv_log_path: Path
    bilstm_model_path: Path
    bert_model_path: Path
    tokenizer_path: Path
    max_history_records: int

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration values from a .env file or environment variables."""
        if env_file:
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()

        if not (api_key := os.getenv("GEMINI_API_KEY")):
            raise ValueError("GEMINI_API_KEY is missing from the environment")

        BASE_DIR = Path(__file__).resolve().parent.parent

        csv_log_path = Path(
            os.getenv(
                "CSV_LOG_PATH",
                str(BASE_DIR / "data/logs/session_history.csv"),
            )
        )

        bilstm_model_path = Path(
            os.getenv(
                "BILSTM_MODEL_PATH",
                str(BASE_DIR / "models/bilstm/bilstm_model.h5"),
            )
        )

        bert_model_path = Path(
            os.getenv(
                "BERT_MODEL_PATH",
                str(BASE_DIR / "models/bert/bert_model"),
            )
        )

        tokenizer_path = Path(
            os.getenv(
                "TOKENIZER_PATH",
                str(BASE_DIR / "models/tokenizer"),
            )
        )

        max_history_records = int(
            os.getenv("MAX_HISTORY_RECORDS", "500")
        )

        return cls(
            gemini_api_key=api_key,
            csv_log_path=csv_log_path,
            bilstm_model_path=bilstm_model_path,
            bert_model_path=bert_model_path,
            tokenizer_path=tokenizer_path,
            max_history_records=max_history_records,
        )