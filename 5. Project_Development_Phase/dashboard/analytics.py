import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from utils.constants import CSV_COLUMNS

logger = logging.getLogger(__name__)


def read_history(csv_path: Path) -> pd.DataFrame:
    """Read prediction history from the CSV log with graceful fallback for missing or empty files."""
    if not csv_path.exists():
        logger.warning("History CSV file not found at %s. Returning empty history.", csv_path)
        return pd.DataFrame(columns=CSV_COLUMNS)

    try:
        df = pd.read_csv(csv_path, dtype=str)
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)
        for column in CSV_COLUMNS:
            if column not in df.columns:
                df[column] = ""
        df = df[CSV_COLUMNS]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as exc:
        logger.exception("Failed to read history CSV from %s", csv_path)
        return pd.DataFrame(columns=CSV_COLUMNS)


def compute_emotion_frequency(history: pd.DataFrame) -> pd.DataFrame:
    """Compute counts and percentages for final emotion predictions."""
    if history.empty or "final_prediction" not in history.columns:
        return pd.DataFrame(columns=["emotion", "count", "percentage"])

    counts = history["final_prediction"].value_counts(dropna=True)
    total = counts.sum() if counts.sum() else 1
    frequency = pd.DataFrame(
        {
            "emotion": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(2),
        }
    )
    return frequency.reset_index(drop=True)


def compute_average_confidence(history: pd.DataFrame) -> pd.DataFrame:
    """Compute average confidences for BiLSTM, BERT, and final predictions."""
    if history.empty:
        return pd.DataFrame(columns=["model", "average_confidence"])

    confidence_cols = ["bilstm_confidence", "bert_confidence", "final_confidence"]
    averages = []
    for column in confidence_cols:
        if column in history.columns:
            series = pd.to_numeric(history[column], errors="coerce")
            averages.append(
                {
                    "model": column.replace("_confidence", "").upper(),
                    "average_confidence": round(series.mean(skipna=True) if not series.dropna().empty else 0.0, 4),
                }
            )
    return pd.DataFrame(averages)


def compare_model_predictions(history: pd.DataFrame) -> pd.DataFrame:
    """Compare BiLSTM and BERT predictions and compute agreement statistics."""
    if history.empty or "bilstm_prediction" not in history.columns or "bert_prediction" not in history.columns:
        return pd.DataFrame(columns=["comparison", "count"])

    agreement = history[history["bilstm_prediction"] == history["bert_prediction"]].shape[0]
    disagreement = history[history["bilstm_prediction"] != history["bert_prediction"]].shape[0]
    return pd.DataFrame(
        [
            {"comparison": "agreement", "count": agreement},
            {"comparison": "disagreement", "count": disagreement},
        ]
    )


def compute_daily_statistics(history: pd.DataFrame) -> pd.DataFrame:
    """Compute daily counts, average confidences, and cumulative totals."""
    if history.empty or "timestamp" not in history.columns:
        return pd.DataFrame(columns=["date", "count", "average_confidence", "cumulative_count"])

    valid_history = history.dropna(subset=["timestamp"]).copy()
    valid_history["date"] = valid_history["timestamp"].dt.date
    if valid_history.empty:
        return pd.DataFrame(columns=["date", "count", "average_confidence", "cumulative_count"])

    daily = (
        valid_history.groupby("date")
        .agg(
            count=("timestamp", "count"),
            average_confidence=("final_confidence", lambda values: round(pd.to_numeric(values, errors="coerce").mean(skipna=True) if not values.dropna().empty else 0.0, 4)),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["cumulative_count"] = daily["count"].cumsum()
    return daily


def compute_mixed_emotion_frequency(history: pd.DataFrame) -> pd.DataFrame:
    """Compute how often mixed emotions occur and which labels appear most often."""
    if history.empty or "mixed_emotions" not in history.columns:
        return pd.DataFrame(columns=["mixed_emotion", "count"])

    mixed_series = history["mixed_emotions"].fillna("")
    mixed_counts: Dict[str, int] = {}
    for value in mixed_series:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
        for emotion in items:
            mixed_counts[emotion] = mixed_counts.get(emotion, 0) + 1

    if not mixed_counts:
        return pd.DataFrame(columns=["mixed_emotion", "count"])

    frequency = pd.DataFrame(
        {
            "mixed_emotion": list(mixed_counts.keys()),
            "count": list(mixed_counts.values()),
        }
    ).sort_values("count", ascending=False)
    return frequency.reset_index(drop=True)


def compute_subject_distribution(history: pd.DataFrame) -> pd.DataFrame:
    """Compute how often each subject field appears in the history."""
    if history.empty or "selected_field" not in history.columns:
        return pd.DataFrame(columns=["subject", "count"])

    counts = history["selected_field"].fillna("General Learning").value_counts().reset_index()
    counts.columns = ["subject", "count"]
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def compute_confidence_trend(history: pd.DataFrame) -> pd.DataFrame:
    """Compute daily average confidence and a rolling weekly average."""
    if history.empty or "timestamp" not in history.columns:
        return pd.DataFrame(columns=["date", "average_confidence", "rolling_confidence"])

    valid_history = history.dropna(subset=["timestamp"]).copy()
    valid_history["timestamp"] = pd.to_datetime(valid_history["timestamp"], errors="coerce")
    valid_history = valid_history.dropna(subset=["timestamp"])
    valid_history["date"] = valid_history["timestamp"].dt.date
    if valid_history.empty:
        return pd.DataFrame(columns=["date", "average_confidence", "rolling_confidence"])

    daily = (
        valid_history.groupby("date")
        .agg(average_confidence=("final_confidence", lambda values: round(pd.to_numeric(values, errors="coerce").mean(skipna=True), 4)))
        .reset_index()
    )
    daily = daily.sort_values("date")
    daily["rolling_confidence"] = daily["average_confidence"].rolling(window=7, min_periods=1).mean()
    return daily


def compute_emotion_timeline(history: pd.DataFrame) -> pd.DataFrame:
    """Count emotion predictions over time for timeline-style analytics."""
    if history.empty or "timestamp" not in history.columns or "final_prediction" not in history.columns:
        return pd.DataFrame(columns=["date", "emotion", "count"])

    valid_history = history.dropna(subset=["timestamp", "final_prediction"]).copy()
    valid_history["timestamp"] = pd.to_datetime(valid_history["timestamp"], errors="coerce")
    valid_history = valid_history.dropna(subset=["timestamp"])
    valid_history["date"] = valid_history["timestamp"].dt.date
    return valid_history.groupby(["date", "final_prediction"]).size().reset_index(name="count")


def compute_average_confidence_by_emotion(history: pd.DataFrame) -> pd.DataFrame:
    """Return average confidence broken down by predicted emotion."""
    if history.empty or "final_prediction" not in history.columns or "final_confidence" not in history.columns:
        return pd.DataFrame(columns=["emotion", "average_confidence"])

    grouped = history.groupby("final_prediction")["final_confidence"].apply(lambda values: round(pd.to_numeric(values, errors="coerce").mean(skipna=True), 4)).reset_index()
    grouped.columns = ["emotion", "average_confidence"]
    return grouped.sort_values("average_confidence", ascending=False).reset_index(drop=True)


def build_summary(history: pd.DataFrame) -> Dict[str, object]:
    """Return a dictionary with summary statistics for dashboard metrics."""
    average_confidence_df = compute_average_confidence(history)
    avg_confidence = 0.0
    if not average_confidence_df.empty:
        avg_confidence = float(average_confidence_df["average_confidence"].mean())
        if pd.isna(avg_confidence):
            avg_confidence = 0.0

    return {
        "total_interactions": int(history.shape[0]),
        "unique_fields": int(history["selected_field"].nunique()) if "selected_field" in history.columns else 0,
        "average_confidence": round(avg_confidence, 4),
        "emotions_tracked": compute_emotion_frequency(history)["emotion"].tolist(),
        "agreement_percentage": round(compare_model_predictions(history).loc[compare_model_predictions(history)["comparison"] == "agreement", "count"].sum() / max(1, history.shape[0]) * 100, 2) if not history.empty else 0.0,
    }
