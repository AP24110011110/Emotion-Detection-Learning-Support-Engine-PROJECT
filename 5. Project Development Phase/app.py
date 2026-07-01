import logging
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from dashboard.analytics import (
    build_summary,
    compare_model_predictions,
    compute_average_confidence,
    compute_average_confidence_by_emotion,
    compute_confidence_trend,
    compute_daily_statistics,
    compute_emotion_frequency,
    compute_emotion_timeline,
    compute_mixed_emotion_frequency,
    compute_subject_distribution,
    read_history,
)
from pipeline.schema import PredictionResult
from dashboard.charts import (
    confidence_by_emotion_chart,
    confidence_scores_chart,
    confidence_trend_chart,
    daily_trend_chart,
    emotion_distribution_chart,
    emotion_timeline_chart,
    mixed_emotion_chart,
    model_comparison_chart,
    subject_distribution_chart,
)
from gemini.gemini_client import GeminiClient
from gemini.response_generator import GeminiResponseGenerator
from pipeline.predictor import build_predictor, EmotionPredictor
from storage.csv_logger import CsvLogger
from storage.session_manager import SessionManager
from utils.config import Config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_FIELDS: List[str] = [
    "Computer Science",
    "Mathematics",
    "Data Science",
    "Physics",
    "Engineering",
    "Statistics",
    "Artificial Intelligence",
    "General Learning",
]


def configure_page() -> None:
    st.set_page_config(
        page_title="Emotion Learning Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>footer {visibility: hidden;} .reportview-container .main .block-container {padding-top: 1rem;}</style>",
        unsafe_allow_html=True,
    )


def load_config() -> Config:
    return Config.load()


@st.cache_resource
def get_predictor(config: Config) -> EmotionPredictor:
    return build_predictor(
        config=config,
        bilstm_tokenizer_path=config.tokenizer_path / "bilstm_tokenizer.json",
        bert_tokenizer_path=config.tokenizer_path / "bert_tokenizer",
        max_length=128,
    )


@st.cache_resource
def get_gemini_generator(config: Config) -> GeminiResponseGenerator:
    client = GeminiClient(api_key=config.gemini_api_key)
    return GeminiResponseGenerator(gemini_client=client)


@st.cache_resource
def get_csv_logger(config: Config) -> CsvLogger:
    return CsvLogger(Path(config.csv_log_path))


def render_sidebar(selected_field: str) -> str:
    st.sidebar.title("Emotion Assistant")
    st.sidebar.markdown(
        "This application analyzes student learning problems, detects emotional state with BiLSTM and BERT, and generates a supportive response using Gemini."
    )
    st.sidebar.markdown("---")
    chosen_field = st.sidebar.selectbox("Select your subject area", PROJECT_FIELDS, index=PROJECT_FIELDS.index(selected_field) if selected_field in PROJECT_FIELDS else 0)
    st.sidebar.markdown(
        "#### How to use\n"
        "1. Enter a short description of your learning challenge.\n"
        "2. Click Analyze Emotion.\n"
        "3. Review the emotion insights, model comparison, and personalized guidance."
    )
    return chosen_field


def render_prediction_panel(title: str, label: str, confidence: float, custom_message: str) -> None:
    st.subheader(title)
    st.metric(label="Emotion", value=label, delta=f"{confidence:.0%} confidence")
    progress = int(confidence * 100)
    st.progress(progress)
    st.caption(custom_message)


def display_results(prediction: "PredictionResult") -> None:
    st.markdown("## Emotion Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        render_prediction_panel(
            "BiLSTM Prediction",
            prediction.bilstm_prediction.top_label,
            prediction.bilstm_prediction.confidence,
            "BiLSTM probability distribution is adjusted with keyword context.",
        )

    with col2:
        render_prediction_panel(
            "BERT Prediction",
            prediction.bert_prediction.top_label,
            prediction.bert_prediction.confidence,
            "BERT uses contextual embeddings for deeper emotion inference.",
        )

    with col3:
        mixed_title = "Mixed Emotions Detected" if prediction.mixed_emotions else "Primary Emotion Only"
        st.subheader(mixed_title)
        if prediction.mixed_emotions:
            for emotion in prediction.mixed_emotions:
                st.write(f"- {emotion}")
        else:
            st.write("No strong secondary emotions detected.")
        st.write(f"Final prediction: **{prediction.final_prediction}**")
        st.write(f"Final confidence: **{prediction.final_confidence:.2%}**")

    st.markdown("---")
    st.markdown("## Personalized Guidance")
    st.write(prediction.gemini_response or "No guidance available.")


def render_history_table(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        st.info("No history is available yet. Start a new analysis to log predictions.")
        return

    st.dataframe(history_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True), use_container_width=True)


def render_analytics(history_df: pd.DataFrame) -> None:
    st.markdown("## Analytics Overview")
    summary = build_summary(history_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total interactions", summary["total_interactions"])
    col2.metric("Tracked subjects", summary["unique_fields"])
    col3.metric("Average confidence", f"{summary['average_confidence']:.2%}")

    emotion_freq = compute_emotion_frequency(history_df)
    confidence_avg = compute_average_confidence(history_df)
    comparison = compare_model_predictions(history_df)
    daily_stats = compute_daily_statistics(history_df)
    mixed_freq = compute_mixed_emotion_frequency(history_df)

    st.plotly_chart(emotion_distribution_chart(emotion_freq), use_container_width=True)
    st.plotly_chart(confidence_scores_chart(confidence_avg), use_container_width=True)
    st.plotly_chart(model_comparison_chart(comparison), use_container_width=True)
    st.plotly_chart(daily_trend_chart(daily_stats), use_container_width=True)
    st.plotly_chart(mixed_emotion_chart(mixed_freq), use_container_width=True)
    st.plotly_chart(confidence_trend_chart(compute_confidence_trend(history_df)), use_container_width=True)
    st.plotly_chart(subject_distribution_chart(compute_subject_distribution(history_df)), use_container_width=True)
    st.plotly_chart(emotion_timeline_chart(compute_emotion_timeline(history_df)), use_container_width=True)
    st.plotly_chart(confidence_by_emotion_chart(compute_average_confidence_by_emotion(history_df)), use_container_width=True)


def run_analysis(
    user_input: str,
    selected_field: str,
    predictor: EmotionPredictor,
    gemini_generator: GeminiResponseGenerator,
    csv_logger: CsvLogger,
) -> None:
    if not user_input.strip():
        SessionManager.set_error("Please enter a learning problem to analyze.")
        return

    try:
        with st.spinner("Analyzing emotion and generating guidance..."):
            prediction = predictor.predict(user_input, selected_field)
            prediction.gemini_response = gemini_generator.generate_response(prediction)
            csv_logger.validate_record(prediction.to_dict())
            csv_logger.append_record(prediction.to_dict())
            SessionManager.add_prediction(prediction)
            SessionManager.set_error("")
    except Exception as exc:
        logger.exception("Prediction or logging failed.")
        SessionManager.set_error("Unable to complete analysis. Check model files and .env settings.")


def main() -> None:
    configure_page()

    try:
        config = load_config()
    except Exception as exc:
        st.error(f"Configuration load failed: {exc}")
        return

    SessionManager.initialize_state()
    selected_field = render_sidebar(SessionManager.get_field())
    SessionManager.set_field(selected_field)

    try:
        predictor = get_predictor(config)
        gemini_generator = get_gemini_generator(config)
        csv_logger = get_csv_logger(config)
    except Exception as exc:
        logger.exception("Failed to initialize application resources.")
        st.error(f"Application initialization failed: {exc}")
        return

    st.header("Describe your learning challenge")
    student_input = st.text_area(
        "Student problem description",
        value=st.session_state.get("student_input", ""),
        height=220,
        placeholder="I feel confused about the latest assignment because...",
        key="student_input",
    )

    col1, col2 = st.columns([1, 1])
    analyze_clicked = col1.button("Analyze Emotion", type="primary")
    clear_clicked = col2.button("Clear Session")

    if clear_clicked:
        SessionManager.reset_session()
        st.experimental_rerun()

    if analyze_clicked:
        run_analysis(student_input, selected_field, predictor, gemini_generator, csv_logger)

    error_message = SessionManager.get_error()
    if error_message:
        st.error(error_message)

    current_prediction = SessionManager.get_current_prediction()
    tab1, tab2, tab3 = st.tabs(["Assistant", "Analytics", "History"])

    with tab1:
        if current_prediction:
            display_results(current_prediction)
        else:
            st.info("Run an analysis to see emotion insights and personalized guidance.")

    history_df = read_history(Path(config.csv_log_path))
    with tab2:
        render_analytics(history_df)

    with tab3:
        st.markdown("## Session History")
        render_history_table(history_df)
        if not history_df.empty:
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download prediction history as CSV",
                data=csv_bytes,
                file_name="emotion_prediction_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
