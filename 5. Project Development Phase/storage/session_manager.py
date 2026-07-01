from typing import Any, Dict, List, Optional

import streamlit as st

from pipeline.schema import PredictionResult


class SessionManager:
    """Manage Streamlit session state for chat history and current prediction."""

    HISTORY_KEY = "emotion_assistant_history"
    CURRENT_PREDICTION_KEY = "emotion_assistant_current_prediction"
    ERROR_KEY = "emotion_assistant_error"
    FIELD_KEY = "emotion_assistant_selected_field"

    @staticmethod
    def initialize_state(default_field: str = "General Learning") -> None:
        """Initialize missing Streamlit session state keys."""
        if SessionManager.HISTORY_KEY not in st.session_state:
            st.session_state[SessionManager.HISTORY_KEY] = []
        if SessionManager.CURRENT_PREDICTION_KEY not in st.session_state:
            st.session_state[SessionManager.CURRENT_PREDICTION_KEY] = None
        if SessionManager.ERROR_KEY not in st.session_state:
            st.session_state[SessionManager.ERROR_KEY] = ""
        if SessionManager.FIELD_KEY not in st.session_state:
            st.session_state[SessionManager.FIELD_KEY] = default_field

    @staticmethod
    def add_prediction(prediction: PredictionResult) -> None:
        """Add a completed prediction to the session history."""
        SessionManager.initialize_state()
        history = st.session_state[SessionManager.HISTORY_KEY]
        history.append(prediction)
        st.session_state[SessionManager.HISTORY_KEY] = history
        st.session_state[SessionManager.CURRENT_PREDICTION_KEY] = prediction
        st.session_state[SessionManager.ERROR_KEY] = ""

    @staticmethod
    def set_error(message: str) -> None:
        """Store an error message in session state."""
        SessionManager.initialize_state()
        st.session_state[SessionManager.ERROR_KEY] = message

    @staticmethod
    def get_error() -> str:
        """Retrieve the current error message."""
        SessionManager.initialize_state()
        return st.session_state.get(SessionManager.ERROR_KEY, "")

    @staticmethod
    def get_history() -> List[PredictionResult]:
        """Return the current session history."""
        SessionManager.initialize_state()
        return st.session_state[SessionManager.HISTORY_KEY]

    @staticmethod
    def get_current_prediction() -> Optional[PredictionResult]:
        """Return the most recent prediction if available."""
        SessionManager.initialize_state()
        return st.session_state.get(SessionManager.CURRENT_PREDICTION_KEY)

    @staticmethod
    def set_field(field: str) -> None:
        """Update the current selected field in session state."""
        SessionManager.initialize_state()
        st.session_state[SessionManager.FIELD_KEY] = field

    @staticmethod
    def get_field() -> str:
        """Return the currently selected field."""
        SessionManager.initialize_state()
        return st.session_state.get(SessionManager.FIELD_KEY, "General Learning")

    @staticmethod
    def reset_session() -> None:
        """Clear the chat history, current prediction, and error state."""
        st.session_state[SessionManager.HISTORY_KEY] = []
        st.session_state[SessionManager.CURRENT_PREDICTION_KEY] = None
        st.session_state[SessionManager.ERROR_KEY] = ""
        st.session_state[SessionManager.FIELD_KEY] = "General Learning"

    @staticmethod
    def history_to_dicts() -> List[Dict[str, Any]]:
        """Return session history as a list of dictionaries for display or export."""
        history = SessionManager.get_history()
        return [prediction.to_dict() for prediction in history if prediction is not None]
