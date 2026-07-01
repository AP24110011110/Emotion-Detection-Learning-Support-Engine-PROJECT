from pipeline.schema import PredictionResult


def build_gemini_prompt(prediction: PredictionResult) -> str:
    """Construct a Gemini prompt that generates an empathetic, field-aware response."""
    mixed_emotion_section = (
        f"The student also shows mixed emotions: {', '.join(prediction.mixed_emotions)}."
        if prediction.mixed_emotions
        else "The student shows a single primary emotion."
    )

    return (
        "You are an empathetic academic coach supporting a student who is learning a technical topic. "
        "Use a calm, encouraging tone and provide practical next steps.\n\n"
        f"Student input: \"{prediction.student_input}\"\n"
        f"Subject area: {prediction.selected_field}\n"
        f"Emotion analysis: {prediction.final_prediction} ({prediction.final_confidence:.0%} confidence).\n"
        f"Model insights: BiLSTM predicted {prediction.bilstm_prediction.top_label} ({prediction.bilstm_prediction.confidence:.0%}), "
        f"BERT predicted {prediction.bert_prediction.top_label} ({prediction.bert_prediction.confidence:.0%}).\n"
        f"{mixed_emotion_section}\n\n"
        "Please respond with a concise, supportive explanation that:\n"
        "- validates the student’s feelings,\n"
        "- suggests one or two actionable learning strategies,\n"
        "- optionally includes a simple emotional coping suggestion,\n"
        "- avoids technical jargon unless it is explained clearly.\n\n"
        "Keep the response under 200 words and make it feel personal and reassuring."
    )


def build_regeneration_prompt(prediction: PredictionResult, revision_reason: str) -> str:
    """Construct a prompt for regenerating the Gemini response with an explicit improvement goal."""
    base_prompt = build_gemini_prompt(prediction)
    return (
        f"{base_prompt}\n\n"
        f"Regeneration request: {revision_reason}. "
        "If possible, add a slightly more empathetic tone and clarify the learning strategy."
    )
