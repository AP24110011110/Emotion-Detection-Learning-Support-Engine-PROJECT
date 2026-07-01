from typing import Dict, List

EMOTION_LABELS: List[str] = [
    "Bored",
    "Confident",
    "Confused",
    "Curious",
    "Frustrated",
]

EMOTION_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
INDEX_TO_EMOTION: Dict[int, str] = {idx: label for label, idx in EMOTION_TO_INDEX.items()}

KEYWORD_BOOSTS: Dict[str, tuple[str, float]] = {
    "bored": ("Bored", 1.4),
    "boring": ("Bored", 1.3),
    "sleepy": ("Bored", 1.2),
    "dull": ("Bored", 1.2),
    "monotonous": ("Bored", 1.2),
    "uninterested": ("Bored", 1.3),
    "daydreaming": ("Bored", 1.1),
    "zoned": ("Bored", 1.1),
    "lost": ("Confused", 1.4),
    "confused": ("Confused", 1.4),
    "unclear": ("Confused", 1.3),
    "stuck": ("Confused", 1.3),
    "mixed": ("Confused", 1.1),
    "dont": ("Confused", 1.1),
    "understand": ("Confused", 1.1),
    "curious": ("Curious", 1.3),
    "interested": ("Curious", 1.3),
    "excited": ("Curious", 1.2),
    "explore": ("Curious", 1.2),
    "discover": ("Curious", 1.2),
    "fascinating": ("Curious", 1.2),
    "learn": ("Curious", 1.0),
    "frustrated": ("Frustrated", 1.4),
    "annoyed": ("Frustrated", 1.3),
    "angry": ("Frustrated", 1.3),
    "upset": ("Frustrated", 1.3),
    "stressed": ("Frustrated", 1.2),
    "nothing": ("Frustrated", 1.1),
    "works": ("Frustrated", 1.1),
    "tired": ("Frustrated", 1.1),
    "confident": ("Confident", 1.4),
    "understood": ("Confident", 1.4),
    "solved": ("Confident", 1.4),
    "easy": ("Confident", 1.2),
    "mastered": ("Confident", 1.3),
    "clear": ("Confident", 1.2),
    "comfortable": ("Confident", 1.2),
    "ready": ("Confident", 1.1),
}

CSV_COLUMNS: List[str] = [
    "timestamp",
    "student_input",
    "selected_field",
    "bilstm_prediction",
    "bilstm_confidence",
    "bert_prediction",
    "bert_confidence",
    "mixed_emotions",
    "final_prediction",
    "final_confidence",
    "gemini_response",
]

MIXED_EMOTION_THRESHOLD: float = 0.15

DEFAULT_BERT_MODEL_NAME: str = "bert-base-uncased"
DEFAULT_MAX_SEQUENCE_LENGTH: int = 128
