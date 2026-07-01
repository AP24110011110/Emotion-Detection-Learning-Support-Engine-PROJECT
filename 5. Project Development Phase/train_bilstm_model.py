"""
BiLSTM Model Training Script
Trains an emotion classifier on educational learning problems.
Exports model to models/bilstm/bilstm_model.h5
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from training.dataset import EmotionDataset
from training.train_bilstm import train_bilstm
from utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_synthetic_dataset(output_path: Path, num_samples_per_emotion: int = 200) -> Path:
    """Generate a synthetic emotion dataset for quick training and testing."""
    
    emotion_templates = {
        "Confused": [
            "I don't understand how recursion works. The concepts seem too abstract.",
            "The assignment is confusing. I read the instructions multiple times but it still doesn't make sense.",
            "I'm lost on the difference between stack and queue data structures.",
            "This algorithm explanation is unclear. I can't follow the logic.",
            "I'm struggling to understand why this approach is better than that one.",
            "The mathematical notation is confusing me.",
            "I can't figure out where I went wrong in my code.",
            "The relationship between these concepts is unclear to me.",
            "I'm confused about when to use inheritance vs composition.",
            "The tutorial doesn't explain the core idea well enough.",
        ],
        "Frustrated": [
            "I've been stuck on this problem for hours with no progress.",
            "This is so frustrating! I keep getting the same error.",
            "Why is this so hard? Everyone else seems to get it.",
            "I'm angry at myself for not understanding this sooner.",
            "The error message doesn't help. I don't know how to fix this.",
            "I'm fed up with debugging. Nothing I try works.",
            "This assignment is impossible. I don't even know where to start.",
            "I'm frustrated that the documentation is so vague.",
            "Why can't things just work? I'm tired of these bugs.",
            "This is making me lose confidence in my abilities.",
        ],
        "Curious": [
            "How does machine learning actually work under the hood?",
            "I'm interested in learning more about blockchain technology.",
            "What's the best way to approach this problem?",
            "I'd like to explore optimization techniques for my algorithm.",
            "Can you explain how neural networks learn patterns?",
            "I'm curious about the trade-offs between different data structures.",
            "How would I scale this solution to handle larger datasets?",
            "What are some advanced techniques used in this field?",
            "I want to understand the mathematical foundation of this concept.",
            "Can you suggest some interesting projects to try?",
        ],
        "Confident": [
            "I understand the core concepts and feel ready to tackle this.",
            "I've mastered the fundamentals and can apply them confidently.",
            "This problem is straightforward now that I understand the pattern.",
            "I'm confident in my solution. I've tested it thoroughly.",
            "I know exactly what approach to take for this challenge.",
            "I'm certain this implementation is correct and efficient.",
            "I feel comfortable explaining this to others.",
            "I've solved similar problems before, so I'm confident here.",
            "My code works well and I understand every part of it.",
            "I'm ready to move on to more advanced topics.",
        ],
        "Bored": [
            "This topic is too basic. I already know all of this.",
            "I'm not interested in this part of the curriculum.",
            "This lesson is too slow-paced for me.",
            "I find the repetitive exercises dull and unmotivating.",
            "When do we get to something more interesting?",
            "I'm bored with the same type of problem over and over.",
            "This material doesn't challenge me at all.",
            "I've seen this concept explained a hundred times already.",
            "Can we move past the basics? I need something more stimulating.",
            "I'm not engaged with this topic. It feels like a waste of time.",
        ],
    }

    data = []
    for emotion, templates in emotion_templates.items():
        for _ in range(num_samples_per_emotion // len(templates) + 1):
            for template in templates:
                if len(data) < num_samples_per_emotion * len(emotion_templates):
                    data.append({"text": template, "emotion": emotion})

    df = pd.DataFrame(data[: num_samples_per_emotion * len(emotion_templates)])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset created: {output_path} ({len(df)} samples)")
    return output_path


def main():
    """Load dataset, train BiLSTM, and export model."""
    config = Config.load()

    raw_data_path = Path("data/raw/emotion_data.csv")

    if not raw_data_path.exists():
        logger.info("Dataset not found. Generating synthetic dataset...")
        create_synthetic_dataset(raw_data_path, num_samples_per_emotion=200)
    else:
        logger.info(f"Using existing dataset: {raw_data_path}")

    logger.info("Starting BiLSTM training...")
    result = train_bilstm(
        dataset_path=raw_data_path,
        tokenizer_path=config.tokenizer_path / "bilstm_tokenizer.json",
        model_path=config.bilstm_model_path,
        num_words=10000,
        max_length=128,
        batch_size=32,
        epochs=12,
    )

    logger.info(f"BiLSTM training complete!")
    logger.info(f"Model saved to: {result['model_path']}")
    logger.info(f"Tokenizer saved to: {result['tokenizer_path']}")


if __name__ == "__main__":
    main()
