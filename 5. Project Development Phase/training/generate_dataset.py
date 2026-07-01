import random
from pathlib import Path

import pandas as pd


EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]


def build_samples() -> list[tuple[str, str]]:
    rng = random.Random(42)
    templates = {
        "Bored": [
            "I feel bored because {context}.",
            "This {context} is so boring that I cannot stay focused.",
            "I am uninterested in {context} because it feels repetitive and dull.",
            "The {context} is monotonous and I keep zoning out.",
            "I find {context} dull and unmotivating because {detail}.",
            "{context} feels sleepy and I cannot stay engaged.",
            "I am not interested in {context} because {detail}.",
            "This {context} is too slow paced and I keep daydreaming.",
            "I wish {context} were more stimulating because {detail}.",
            "The lesson on {context} is so repetitive that I lose focus.",
        ],
        "Confident": [
            "I feel confident because {context}.",
            "I finally understood {context} and I feel ready to proceed.",
            "I solved the {context} and I am comfortable explaining it.",
            "I have mastered {context} and I can apply it with ease.",
            "I am sure about {context} because {detail}.",
            "The solution for {context} is clear now and I feel prepared.",
            "I understand {context} well enough to teach it to someone else.",
            "I handled the {context} successfully and I feel assured.",
            "The approach for {context} now makes sense and I feel ready.",
            "I am comfortable with {context} because {detail}.",
        ],
        "Confused": [
            "I am confused about {context} because {detail}.",
            "I do not understand {context} and I feel stuck.",
            "This {context} is unclear and I am not sure how to proceed.",
            "I am lost in {context} because the instructions are vague.",
            "I have no idea how to start {context} because {detail}.",
            "The explanation for {context} is mixed up and I cannot follow it.",
            "I am unsure how to use {context} correctly in this assignment.",
            "I feel puzzled by {context} and I cannot separate the steps.",
            "The concept of {context} seems unclear to me in this problem.",
            "I am perplexed by {context} because {detail}.",
        ],
        "Curious": [
            "I am curious about {context} because {detail}.",
            "I want to explore {context} further and learn more.",
            "I am interested in understanding {context} in more depth.",
            "This {context} feels fascinating and I want to investigate it.",
            "I am eager to discover how {context} works in practice.",
            "I want to learn more about {context} and its real world applications.",
            "The idea behind {context} intrigues me and I want to study it.",
            "I am keen to find out how {context} behaves under different conditions.",
            "I would like to explore {context} because it seems promising.",
            "I am fascinated by {context} and I want to understand the reasoning.",
        ],
        "Frustrated": [
            "I am frustrated because {context} keeps failing.",
            "This {context} is making me upset because {detail}.",
            "I feel annoyed by {context} and nothing seems to work.",
            "The {context} is stressing me out and I feel stuck.",
            "I am fed up with {context} because {detail}.",
            "I am irritated by {context} and I cannot make progress.",
            "Nothing works in {context} and I feel exhausted by the errors.",
            "This {context} is so frustrating that I want to restart.",
            "I am angry about {context} because {detail}.",
            "The deadline for {context} is causing me a lot of stress.",
        ],
    }

    contexts = [
        "the lecture on recursion",
        "the coding assignment",
        "this mathematics proof",
        "the exam preparation",
        "the project deadline",
        "the presentation slides",
        "the online lab exercise",
        "the data science project",
        "the debugging session",
        "the statistics homework",
        "the algorithm explanation",
        "the machine learning model",
        "the physics problem set",
        "the engineering design task",
        "the programming challenge",
        "the research paper review",
        "the group assignment",
        "the self study module",
        "the exam review notes",
        "the quiz preparation",
    ]

    details = [
        "the examples are too repetitive",
        "the instructions are not clear enough",
        "the pace is too slow for my level",
        "the task feels too predictable",
        "the same concept is explained too many times",
        "the explanation uses unfamiliar words",
        "I can already see the pattern",
        "the reasoning finally clicked",
        "I can explain the result to others",
        "the output matches my expectation",
        "I need more examples before I can proceed",
        "the logic is hard to connect to the examples",
        "the error keeps appearing even after retries",
        "the tools are behaving inconsistently",
        "the deadline is approaching quickly",
        "the feedback is too generic",
        "the implementation keeps breaking",
        "I want to understand the intuition behind it",
        "the problem is more interesting than the lecture",
        "the assignment is challenging but manageable",
    ]

    samples: list[tuple[str, str]] = []
    target_per_emotion = 1200
    for emotion in EMOTIONS:
        for index in range(target_per_emotion):
            context = rng.choice(contexts)
            detail = rng.choice(details)
            template = rng.choice(templates[emotion])
            sentence = template.format(context=context, detail=detail)
            if index % 3 == 0:
                sentence = sentence.replace(".", "!")
            if index % 5 == 0:
                sentence = sentence.replace("I ", "I really ", 1)
            sentence = sentence + f" ({index + 1})"
            samples.append((sentence, emotion))

    rng.shuffle(samples)
    return samples


def generate_dataset(output_path: Path) -> None:
    samples = build_samples()
    dataframe = pd.DataFrame(samples, columns=["text", "emotion"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    generate_dataset(Path("data/raw/emotion_data.csv"))
