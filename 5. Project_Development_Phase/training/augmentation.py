import random
import re
from typing import List, Sequence


class TextAugmenter:
    """Create lightweight label-preserving text variations for training."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)
        self.synonym_map = {
            "bored": ["dull", "uninterested", "sleepy", "restless", "disengaged"],
            "confident": ["assured", "certain", "comfortable", "ready", "secure"],
            "confused": ["puzzled", "unclear", "mixed up", "unsure", "perplexed"],
            "curious": ["eager", "intrigued", "interested", "fascinated", "keen"],
            "frustrated": ["annoyed", "upset", "stressed", "irritated", "fed up"],
            "understand": ["grasp", "comprehend", "follow"],
            "learn": ["study", "explore", "master"],
            "problem": ["issue", "challenge", "task"],
            "difficult": ["hard", "tough", "tricky"],
        }
        self.insertions = ["really", "quite", "honestly", "clearly", "still"]
        self.punctuation_variants = [".", "!", "?", "...", "!!"]

    def augment(self, text: str, max_variations: int = 2) -> List[str]:
        """Return a small set of augmented variations for a single text sample."""
        if not text:
            return []

        text = str(text).strip()
        variations = {text}
        current = text
        for _ in range(max_variations):
            candidate = self._apply_random_augment(current)
            if candidate and candidate not in variations:
                variations.add(candidate)
                current = candidate
        return list(variations)

    def augment_many(self, texts: Sequence[str], max_variations: int = 2) -> List[str]:
        """Augment a sequence of texts and return the expanded list."""
        expanded: List[str] = []
        for text in texts:
            expanded.extend(self.augment(text, max_variations=max_variations))
        return expanded

    def _apply_random_augment(self, text: str) -> str:
        tokens = text.split()
        if not tokens:
            return text

        candidate = list(tokens)
        if self.random.random() < 0.5:
            candidate = self._replace_words(candidate)
        if self.random.random() < 0.5:
            candidate = self._drop_words(candidate)
        if self.random.random() < 0.5:
            candidate = self._insert_words(candidate)
        if self.random.random() < 0.5:
            candidate = self._shuffle_punctuation(candidate)
        if self.random.random() < 0.5:
            candidate = self._apply_casing(candidate)

        augmented = " ".join(candidate).strip()
        augmented = re.sub(r"\s+", " ", augmented)
        return augmented

    def _replace_words(self, tokens: List[str]) -> List[str]:
        replaced = []
        for token in tokens:
            clean_token = re.sub(r"[^a-zA-Z]", "", token.lower())
            if clean_token in self.synonym_map and self.random.random() < 0.35:
                replaced.append(self.random.choice(self.synonym_map[clean_token]))
            else:
                replaced.append(token)
        return replaced

    def _drop_words(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if self.random.random() >= 0.12]

    def _insert_words(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return tokens
        if self.random.random() < 0.3:
            position = self.random.randint(0, len(tokens))
            tokens = tokens[:position] + [self.random.choice(self.insertions)] + tokens[position:]
        return tokens

    def _shuffle_punctuation(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return tokens
        last_index = len(tokens) - 1
        tokens[last_index] = tokens[last_index] + self.random.choice(self.punctuation_variants)
        return tokens

    def _apply_casing(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return tokens
        updated = []
        for index, token in enumerate(tokens):
            if self.random.random() < 0.25 and token.isalpha():
                updated.append(token.capitalize())
            else:
                updated.append(token)
        return updated
