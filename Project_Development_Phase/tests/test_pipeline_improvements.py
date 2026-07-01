import unittest

from pipeline.keyword_enhancer import enhance_prediction
from pipeline.mixed_emotion import summarize_mixed_emotions
from training.augmentation import TextAugmenter
from utils.constants import EMOTION_LABELS


class PipelineImprovementsTests(unittest.TestCase):
    def test_augmenter_produces_varied_text(self) -> None:
        augmenter = TextAugmenter(seed=7)
        variations = augmenter.augment("I am bored and confused by this lecture.")
        self.assertGreaterEqual(len(variations), 2)
        self.assertTrue(any("bored" in item.lower() or "confused" in item.lower() for item in variations))

    def test_keyword_enhancement_strengthens_distinct_emotions(self) -> None:
        base = {label: 0.2 for label in EMOTION_LABELS}
        enhanced = enhance_prediction(base, "I am confused and stuck on this recursion problem")
        self.assertGreater(enhanced["Confused"], enhanced["Confident"])
        self.assertGreater(enhanced["Confused"], enhanced["Bored"])

    def test_mixed_emotion_detection_handles_confused_and_curious(self) -> None:
        bilstm_probs = {
            "Bored": 0.05,
            "Confident": 0.1,
            "Confused": 0.42,
            "Curious": 0.33,
            "Frustrated": 0.1,
        }
        bert_probs = {
            "Bored": 0.03,
            "Confident": 0.08,
            "Confused": 0.38,
            "Curious": 0.41,
            "Frustrated": 0.1,
        }
        mixed = summarize_mixed_emotions(bilstm_probs, bert_probs, text="I do not understand recursion but I am excited to learn more")
        self.assertIn("Curious", mixed)
        self.assertIn("Confused", mixed)

    def test_mixed_emotion_detection_avoids_overreporting_secondary_emotions(self) -> None:
        bilstm_probs = {
            "Bored": 0.03,
            "Confident": 0.84,
            "Confused": 0.05,
            "Curious": 0.04,
            "Frustrated": 0.04,
        }
        bert_probs = {
            "Bored": 0.02,
            "Confident": 0.88,
            "Confused": 0.04,
            "Curious": 0.03,
            "Frustrated": 0.03,
        }
        mixed = summarize_mixed_emotions(bilstm_probs, bert_probs, text="I finally understood dynamic programming after practicing all week. I feel confident solving similar problems now.")
        self.assertEqual([], mixed)


if __name__ == "__main__":
    unittest.main()
