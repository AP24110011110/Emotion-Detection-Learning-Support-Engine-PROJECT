import logging
from typing import Optional

from gemini.gemini_client import GeminiClient
from gemini.prompt_builder import build_gemini_prompt, build_regeneration_prompt
from pipeline.schema import PredictionResult


class GeminiResponseGenerator:
    """Generate and regenerate text responses using the Gemini API."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        model: Optional[str] = None,
        max_output_tokens: int = 256,
        temperature: float = 0.75,
    ) -> None:
        self.client = gemini_client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def generate_response(self, prediction: PredictionResult) -> str:
        """Generate a Gemini response for the given prediction."""
        prompt = build_gemini_prompt(prediction)
        try:
            response_text = self.client.generate_text(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            return response_text.strip() or self.fallback_response(prediction)
        except Exception as exc:
            logging.exception("Gemini generation failed")
            return self.fallback_response(prediction)

    def regenerate_response(self, prediction: PredictionResult, revision_reason: str) -> str:
        """Regenerate the Gemini response with an explicit improvement guideline."""
        prompt = build_regeneration_prompt(prediction, revision_reason)
        try:
            response_text = self.client.generate_text(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            return response_text.strip() or self.fallback_response(prediction)
        except Exception as exc:
            logging.exception("Gemini regeneration failed")
            return self.fallback_response(prediction)

    @staticmethod
    def fallback_response(prediction: PredictionResult) -> str:
        """Return a safe fallback response if Gemini is unavailable."""
        emotions = ", ".join(prediction.mixed_emotions) if prediction.mixed_emotions else prediction.final_prediction
        return (
            f"I hear that you are feeling {emotions.lower()} about your {prediction.selected_field} work. "
            "It can help to take a short break, identify one small learning goal, and ask for help on a specific part of the problem. "
            "Remember that confusion and frustration are natural when learning something new, and a steady step-by-step approach often makes progress feel more manageable."
        )
