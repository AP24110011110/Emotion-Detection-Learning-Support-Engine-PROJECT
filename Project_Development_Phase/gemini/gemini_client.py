import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class GeminiClient:
    """Lightweight wrapper over the Gemini text generation API."""

    api_key: str
    base_url: str = "https://gemini.googleapis.com/v1/models/{model}:generate"
    default_model: str = "gemini-1.0"
    timeout: int = 30

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key.startswith("ya29.") or self.api_key.startswith("goog"):
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_params(self) -> Dict[str, str]:
        if not (self.api_key.startswith("ya29.") or self.api_key.startswith("goog")):
            return {"key": self.api_key}
        return {}

    def _build_url(self, model: Optional[str] = None) -> str:
        model_name = model or self.default_model
        return self.base_url.format(model=model_name)

    def _request(self, payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        url = self._build_url(model=model)
        headers = self._build_headers()
        params = self._build_params()

        response = requests.post(url, json=payload, headers=headers, params=params, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logging.error("Gemini API returned an error: %s", response.text)
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Gemini API returned an unexpected response format.")
        return data

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.8,
        max_output_tokens: int = 256,
    ) -> str:
        """Generate text from Gemini using the provided prompt."""
        payload = {
            "prompt": {"text": prompt},
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        response = self._request(payload, model=model)

        if "candidates" in response and isinstance(response["candidates"], list):
            candidate = response["candidates"][0]
            if isinstance(candidate, dict) and "content" in candidate:
                return str(candidate["content"]).strip()

        if "output" in response and isinstance(response["output"], dict):
            return str(response["output"].get("content", "")).strip()

        raise RuntimeError("Gemini API response did not contain generated text.")
