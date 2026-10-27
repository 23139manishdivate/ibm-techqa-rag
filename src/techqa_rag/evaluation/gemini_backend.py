"""Gemini API adapter for controlled TechQA reader evaluations."""

from __future__ import annotations

import os
import time
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def create_gemini_client() -> Any:
    """Create a Google GenAI client without logging or persisting its API key."""
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("gemini_api_key")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY as a user environment variable, "
            "then restart the terminal or application that runs this script."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini support requires the Google GenAI SDK. Install it with: "
            "pip install -r requirements.txt"
        ) from exc
    return genai.Client(api_key=api_key)


def generate_with_gemini(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    question: str,
    context: str,
    max_output_tokens: int,
    retries: int = 3,
) -> dict[str, Any]:
    """Generate one answer and return text plus API-reported token usage."""
    from google.genai import types

    user_message = (
        f"Question:\n{question}\n\n"
        f"Documentation context:\n{context}\n\n"
        "Give a direct answer supported by the context."
    )
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            return {
                "answer": (response.text or "").strip(),
                "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
                "generated_token_count": int(output_tokens) if output_tokens is not None else None,
            }
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            status_text = str(status).upper()
            retryable = (
                status in {429, 500, 502, 503, 504}
                or any(code in status_text for code in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            )
            if not retryable or attempt >= retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 16))
    raise RuntimeError("Gemini generation exhausted retries without a response")
