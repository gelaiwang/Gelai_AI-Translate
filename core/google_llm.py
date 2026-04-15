from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import google.generativeai as legacy_genai  # type: ignore
except ImportError:  # pragma: no cover
    legacy_genai = None  # type: ignore

try:
    from google import genai as vertex_genai  # type: ignore
    from google.genai import types as vertex_types  # type: ignore
except ImportError:  # pragma: no cover
    vertex_genai = None  # type: ignore
    vertex_types = None  # type: ignore

from config import (
    GEMINI_API_KEY,
    VERTEX_API_VERSION,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
)

GOOGLE_LLM_SERVICES = {"gemini", "vertex"}
GOOGLE_SERVICE_DISPLAY_NAMES = {
    "gemini": "Gemini",
    "vertex": "Google Vertex AI",
}
GOOGLE_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


@dataclass
class GoogleTextClient:
    service: str
    model_name: str
    client: Any


def is_google_service(service: str) -> bool:
    return (service or "").strip().lower() in GOOGLE_LLM_SERVICES


def get_google_service_display_name(service: str) -> str:
    return GOOGLE_SERVICE_DISPLAY_NAMES.get((service or "").strip().lower(), service)


def build_google_text_client(service: str, model_name: str) -> GoogleTextClient:
    service = (service or "").strip().lower()
    if not model_name:
        raise RuntimeError("Google LLM model name is not configured.")

    if service == "gemini":
        if legacy_genai is None:
            raise RuntimeError("Missing dependency: google-generativeai")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        legacy_genai.configure(api_key=GEMINI_API_KEY)
        model = legacy_genai.GenerativeModel(model_name, safety_settings=GOOGLE_SAFETY_SETTINGS)
        return GoogleTextClient(service=service, model_name=model_name, client=model)

    if service == "vertex":
        if vertex_genai is None or vertex_types is None:
            raise RuntimeError("Missing dependency: google-genai")
        if not VERTEX_PROJECT:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured for Vertex AI.")
        if not VERTEX_LOCATION:
            raise RuntimeError("GOOGLE_CLOUD_LOCATION is not configured for Vertex AI.")
        client = vertex_genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
            http_options=vertex_types.HttpOptions(api_version=VERTEX_API_VERSION),
        )
        return GoogleTextClient(service=service, model_name=model_name, client=client)

    raise RuntimeError(f"Unsupported Google LLM service: {service}")


def generate_google_text(client: GoogleTextClient, prompt: str, temperature: float) -> str:
    if client.service == "gemini":
        generation_config = legacy_genai.types.GenerationConfig(temperature=temperature)
        response = client.client.generate_content(prompt, generation_config=generation_config)
        if not response.parts:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason:
                raise RuntimeError(f"Gemini blocked: {block_reason}")
            raise RuntimeError("Gemini returned empty content.")
        return response.text.strip()

    if client.service == "vertex":
        config = vertex_types.GenerateContentConfig(temperature=temperature)
        response = client.client.models.generate_content(
            model=client.model_name,
            contents=prompt,
            config=config,
        )
        text = (getattr(response, "text", "") or "").strip()
        if text:
            return text
        raise RuntimeError("Vertex AI returned empty content.")

    raise RuntimeError(f"Unsupported Google LLM service: {client.service}")
