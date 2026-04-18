from __future__ import annotations

import re
from typing import Optional

from openai import OpenAI

from config import (
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_API_KEY,
    GEMINI_API_KEY,
    GROK_API_KEY,
    LOCAL_API_BASE_URL,
    LOCAL_API_KEY,
    LOCAL_TIMEOUT,
    VERTEX_PROJECT,
)
from core.google_llm import build_google_text_client, is_google_service


def build_primary_llm_client(
    *,
    translation_service: str,
    active_api_key: str | None,
    active_model_name: str | None,
    local_api_base_url: str,
    local_timeout: int,
    deepseek_api_base_url: str,
    model_name: Optional[str] = None,
) -> object:
    target_model_name = model_name or active_model_name
    if not target_model_name:
        raise RuntimeError("LLM 模型名称未配置")
    if translation_service != "vertex" and not active_api_key:
        raise RuntimeError("LLM API key 未配置")
    if translation_service == "grok":
        return OpenAI(api_key=active_api_key, base_url="https://api.x.ai/v1")
    if is_google_service(translation_service):
        return build_google_text_client(translation_service, target_model_name)
    if translation_service == "local":
        return OpenAI(base_url=local_api_base_url, api_key=active_api_key, timeout=local_timeout)
    if translation_service == "deepseek":
        return OpenAI(api_key=active_api_key, base_url=deepseek_api_base_url)
    raise RuntimeError(f"Unsupported translation service: {translation_service}")


def build_client_for_provider(service_name: str, model_name: str) -> object:
    if service_name == "grok":
        if not GROK_API_KEY:
            raise RuntimeError("Grok API key 未配置")
        return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
    if service_name == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini API key 未配置")
        return build_google_text_client("gemini", model_name)
    if service_name == "vertex":
        if not VERTEX_PROJECT:
            raise RuntimeError("Vertex project 未配置")
        return build_google_text_client("vertex", model_name)
    if service_name == "local":
        if not LOCAL_API_BASE_URL:
            raise RuntimeError("Local API base URL 未配置")
        return OpenAI(base_url=LOCAL_API_BASE_URL, api_key=LOCAL_API_KEY, timeout=LOCAL_TIMEOUT)
    if service_name == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DeepSeek API key 未配置")
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE_URL)
    raise RuntimeError(f"Unsupported translation service: {service_name}")


def resolve_client_for_provider(
    client_cache: dict[tuple[str, str], object],
    service_name: str,
    model_name: str,
) -> object:
    key = (service_name, model_name)
    if key not in client_cache:
        client_cache[key] = build_client_for_provider(service_name, model_name)
    return client_cache[key]


def determine_provider_plan(
    *,
    translation_service: str,
    active_model_name: str,
    service_display_name: str,
    gemini_config_ok: bool,
    gemini_retry_model_name: str = "gemini-3-flash-preview",
) -> tuple[list[tuple[str, str, str]], int]:
    if translation_service == "vertex" and gemini_config_ok:
        provider_plan = [
            ("vertex", active_model_name, service_display_name),
            ("gemini", gemini_retry_model_name, "Gemini API"),
        ]
        return provider_plan, 2
    provider_plan = [
        (translation_service, active_model_name, service_display_name),
    ]
    return provider_plan, 3


def sanitize_model_for_filename(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def short_stem_for_filename(stem: str, max_len: int = 48) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "batch"
    if len(sanitized) <= max_len:
        return sanitized
    return sanitized[:max_len].rstrip("._-") or "batch"
