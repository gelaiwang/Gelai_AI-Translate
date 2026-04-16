from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class FakeSubtitle:
    index: int
    start: timedelta
    end: timedelta
    content: str


def build_fake_srt_module() -> types.ModuleType:
    module = types.ModuleType("srt")
    module.Subtitle = FakeSubtitle

    def compose(subtitles: list[FakeSubtitle]) -> str:
        blocks: list[str] = []
        for subtitle in subtitles:
            blocks.append(
                f"{subtitle.index}\n"
                f"{subtitle.start} --> {subtitle.end}\n"
                f"{subtitle.content}"
            )
        return "\n\n".join(blocks)

    module.compose = compose
    return module


def build_fake_openai_module() -> types.ModuleType:
    module = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - only used for import compatibility
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    module.OpenAI = OpenAI
    return module


def build_fake_config_module() -> types.ModuleType:
    module = types.ModuleType("config")
    module.DEEPSEEK_API_BASE_URL = "https://example.invalid/v1"
    module.DEEPSEEK_API_KEY = ""
    module.GEMINI_API_KEY = ""
    module.GROK_API_KEY = ""
    module.LOCAL_API_BASE_URL = "http://localhost:11434/v1"
    module.LOCAL_API_KEY = "ollama"
    module.LOCAL_TIMEOUT = 150
    module.VERTEX_PROJECT = ""
    return module


def build_fake_google_llm_module() -> types.ModuleType:
    module = types.ModuleType("core.google_llm")

    def build_google_text_client(service: str, model_name: str) -> dict[str, str]:
        return {"service": service, "model_name": model_name}

    def is_google_service(service: str) -> bool:
        return service in {"gemini", "vertex"}

    module.build_google_text_client = build_google_text_client
    module.is_google_service = is_google_service
    return module


@contextmanager
def patched_modules(module_map: dict[str, types.ModuleType]) -> Iterator[None]:
    original: dict[str, types.ModuleType | None] = {}
    try:
        for name, module in module_map.items():
            original[name] = sys.modules.get(name)
            sys.modules[name] = module
        yield
    finally:
        for name, old in original.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def load_module(module_name: str, relative_path: str, module_map: dict[str, types.ModuleType]):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    with patched_modules(module_map):
        spec.loader.exec_module(module)
    return module
