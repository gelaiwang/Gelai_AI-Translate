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


def build_fake_step1_config_module(workdir: Path) -> types.ModuleType:
    module = types.ModuleType("config")
    module.AUTH_METHOD = "none"
    module.COOKIES_FILE = None
    module.COOKIES_FROM_BROWSER = None
    module.DOWNLOAD_MAX_SLEEP_INTERVAL = 0.0
    module.DOWNLOAD_MIN_SLEEP_INTERVAL = 0.0
    module.DOWNLOAD_RATE_LIMIT = None
    module.DOWNLOAD_SLEEP_REQUESTS = 0.0
    module.DOWNLOAD_VIDEO = True
    module.FALLBACK_CLIENTS = ["tv", "web"]
    module.FALLBACK_FORMATS = ["best"]
    module.PLAYLIST_ITEMS = None
    module.PLAYER_CLIENT = ["mweb"]
    module.PRINT_FORMATS_ON_FAIL = False
    module.VIDEO_URL = ""
    module.YOUTUBE_FETCH_POT = "auto"
    module.YOUTUBE_BGUTIL_PROVIDER_ROOT = Path(".")
    module.YOUTUBE_JSC_TRACE = False
    module.YOUTUBE_PO_TOKENS = []
    module.YOUTUBE_POT_BASE_URL = "http://127.0.0.1:4416"
    module.YOUTUBE_POT_DISABLE_INNERTUBE = False
    module.YOUTUBE_POT_PROVIDER = "none"
    module.YOUTUBE_POT_TRACE = False
    module.Download_WORKDIR = workdir
    return module


def build_fake_llm_translate_config_module(*, workdir: Path, service: str = "gemini") -> types.ModuleType:
    module = types.ModuleType("config")
    module.WORKDIR = workdir
    module.TRANSLATION_SERVICE = service
    module.GROK_API_KEY = ""
    module.GROK_MODEL_NAME = "grok-4"
    module.GEMINI_API_KEY = "gemini-key"
    module.GEMINI_MODEL_NAME = "gemini-3-flash-preview"
    module.VERTEX_MODEL_NAME = "gemini-3-flash-preview"
    module.VERTEX_PROJECT = ""
    module.LOCAL_API_BASE_URL = "http://localhost:11434/v1"
    module.LOCAL_API_KEY = "ollama"
    module.LOCAL_MODEL_NAME = "qwen3:30b"
    module.LOCAL_TIMEOUT = 150
    module.LOCAL_CONTEXT_LENGTH = 8192
    module.LOCAL_SHOW_THINKING = False
    module.LOCAL_TOP_P = 0.5
    module.LOCAL_ENABLE_TWO_STAGE = True
    module.LOCAL_WINDOW_SIZE = 40
    module.LOCAL_OVERLAP = 10
    module.LOCAL_STAGE1_TEMPERATURE = 0.2
    module.LOCAL_STAGE2_TEMPERATURE = 0.7
    module.DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"
    module.DEEPSEEK_API_KEY = ""
    module.DEEPSEEK_MODEL_NAME = "deepseek-chat"
    module.LLM_TEMPERATURE_TRANSLATE_CLOUD = 0.5
    module.LLM_TEMPERATURE_TRANSLATE_LOCAL = 0.5
    module.PROMPT_TEMPLATE_FILE = workdir / "translate.txt"
    module.TRANSLATION_CONTEXT_TEMPLATE_FILE = workdir / "translation_context.txt"
    module.TRANSLATION_CONTEXT_ENABLED = False
    module.TRANSLATION_CONTEXT_FORCE_REGENERATE = False
    module.TRANSLATION_CONTEXT_FILE_NAME = "translation_context.txt"
    module.TRANSLATION_CONTEXT_SOURCE_MAX_CHARS = 12000
    return module


def build_fake_google_llm_module() -> types.ModuleType:
    module = types.ModuleType("core.google_llm")

    def build_google_text_client(service: str, model_name: str) -> dict[str, str]:
        return {"service": service, "model_name": model_name}

    def generate_google_text(client, prompt: str, temperature: float) -> str:
        return f"{client['service']}:{temperature}:{prompt}"

    def is_google_service(service: str) -> bool:
        return service in {"gemini", "vertex"}

    module.build_google_text_client = build_google_text_client
    module.generate_google_text = generate_google_text
    module.is_google_service = is_google_service
    return module


def build_fake_rich_module() -> types.ModuleType:
    module = types.ModuleType("rich")
    module.print = lambda *args, **kwargs: None
    return module


def build_fake_rich_panel_module() -> types.ModuleType:
    module = types.ModuleType("rich.panel")

    class Panel:
        def __init__(self, renderable, **kwargs) -> None:
            self.renderable = renderable
            self.kwargs = kwargs

    module.Panel = Panel
    return module


def build_fake_rich_console_module() -> types.ModuleType:
    module = types.ModuleType("rich.console")

    class Console:
        def print(self, *args, **kwargs) -> None:
            return None

    module.Console = Console
    return module


def build_fake_rich_progress_module() -> types.ModuleType:
    module = types.ModuleType("rich.progress")

    class Progress:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def add_task(self, *args, **kwargs) -> int:
            return 1

        def update(self, *args, **kwargs) -> None:
            return None

    class _Column:
        def __init__(self, *args, **kwargs) -> None:
            pass

    module.Progress = Progress
    module.BarColumn = _Column
    module.SpinnerColumn = _Column
    module.TextColumn = _Column
    module.TimeElapsedColumn = _Column
    return module


def build_fake_yt_dlp_module() -> types.ModuleType:
    module = types.ModuleType("yt_dlp")

    class YoutubeDL:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    module.YoutubeDL = YoutubeDL
    return module


def build_fake_youtube_metadata_module() -> types.ModuleType:
    module = types.ModuleType("core.youtube_metadata")
    module.fetch_video_metadata = lambda *args, **kwargs: {}
    return module


def build_fake_translation_context_module() -> types.ModuleType:
    module = types.ModuleType("core.translation_context")
    module.detect_series_marker = lambda title: (False, "", None)
    module.ensure_series_registry_entry = lambda **kwargs: ("", "")
    module.resolve_translation_context_text = lambda **kwargs: "context"
    return module


def build_fake_translation_text_module() -> types.ModuleType:
    module = types.ModuleType("core.translation_text")
    module.DEFAULT_CONTEXT_BLOCK = "default context"
    module.augment_prompt_with_hints = lambda base, hints: base
    module.chunk_subtitles_by_line_limit = lambda subtitles, max_lines_per_batch: [subtitles]
    module.clean_llm_output = lambda text: text.strip()
    module.create_sliding_window_chunks = lambda lines, window_size, overlap: []
    module.escape_braces = lambda text: text
    module.extract_lines_from_output = lambda text: [line for line in text.splitlines() if line]
    module.merge_overlapping_translations = lambda chunk_results, total_lines: {}
    module.prepare_lines_for_batch = lambda subtitles: ([], [])
    module.preprocess_lines_with_numbers = lambda lines: [f"{i+1}: {line}" for i, line in enumerate(lines)]
    module.reconstruct_subtitles_from_lines = lambda original_batch, translated_text, line_counts: original_batch
    module.record_batch_error_hint = lambda storage, batch_index, error: None
    module.subtitle_to_lines = lambda subtitle: [subtitle.content]
    module.validate_line_count = lambda lines, expected_count: (len(lines) == expected_count, "")
    module.write_plain_txt = lambda subtitles, txt_path: txt_path.write_text("", encoding="utf-8")
    return module


def build_fake_translation_runtime_module() -> types.ModuleType:
    module = types.ModuleType("core.translation_runtime")
    module.build_primary_llm_client = lambda **kwargs: {"client": "ok", "kwargs": kwargs}
    module.determine_provider_plan = lambda **kwargs: ([("gemini", "gemini-3-flash-preview", "Gemini")], 3)
    module.resolve_client_for_provider = lambda client_cache, service_name, model_name: client_cache.get(
        (service_name, model_name),
        {"service": service_name, "model_name": model_name},
    )
    module.sanitize_model_for_filename = lambda model_name: model_name.replace("/", "_")
    module.short_stem_for_filename = lambda stem, max_len=48: stem[:max_len]
    return module


def build_fake_translation_batches_module() -> types.ModuleType:
    module = types.ModuleType("core.translation_batches")
    module.execute_batch_translation = lambda **kwargs: ([], [], [])
    return module


def build_fake_translation_prompts_module() -> types.ModuleType:
    module = types.ModuleType("core.translation_prompts")
    module.load_local_stage_prompts = lambda prompts_dir: ("stage1 {srt_content_for_llm}", "stage2 {srt_content_for_llm}")
    module.load_prompt_template = lambda path, required_placeholders=None: "\n".join(required_placeholders or [])
    return module


def build_fake_translation_validation_module() -> types.ModuleType:
    module = types.ModuleType("core.translation_validation")
    module.validate_llm_translation = lambda **kwargs: True
    return module


def _ensure_package_modules(module_map: dict[str, types.ModuleType]) -> list[str]:
    created: list[str] = []
    for name in list(module_map):
        parts = name.split(".")
        for idx in range(1, len(parts)):
            package_name = ".".join(parts[:idx])
            if package_name not in module_map and package_name not in sys.modules:
                package_module = types.ModuleType(package_name)
                package_module.__path__ = []  # type: ignore[attr-defined]
                sys.modules[package_name] = package_module
                created.append(package_name)
    return created


@contextmanager
def patched_modules(module_map: dict[str, types.ModuleType]) -> Iterator[None]:
    original: dict[str, types.ModuleType | None] = {}
    created_packages: list[str] = []
    try:
        created_packages = _ensure_package_modules(module_map)
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
        for name in created_packages:
            sys.modules.pop(name, None)


def load_module(module_name: str, relative_path: str, module_map: dict[str, types.ModuleType]):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    with patched_modules(module_map):
        spec.loader.exec_module(module)
    return module
