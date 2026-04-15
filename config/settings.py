from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = BASE_DIR / "config.yaml"
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def _load_yaml_config() -> dict[str, Any]:
    if not CONFIG_YAML_PATH.exists():
        return {}
    with CONFIG_YAML_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


_yaml_cfg = _load_yaml_config()


def _cfg(*keys: str, default: Any = None) -> Any:
    current: Any = _yaml_cfg
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


WORKDIR = Path(_cfg("workdir", default=(BASE_DIR / "workdir"))).expanduser()
Download_WORKDIR = Path(_cfg("download_workdir", default=WORKDIR)).expanduser()

PROMPTS_DIR = BASE_DIR / "config" / "prompts"
PROMPT_TEMPLATE_FILE = PROMPTS_DIR / "translate.txt"
TRANSLATION_CONTEXT_TEMPLATE_FILE = PROMPTS_DIR / "translation_context.txt"

AUTH_METHOD = str(_cfg("video", "auth_method", default="none")).strip().lower()
COOKIES_FROM_BROWSER = _cfg("video", "cookies_from_browser", default=None)
_cookies_file = _cfg("video", "cookies_file", default=None)
COOKIES_FILE = (BASE_DIR / _cookies_file).resolve() if _cookies_file and not Path(_cookies_file).is_absolute() else (Path(_cookies_file).expanduser() if _cookies_file else None)
VIDEO_URL = str(_cfg("video", "url", default="")).strip()
DOWNLOAD_VIDEO = bool(_cfg("video", "download_video", default=True))
PLAYLIST_ITEMS = _cfg("video", "playlist_items", default=None)
PLAYER_CLIENT = _cfg("video", "player_client", default=["mweb", "web_safari", "tv"]) or ["mweb", "web_safari", "tv"]
FALLBACK_CLIENTS = _cfg("video", "fallback_clients", default=["tv", "web_safari", "mweb", "android", "default", "web"]) or ["tv", "web_safari", "mweb", "android", "default", "web"]
FALLBACK_FORMATS = _cfg("video", "fallback_formats", default=[
    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best",
    "bestvideo+bestaudio/best",
    "best",
]) or [
    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best",
    "bestvideo+bestaudio/best",
    "best",
]
PRINT_FORMATS_ON_FAIL = bool(_cfg("video", "print_formats_on_fail", default=True))

DOWNLOAD_MIN_SLEEP_INTERVAL = float(_cfg("video", "rate_limit", "min_sleep_interval", default=0) or 0)
DOWNLOAD_MAX_SLEEP_INTERVAL = float(_cfg("video", "rate_limit", "max_sleep_interval", default=0) or 0)
DOWNLOAD_SLEEP_REQUESTS = float(_cfg("video", "rate_limit", "sleep_requests", default=0) or 0)
DOWNLOAD_RATE_LIMIT = _cfg("video", "rate_limit", "download_rate", default=None)

YOUTUBE_PO_TOKENS = _cfg("video", "po_token", default=[]) or []
YOUTUBE_FETCH_POT = _cfg("video", "fetch_pot", default="auto")
YOUTUBE_POT_TRACE = bool(_cfg("video", "pot_trace", default=False))
YOUTUBE_JSC_TRACE = bool(_cfg("video", "jsc_trace", default=False))
YOUTUBE_POT_PROVIDER = str(_cfg("video", "pot_provider", default="none")).strip().lower()
YOUTUBE_POT_BASE_URL = _cfg("video", "pot_base_url", default="http://127.0.0.1:4416")
_pot_script_path = _cfg("video", "pot_script_path", default="")
YOUTUBE_POT_SCRIPT_PATH = Path(_pot_script_path).expanduser() if _pot_script_path else None
_bgutil_provider_root = _cfg(
    "video",
    "bgutil_provider_root",
    default=os.getenv("BGUTIL_PROVIDER_ROOT", ""),
)
YOUTUBE_BGUTIL_PROVIDER_ROOT = Path(_bgutil_provider_root).expanduser()
YOUTUBE_POT_DISABLE_INNERTUBE = bool(_cfg("video", "pot_disable_innertube", default=False))

ASR_LANGUAGE = str(_cfg("asr", "language", default="en"))
ASR_ALIGNMENT_LANGUAGE = str(_cfg("asr", "alignment_language", default=ASR_LANGUAGE))
ASR_WHISPER_MODEL = str(_cfg("asr", "model_name", default="medium"))
ASR_DEVICE = str(_cfg("asr", "device", default="auto")).strip().lower()
ASR_COMPUTE_TYPE = str(_cfg("asr", "compute_type", default="auto")).strip().lower()
ASR_BATCH_SIZE = _cfg("asr", "batch_size", default="auto")
ASR_USE_VOCAL_SEPARATION = bool(_cfg("asr", "use_vocal_separation", default=False))
ASR_SNR_THRESHOLD = float(_cfg("asr", "snr_threshold", default=20.0) or 20.0)
ASR_SPEAKER_DIARIZATION = bool(_cfg("asr", "speaker_diarization", default=False))
ASR_MIN_SPEAKERS = _cfg("asr", "min_speakers", default=None)
ASR_MAX_SPEAKERS = _cfg("asr", "max_speakers", default=None)
ASR_HF_TOKEN_ENV = str(_cfg("asr", "hf_token_env", default="HF_TOKEN")).strip() or "HF_TOKEN"
ASR_HF_TOKEN = os.getenv(ASR_HF_TOKEN_ENV, "").strip()

TRANSLATION_SERVICE = str(_cfg("services", "translation", default="gemini")).strip().lower()
SEGMENTATION_SERVICE = str(_cfg("services", "segmentation", default="rule")).strip().lower()

PAUSE_THRESHOLD = float(_cfg("segmentation", "pause_threshold", default=0.4) or 0.4)
MIN_SEGMENT_LENGTH = int(_cfg("segmentation", "min_segment_length", default=15) or 15)
MAX_SEGMENT_LENGTH = int(_cfg("segmentation", "max_segment_length", default=65) or 65)

LLM_TEMPERATURE_TRANSLATE_CLOUD = float(_cfg("temperatures", "translate_cloud", default=0.5) or 0.5)
LLM_TEMPERATURE_TRANSLATE_LOCAL = float(_cfg("temperatures", "translate_local", default=0.5) or 0.5)
LLM_TEMPERATURE_SEGMENTATION = float(_cfg("temperatures", "segmentation", default=0.1) or 0.1)

TRANSLATION_CONTEXT_ENABLED = bool(_cfg("translation_context", "enabled", default=False))
TRANSLATION_CONTEXT_FORCE_REGENERATE = bool(_cfg("translation_context", "force_regenerate", default=False))
TRANSLATION_CONTEXT_FILE_NAME = str(
    _cfg("translation_context", "file_name", default="translation_context.txt")
).strip() or "translation_context.txt"
TRANSLATION_CONTEXT_SOURCE_MAX_CHARS = int(
    _cfg("translation_context", "source_max_chars", default=12000) or 12000
)

CLOUD_LINE_BATCH_SIZE = int(_cfg("video", "cloud_line_batch_size", default=50) or 50)
LOCAL_LINE_BATCH_SIZE = int(_cfg("video", "local_line_batch_size", default=20) or 20)

GROK_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL_NAME = str(_cfg("models", "grok", "name", default="grok-4"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = str(_cfg("models", "gemini", "translate", default="gemini-3-flash-preview"))
GEMINI_MODEL_SEGMENTATION = str(_cfg("models", "gemini", "segmentation", default=GEMINI_MODEL_NAME))

VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("VERTEX_PROJECT_ID", "")).strip()
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", os.getenv("VERTEX_LOCATION", "global")).strip() or "global"
VERTEX_API_VERSION = str(os.getenv("VERTEX_API_VERSION", "v1"))
VERTEX_MODEL_NAME = str(_cfg("models", "vertex", "translate", default="gemini-3-flash-preview"))
VERTEX_MODEL_SEGMENTATION = str(_cfg("models", "vertex", "segmentation", default=VERTEX_MODEL_NAME))

LOCAL_API_BASE_URL = str(_cfg("models", "local", "api_base_url", default="http://localhost:11434/v1"))
LOCAL_API_KEY = str(_cfg("models", "local", "api_key", default="ollama"))
LOCAL_MODEL_NAME = str(_cfg("models", "local", "translate", default="qwen3:30b"))
LOCAL_MODEL_SEGMENTATION = str(_cfg("models", "local", "segmentation", default=LOCAL_MODEL_NAME))
LOCAL_TIMEOUT = int(_cfg("models", "local", "timeout", default=150) or 150)
LOCAL_CONTEXT_LENGTH = int(_cfg("models", "local", "context_length", default=8192) or 8192)
LOCAL_SHOW_THINKING = bool(_cfg("models", "local", "show_thinking", default=False))
LOCAL_TOP_P = float(_cfg("models", "local", "top_p", default=0.5) or 0.5)
LOCAL_ENABLE_TWO_STAGE = bool(_cfg("models", "local", "enable_two_stage", default=True))
LOCAL_WINDOW_SIZE = int(_cfg("models", "local", "window_size", default=40) or 40)
LOCAL_OVERLAP = int(_cfg("models", "local", "overlap", default=10) or 10)
LOCAL_STAGE1_TEMPERATURE = float(_cfg("models", "local", "stage1_temperature", default=0.2) or 0.2)
LOCAL_STAGE2_TEMPERATURE = float(_cfg("models", "local", "stage2_temperature", default=0.7) or 0.7)

DEEPSEEK_API_BASE_URL = str(_cfg("models", "deepseek", "api_base_url", default="https://api.deepseek.com/v1"))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL_NAME = str(_cfg("models", "deepseek", "translate", default="deepseek-chat"))
DEEPSEEK_MODEL_SEGMENTATION = str(_cfg("models", "deepseek", "segmentation", default=DEEPSEEK_MODEL_NAME))

_font_path = str(_cfg("render", "font_path", default="")).strip()
FONT_PATH = Path(_font_path).expanduser() if _font_path else None
_platform_system = platform.system().lower()
_default_font_family_en = "Arial"
_default_font_family_cn = "Noto Sans CJK SC"
if _platform_system == "darwin":
    _default_font_family_en = "Helvetica"
    _default_font_family_cn = "PingFang SC"
elif _platform_system == "windows":
    _default_font_family_en = "Arial"
    _default_font_family_cn = "Microsoft YaHei"
elif _platform_system == "linux":
    _default_font_family_en = "DejaVu Sans"
    _default_font_family_cn = "Noto Sans CJK SC"

FONT_FAMILY_EN = str(_cfg("render", "font_family_en", default=_default_font_family_en))
FONT_FAMILY_CN = str(_cfg("render", "font_family_cn", default=_default_font_family_cn))
SUBTITLE_FONT_SIZE_EN = int(_cfg("render", "subtitle_font_size_en", default=13) or 13)
SUBTITLE_FONT_SIZE_CN = int(_cfg("render", "subtitle_font_size_cn", default=22) or 22)
RENDER_FFMPEG_BIN = str(_cfg("render", "ffmpeg_bin", default="ffmpeg")).strip() or "ffmpeg"
RENDER_FFPROBE_BIN = str(_cfg("render", "ffprobe_bin", default="ffprobe")).strip() or "ffprobe"
RENDER_VIDEO_CODEC = str(_cfg("render", "video_codec", default="auto")).strip().lower() or "auto"
RENDER_VIDEO_PRESET = str(_cfg("render", "video_preset", default="slow")).strip() or "slow"
RENDER_VIDEO_CRF = str(_cfg("render", "video_crf", default="20")).strip() or "20"
RENDER_OUTPUT_SUFFIX = str(_cfg("render", "output_suffix", default="Done_")).strip() or "Done_"
