from __future__ import annotations

from typing import Any

import yt_dlp

from config import (
    AUTH_METHOD,
    COOKIES_FILE,
    COOKIES_FROM_BROWSER,
    FALLBACK_CLIENTS,
    PLAYER_CLIENT,
    YOUTUBE_FETCH_POT,
    YOUTUBE_JSC_TRACE,
    YOUTUBE_PO_TOKENS,
    YOUTUBE_POT_BASE_URL,
    YOUTUBE_POT_DISABLE_INNERTUBE,
    YOUTUBE_POT_PROVIDER,
    YOUTUBE_POT_SCRIPT_PATH,
    YOUTUBE_POT_TRACE,
)


DEFAULT_FALLBACK_CLIENTS = ["tv", "web_safari", "mweb", "android", "default", "web"]
COOKIE_SAFE_CLIENTS = {"default", "web", "web_safari", "web_embedded", "web_music", "web_creator", "mweb", "tv", "tv_downgraded", "tv_simply"}
AUTHENTICATED_AUTH_METHODS = {"browser", "cookies_file", "oauth"}
COOKIE_UNSAFE_FALLBACK_CLIENTS = ["android"]


class YTDLPLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.debugs: list[str] = []

    def debug(self, msg: str) -> None:
        text = str(msg).strip()
        if text:
            self.debugs.append(text)

    def warning(self, msg: str) -> None:
        text = str(msg).strip()
        if text:
            self.warnings.append(text)

    def error(self, msg: str) -> None:
        text = str(msg).strip()
        if text:
            self.errors.append(text)

    def combined_messages(self) -> list[str]:
        return [*self.warnings, *self.errors]


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        val = str(item).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        result.append(val)
    return result


def normalize_probe_clients(cli_clients: str | None = None) -> list[str]:
    auth_uses_cookies = AUTH_METHOD in AUTHENTICATED_AUTH_METHODS
    if auth_uses_cookies:
        preferred = ["default"]
    else:
        preferred = dedupe_keep_order(PLAYER_CLIENT if isinstance(PLAYER_CLIENT, list) else [PLAYER_CLIENT])
    fallback_cfg = FALLBACK_CLIENTS if isinstance(FALLBACK_CLIENTS, list) else [FALLBACK_CLIENTS]
    fallback = dedupe_keep_order(fallback_cfg if fallback_cfg else DEFAULT_FALLBACK_CLIENTS)
    if cli_clients:
        fallback = dedupe_keep_order(cli_clients.split(","))
    merged = dedupe_keep_order([*preferred, *fallback])
    if auth_uses_cookies:
        merged = [client for client in merged if client in COOKIE_SAFE_CLIENTS]
        merged = dedupe_keep_order([*merged, *COOKIE_UNSAFE_FALLBACK_CLIENTS])
        return merged or ["default", "tv", "web_safari", "mweb", "web", "android"]
    return merged or DEFAULT_FALLBACK_CLIENTS


def apply_auth_options(ydl_opts: dict[str, Any], *, use_auth: bool = True) -> dict[str, Any]:
    if not use_auth:
        return ydl_opts
    if AUTH_METHOD == "browser":
        if COOKIES_FROM_BROWSER:
            ydl_opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
    elif AUTH_METHOD == "oauth":
        ydl_opts["username"] = "oauth"
        ydl_opts["password"] = ""
    elif AUTH_METHOD == "cookies_file":
        if COOKIES_FILE and COOKIES_FILE.exists():
            ydl_opts["cookiefile"] = str(COOKIES_FILE)
    return ydl_opts


def build_youtube_extractor_args(*clients: str) -> dict[str, dict[str, list[str]]]:
    normalized_clients = [client.strip() for client in clients if client and client.strip()]
    extractor_args: dict[str, dict[str, list[str]]] = {}
    youtube_args: dict[str, list[str]] = {}
    if normalized_clients:
        youtube_args["player_client"] = normalized_clients
    if YOUTUBE_PO_TOKENS:
        youtube_args["po_token"] = YOUTUBE_PO_TOKENS
    if YOUTUBE_FETCH_POT:
        youtube_args["fetch_pot"] = [YOUTUBE_FETCH_POT]
    if YOUTUBE_POT_TRACE:
        youtube_args["pot_trace"] = ["true"]
    if YOUTUBE_JSC_TRACE:
        youtube_args["jsc_trace"] = ["true"]
    extractor_args["youtube"] = youtube_args

    if YOUTUBE_POT_PROVIDER == "bgutil_http" and YOUTUBE_POT_BASE_URL:
        provider_args: dict[str, list[str]] = {"base_url": [YOUTUBE_POT_BASE_URL]}
        if YOUTUBE_POT_DISABLE_INNERTUBE:
            provider_args["disable_innertube"] = ["true"]
        extractor_args["youtubepot-bgutilhttp"] = provider_args
    elif YOUTUBE_POT_PROVIDER == "bgutil_script" and YOUTUBE_POT_SCRIPT_PATH:
        provider_args = {"script_path": [str(YOUTUBE_POT_SCRIPT_PATH)]}
        if YOUTUBE_POT_DISABLE_INNERTUBE:
            provider_args["disable_innertube"] = ["true"]
        extractor_args["youtubepot-bgutilscript"] = provider_args
    return extractor_args


def classify_download_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "the page needs to be reloaded" in msg:
        return "page_reload"
    if "only images are available for download" in msg:
        return "image_only_formats"
    if "timed out after" in msg and ("deno" in msg or "generate_once.ts" in msg):
        return "po_token_runtime"
    if "po token" in msg or "botguard" in msg or "challenge" in msg:
        return "po_token"
    if "sign in" in msg or "login" in msg or "cookie" in msg or "oauth" in msg:
        return "auth_required"
    if "requested format is not available" in msg or "no video formats found" in msg:
        return "format_unavailable"
    if "timed out" in msg or "http error" in msg or "network" in msg or "connection" in msg:
        return "network"
    return "other"


def should_retry_without_auth(error_type: str) -> bool:
    return AUTH_METHOD in AUTHENTICATED_AUTH_METHODS and error_type in {"page_reload", "auth_required"}


def fetch_video_metadata(video_url: str, clients: list[str] | None = None) -> dict[str, Any] | None:
    probe_clients = clients or normalize_probe_clients()
    for client in probe_clients:
        info = None
        auth_sequence = (True, False)
        if AUTH_METHOD in AUTHENTICATED_AUTH_METHODS and client in COOKIE_UNSAFE_FALLBACK_CLIENTS:
            auth_sequence = (False,)
        for use_auth in auth_sequence:
            logger = YTDLPLogger()
            probe_opts: dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
                "logger": logger,
                "extractor_args": build_youtube_extractor_args(client),
            }
            apply_auth_options(probe_opts, use_auth=use_auth)
            try:
                with yt_dlp.YoutubeDL(probe_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                break
            except Exception as exc:
                if use_auth and should_retry_without_auth(classify_download_error(exc)):
                    continue
                break
        if isinstance(info, dict):
            return info
    return None
