from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yt_dlp
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from core.youtube_metadata import fetch_video_metadata
from config import (
    AUTH_METHOD,
    COOKIES_FILE,
    COOKIES_FROM_BROWSER,
    DOWNLOAD_MAX_SLEEP_INTERVAL,
    DOWNLOAD_MIN_SLEEP_INTERVAL,
    DOWNLOAD_RATE_LIMIT,
    DOWNLOAD_SLEEP_REQUESTS,
    DOWNLOAD_VIDEO,
    FALLBACK_CLIENTS,
    FALLBACK_FORMATS,
    PLAYLIST_ITEMS,
    PLAYER_CLIENT,
    PRINT_FORMATS_ON_FAIL,
    VIDEO_URL,
    YOUTUBE_FETCH_POT,
    YOUTUBE_BGUTIL_PROVIDER_ROOT,
    YOUTUBE_JSC_TRACE,
    YOUTUBE_PO_TOKENS,
    YOUTUBE_POT_BASE_URL,
    YOUTUBE_POT_DISABLE_INNERTUBE,
    YOUTUBE_POT_PROVIDER,
    YOUTUBE_POT_TRACE,
    Download_WORKDIR as WORKDIR,
)

console = Console()

DOWNLOAD_RECORD_FILE = WORKDIR / "downloaded_ids.json"
MIN_VIDEO_SIZE_BYTES = 1 * 1024 * 1024
MIN_VIDEO_HEIGHT = 720

DEFAULT_FORMAT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best"
DEFAULT_FALLBACK_FORMATS = [DEFAULT_FORMAT, "bestvideo+bestaudio/best", "best"]
DEFAULT_FALLBACK_CLIENTS = ["tv", "web_safari", "mweb", "android", "default", "web"]
COOKIE_SAFE_CLIENTS = {"default", "web", "web_safari", "web_embedded", "web_music", "web_creator", "mweb", "tv", "tv_downgraded", "tv_simply"}
AUTHENTICATED_AUTH_METHODS = {"browser", "cookies_file"}
COOKIE_UNSAFE_FALLBACK_CLIENTS = ["android"]
BGUTIL_HTTP_SENTINEL_HOME = WORKDIR / ".bgutil_http_only"


class _YTDLPLogger:
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


def time_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_port_open(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _wait_for_port(host: str, port: int, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def _provider_host_port() -> tuple[str, int]:
    parsed = urlparse(YOUTUBE_POT_BASE_URL)
    return parsed.hostname or "127.0.0.1", parsed.port or 4416


def ensure_bgutil_provider() -> None:
    if YOUTUBE_POT_PROVIDER != "bgutil_http":
        return

    host, port = _provider_host_port()
    if _is_port_open(host, port):
        console.print(f"[cyan]bgutil provider already listening on {host}:{port}[/cyan]")
        return

    if not str(YOUTUBE_BGUTIL_PROVIDER_ROOT).strip() or str(YOUTUBE_BGUTIL_PROVIDER_ROOT) == ".":
        raise RuntimeError(
            "video.bgutil_provider_root is required when pot_provider=bgutil_http. "
            "Set it in config.yaml or provide BGUTIL_PROVIDER_ROOT in the environment."
        )

    provider_server_dir = YOUTUBE_BGUTIL_PROVIDER_ROOT / "server"
    provider_entry = provider_server_dir / "build" / "main.js"
    if not provider_entry.exists():
        raise RuntimeError(
            "bgutil provider not found. "
            f"Expected entry: {provider_entry}. "
            "Set video.bgutil_provider_root in config.yaml or BGUTIL_PROVIDER_ROOT in the environment."
        )

    if not shutil.which("node"):
        raise RuntimeError("node is required for pot_provider=bgutil_http but was not found on PATH.")

    console.print(f"[cyan]Starting bgutil provider on {host}:{port}[/cyan]")
    subprocess.Popen(
        ["node", str(provider_entry), "--port", str(port)],
        cwd=str(provider_server_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not _wait_for_port(host, port):
        raise RuntimeError(f"bgutil provider failed to start on {host}:{port}")
    console.print(f"[green]bgutil provider is ready on {host}:{port}[/green]")


def apply_auth_options(ydl_opts: dict[str, Any], *, use_auth: bool = True) -> dict[str, Any]:
    if not use_auth:
        return ydl_opts
    if AUTH_METHOD == "browser":
        if COOKIES_FROM_BROWSER:
            ydl_opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
    elif AUTH_METHOD == "cookies_file":
        if COOKIES_FILE and COOKIES_FILE.exists():
            ydl_opts["cookiefile"] = str(COOKIES_FILE)
    return ydl_opts


def apply_rate_limit_options(ydl_opts: dict[str, Any]) -> dict[str, Any]:
    if DOWNLOAD_MIN_SLEEP_INTERVAL > 0:
        ydl_opts["sleep_interval"] = DOWNLOAD_MIN_SLEEP_INTERVAL
        if DOWNLOAD_MAX_SLEEP_INTERVAL > DOWNLOAD_MIN_SLEEP_INTERVAL:
            ydl_opts["max_sleep_interval"] = DOWNLOAD_MAX_SLEEP_INTERVAL

    if DOWNLOAD_SLEEP_REQUESTS > 0:
        ydl_opts["sleep_requests"] = DOWNLOAD_SLEEP_REQUESTS

    if DOWNLOAD_RATE_LIMIT:
        rate_str = str(DOWNLOAD_RATE_LIMIT).strip().upper()
        rate_bytes = None
        try:
            if rate_str.endswith("K"):
                rate_bytes = int(float(rate_str[:-1]) * 1024)
            elif rate_str.endswith("M"):
                rate_bytes = int(float(rate_str[:-1]) * 1024 * 1024)
            elif rate_str.endswith("G"):
                rate_bytes = int(float(rate_str[:-1]) * 1024 * 1024 * 1024)
            else:
                rate_bytes = int(rate_str)
        except ValueError:
            rate_bytes = None
        if rate_bytes:
            ydl_opts["ratelimit"] = rate_bytes

    return ydl_opts


def _build_youtube_extractor_args(*clients: str) -> dict[str, dict[str, list[str]]]:
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
        # Force script-provider checks to fail fast while using the HTTP provider.
        extractor_args["youtubepot-bgutilscript"] = {"server_home": [str(BGUTIL_HTTP_SENTINEL_HOME)]}

    return extractor_args


def load_downloaded_ids() -> set[str]:
    if DOWNLOAD_RECORD_FILE.exists():
        try:
            with DOWNLOAD_RECORD_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("downloaded_ids", []))
        except Exception:
            return set()
    return set()


def save_downloaded_ids(ids: set[str]) -> None:
    DOWNLOAD_RECORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DOWNLOAD_RECORD_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {"downloaded_ids": sorted(ids), "last_updated": datetime.now().isoformat()},
            f,
            ensure_ascii=False,
            indent=2,
        )


def add_downloaded_id(video_id: str, existing_ids: set[str]) -> set[str]:
    existing_ids.add(video_id)
    save_downloaded_ids(existing_ids)
    return existing_ids


def _probe_video_resolution(file_path: Path) -> tuple[int | None, int | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(file_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None

    if result.returncode != 0:
        return None, None

    raw = (result.stdout or "").strip()
    if not raw or "x" not in raw:
        return None, None

    width_str, height_str = raw.split("x", 1)
    try:
        return int(width_str), int(height_str)
    except ValueError:
        return None, None


def _matching_video_files(workdir: Path, video_id: str, root_only: bool = False) -> list[Path]:
    matches = list(workdir.glob(f"*{video_id}*.mp4")) if root_only else list(workdir.rglob(f"*{video_id}*.mp4"))

    def sort_key(path: Path) -> tuple[int, int, float]:
        width, height = _probe_video_resolution(path)
        stat = path.stat()
        return (height or 0, stat.st_size, stat.st_mtime)

    return sorted(matches, key=sort_key, reverse=True)


def _remove_existing_video_artifacts(workdir: Path, video_id: str) -> int:
    removed = 0
    patterns = [
        f"*{video_id}*.mp4",
        f"*{video_id}*.info.json",
        f"*{video_id}*.webp",
        f"*{video_id}*.jpg",
        f"*{video_id}*.png",
        f"*{video_id}*.m4a",
        f"*{video_id}*.part*",
    ]
    for pattern in patterns:
        for path in workdir.glob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def _extract_max_available_height(info: dict[str, Any] | None) -> int | None:
    if not isinstance(info, dict):
        return None
    formats = info.get("formats") or []
    heights: list[int] = []
    for fmt in formats:
        if not isinstance(fmt, dict):
            continue
        height = fmt.get("height")
        try:
            if height:
                heights.append(int(height))
        except (TypeError, ValueError):
            continue
    return max(heights) if heights else None


def _probe_source_max_height(video_url: str, clients: list[str]) -> int | None:
    max_height: int | None = None
    for client in clients:
        info = fetch_video_metadata(video_url, clients=[client])
        if info is None:
            continue
        client_max_height = _extract_max_available_height(info)
        if client_max_height is not None:
            max_height = client_max_height if max_height is None else max(max_height, client_max_height)
    return max_height


def verify_downloaded_video(
    workdir: Path,
    video_id: str,
    root_only: bool = False,
    required_min_height: int | None = MIN_VIDEO_HEIGHT,
) -> tuple[bool, str]:
    if root_only:
        part_files = list(workdir.glob(f"*{video_id}*.part*"))
        if part_files:
            return False, f"incomplete download exists: {part_files[0].name}"
    matches = _matching_video_files(workdir, video_id, root_only=root_only)

    if not matches:
        return False, "no mp4 found"
    best_failure = "no valid mp4 found"
    for file_path in matches:
        file_size = file_path.stat().st_size
        if file_size < MIN_VIDEO_SIZE_BYTES:
            best_failure = f"file too small: {file_path.name} ({file_size} bytes)"
            continue

        width, height = _probe_video_resolution(file_path)
        if width is None or height is None:
            best_failure = f"unable to probe video resolution via ffprobe: {file_path.name}"
            continue
        if required_min_height is not None and height < required_min_height:
            best_failure = (
                f"resolution too low: {file_path.name} ({width}x{height} < {required_min_height}p)"
            )
            continue

        return True, f"{file_path.name} ({file_size / (1024 * 1024):.2f} MB, {width}x{height})"

    return False, best_failure


def init_download_record_from_info_json() -> set[str]:
    existing_ids = load_downloaded_ids()
    changed = False

    for part_file in WORKDIR.glob("*.part*"):
        base = re.sub(r"(\.f\d+)?\.(mp4|m4a)\.part.*$", "", part_file.name)
        if "_" not in base:
            continue
        video_id = base.rsplit("_", 1)[-1]
        if video_id in existing_ids:
            existing_ids.discard(video_id)
            changed = True

    for info_file in WORKDIR.rglob("*.info.json"):
        try:
            data = json.loads(info_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
        except Exception:
            continue
        video_id = data.get("id")
        if not video_id:
            continue
        if info_file.parent == WORKDIR:
            source_max_height = _extract_max_available_height(data)
            required_min_height = MIN_VIDEO_HEIGHT if source_max_height and source_max_height >= MIN_VIDEO_HEIGHT else None
            valid, _ = verify_downloaded_video(
                WORKDIR,
                video_id,
                root_only=True,
                required_min_height=required_min_height,
            )
            if valid:
                if video_id not in existing_ids:
                    existing_ids.add(video_id)
                    changed = True
            else:
                if video_id in existing_ids:
                    existing_ids.discard(video_id)
                    changed = True
        else:
            if video_id not in existing_ids:
                existing_ids.add(video_id)
                changed = True

    if changed:
        save_downloaded_ids(existing_ids)
    return existing_ids


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        val = str(item).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        result.append(val)
    return result


def _normalize_clients(cli_clients: str | None) -> list[str]:
    auth_uses_cookies = AUTH_METHOD in AUTHENTICATED_AUTH_METHODS
    if auth_uses_cookies:
        preferred = ["default"]
    else:
        preferred = _dedupe_keep_order(PLAYER_CLIENT if isinstance(PLAYER_CLIENT, list) else [PLAYER_CLIENT])
    fallback_cfg = FALLBACK_CLIENTS if isinstance(FALLBACK_CLIENTS, list) else [FALLBACK_CLIENTS]
    fallback = _dedupe_keep_order(fallback_cfg if fallback_cfg else DEFAULT_FALLBACK_CLIENTS)
    if cli_clients:
        fallback = _dedupe_keep_order(cli_clients.split(","))
    merged = _dedupe_keep_order([*preferred, *fallback])
    if auth_uses_cookies:
        merged = [client for client in merged if client in COOKIE_SAFE_CLIENTS]
        merged = _dedupe_keep_order([*merged, *COOKIE_UNSAFE_FALLBACK_CLIENTS])
        return merged or ["default", "tv", "web_safari", "mweb", "web", "android"]
    return merged or DEFAULT_FALLBACK_CLIENTS


def _normalize_formats(primary_fmt: str, cli_formats: str | None) -> list[str]:
    fallback_cfg = FALLBACK_FORMATS if isinstance(FALLBACK_FORMATS, list) else [FALLBACK_FORMATS]
    fallback = _dedupe_keep_order(fallback_cfg if fallback_cfg else DEFAULT_FALLBACK_FORMATS)
    if cli_formats:
        fallback = _dedupe_keep_order(cli_formats.split(","))
    merged = _dedupe_keep_order([primary_fmt, *fallback])
    return merged or DEFAULT_FALLBACK_FORMATS


def _classify_download_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "the page needs to be reloaded" in msg:
        return "page_reload"
    if "only images are available for download" in msg:
        return "image_only_formats"
    if "timed out after" in msg and ("deno" in msg or "generate_once.ts" in msg):
        return "po_token_runtime"
    if "po token" in msg or "botguard" in msg or "challenge" in msg:
        return "po_token"
    if "sign in" in msg or "login" in msg or "cookie" in msg:
        return "auth_required"
    if "requested format is not available" in msg or "no video formats found" in msg:
        return "format_unavailable"
    if "timed out" in msg or "http error" in msg or "network" in msg or "connection" in msg:
        return "network"
    return "other"


def _should_continue_format_fallback(error_type: str) -> bool:
    return error_type in {"format_unavailable"}


def _should_retry_without_auth(error_type: str) -> bool:
    return AUTH_METHOD in AUTHENTICATED_AUTH_METHODS and error_type in {"page_reload", "auth_required"}


def _print_formats_summary(video_url: str, clients: list[str]) -> None:
    console.print("[cyan]可用格式摘要:[/cyan]")
    for client in clients:
        try:
            info = fetch_video_metadata(video_url, clients=[client])
            if not info:
                raise RuntimeError("probe returned no metadata")
            formats = (info or {}).get("formats") or []
            console.print(f"  [bold]{client}[/bold]: {len(formats)} formats")
            for f in formats[:12]:
                fid = f.get("format_id", "N/A")
                ext = f.get("ext", "N/A")
                resolution = f.get("resolution") or f"{f.get('height') or '?'}p"
                vcodec = f.get("vcodec", "N/A")
                acodec = f.get("acodec", "N/A")
                console.print(f"    - id={fid} ext={ext} res={resolution} v={vcodec} a={acodec}")
        except Exception as exc:
            console.print(f"  [yellow]{client}: probe failed: {exc}[/yellow]")


def _build_single_video_opts(
    *,
    fmt: str,
    client: str,
    download_video: bool,
    use_auth: bool = True,
    progress_hook=None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "outtmpl": str(WORKDIR / "%(title)s_%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "writethumbnail": True,
        "writeinfojson": True,
        "extractor_args": _build_youtube_extractor_args(client),
    }
    if progress_hook:
        opts["progress_hooks"] = [progress_hook]

    if download_video:
        opts["format"] = fmt
        opts["merge_output_format"] = "mp4"
        apply_rate_limit_options(opts)
    else:
        opts["skip_download"] = True
        opts["format"] = "best"
        opts["ignore_no_formats_error"] = True

    apply_auth_options(opts, use_auth=use_auth)
    return opts


def _attempt_download(
    *,
    video_url: str,
    video_title: str,
    video_id: str,
    client: str,
    fmt: str,
    download_video: bool,
    required_min_height: int | None = MIN_VIDEO_HEIGHT,
) -> dict[str, Any]:
    progress_state = [None]
    used_auth_fallback = False
    if download_video:
        _remove_existing_video_artifacts(WORKDIR, video_id)

    auth_sequence = (True, False)
    if AUTH_METHOD in AUTHENTICATED_AUTH_METHODS and client in COOKIE_UNSAFE_FALLBACK_CLIENTS:
        auth_sequence = (False,)

    def hook(d):
        if d.get("status") == "downloading" and progress_state[0] is not None:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes")
            if total and done:
                progress.update(
                    progress_state[0],
                    total=total,
                    completed=done,
                    description=f"下载中: {video_title} {d.get('_percent_str', '')} ({d.get('_speed_str', '')})",
                )
            else:
                progress.update(progress_state[0], description=f"下载中: {video_title}")
        elif d.get("status") == "finished" and progress_state[0] is not None:
            total = d.get("total_bytes") or 1
            progress.update(progress_state[0], total=total, completed=total, description=f"[green]完成[/green]: {video_title}")

    for use_auth in auth_sequence:
        logger = _YTDLPLogger()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                progress_state[0] = progress.add_task(f"准备下载: {video_title} [client={client} fmt={fmt}]", total=None)
                opts = _build_single_video_opts(
                    fmt=fmt,
                    client=client,
                    download_video=download_video,
                    use_auth=use_auth,
                    progress_hook=hook,
                )
                opts["logger"] = logger
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([video_url])
            used_auth_fallback = used_auth_fallback or not use_auth
            break
        except yt_dlp.utils.DownloadError as exc:
            error_message = "\n".join(_dedupe_keep_order([*logger.combined_messages(), str(exc)]))
            error_type = _classify_download_error(Exception(error_message))
            if use_auth and _should_retry_without_auth(error_type):
                used_auth_fallback = True
                continue
            return {
                "success": False,
                "error_type": error_type,
                "error_message": error_message,
            }
        except Exception as exc:
            error_message = "\n".join(_dedupe_keep_order([*logger.combined_messages(), str(exc)]))
            error_type = _classify_download_error(Exception(error_message))
            if use_auth and _should_retry_without_auth(error_type):
                used_auth_fallback = True
                continue
            return {
                "success": False,
                "error_type": error_type,
                "error_message": error_message,
            }

    if not download_video:
        has_info = bool(list(WORKDIR.glob(f"*{video_id}*.info.json")))
        if has_info:
            return {"success": True, "error_type": "", "error_message": "", "used_auth_fallback": used_auth_fallback}
        return {"success": False, "error_type": "other", "error_message": "info.json not found"}

    valid, msg = verify_downloaded_video(
        WORKDIR,
        video_id,
        root_only=True,
        required_min_height=required_min_height,
    )
    if valid:
        return {"success": True, "error_type": "", "error_message": "", "used_auth_fallback": used_auth_fallback}
    return {"success": False, "error_type": "other", "error_message": f"verify_failed: {msg}"}


def _extract_entries(url: str, clients: list[str]) -> list[dict[str, Any]]:
    def extract_once(*, use_auth: bool) -> dict[str, Any] | None:
        logger = _YTDLPLogger()
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "ignoreerrors": True,
            "logger": logger,
            "extractor_args": _build_youtube_extractor_args(*clients),
        }
        apply_auth_options(opts, use_auth=use_auth)
        if PLAYLIST_ITEMS:
            opts["playlist_items"] = PLAYLIST_ITEMS
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    info = extract_once(use_auth=True)
    if not info and AUTH_METHOD in AUTHENTICATED_AUTH_METHODS:
        info = extract_once(use_auth=False)
    if not info:
        return []
    if "entries" in info and info["entries"]:
        return [entry for entry in info["entries"] if entry]
    if "id" in info:
        return [info]
    return []


def download_videos_enhanced(
    url: str,
    *,
    fmt: str = DEFAULT_FORMAT,
    download_video: bool = DOWNLOAD_VIDEO,
    fallback_clients_raw: str | None = None,
    fallback_formats_raw: str | None = None,
    print_formats_on_fail: bool = PRINT_FORMATS_ON_FAIL,
) -> None:
    clients = _normalize_clients(fallback_clients_raw)
    formats = _normalize_formats(fmt, fallback_formats_raw)

    console.print(f"[bold magenta]=== Download Task Started ({time_tag()}) ===[/bold magenta]")
    console.print(f"Target URL: {url}")
    console.print(f"Workdir: {WORKDIR}")
    console.print(f"Clients: {clients}")
    if AUTH_METHOD in AUTHENTICATED_AUTH_METHODS:
        console.print("[cyan]Auth mode detected: using yt-dlp default authenticated client first and filtering cookie-unsafe clients[/cyan]")
    if download_video:
        console.print(f"Formats: {formats}")
    else:
        console.print("Mode: thumbnail + info.json only")

    WORKDIR.mkdir(parents=True, exist_ok=True)
    downloaded_ids = init_download_record_from_info_json()
    entries = _extract_entries(url, clients)
    if not entries:
        raise RuntimeError("No downloadable entries found.")

    candidates: list[dict[str, str]] = []
    for entry in entries:
        title = entry.get("title", "N/A")
        if title == "[Private video]":
            continue
        video_id = entry.get("id", "N/A")
        video_url = entry.get("webpage_url") or entry.get("url")
        if video_id == "N/A" or not video_url:
            continue
        filename = f"{title}_{video_id}.mp4".replace("/", "_").replace("\\", "_")
        candidates.append({"id": video_id, "title": title, "url": video_url, "filename": filename})

    videos_to_download = [v for v in candidates if v["id"] not in downloaded_ids]
    if not videos_to_download:
        console.print("[green]No new videos to download.[/green]")
        return

    success_count = 0
    failure_count = 0
    for idx, video in enumerate(videos_to_download, 1):
        title_for_log = video["filename"]
        video_id = video["id"]
        video_url = video["url"]
        console.print(f"\n[bold]({idx}/{len(videos_to_download)}) Starting {title_for_log}[/bold]")

        success = False
        attempt_failures: list[str] = []
        current_formats = ["best"] if not download_video else formats
        required_min_height: int | None = None
        if download_video:
            max_available_height = _probe_source_max_height(video_url, clients)
            if max_available_height is not None and max_available_height >= MIN_VIDEO_HEIGHT:
                required_min_height = MIN_VIDEO_HEIGHT
                console.print(
                    f"[cyan]Source max height: {max_available_height}p, enforcing >= {required_min_height}p[/cyan]"
                )
            elif max_available_height is not None:
                console.print(
                    f"[cyan]Source max height: {max_available_height}p, accepting best available below {MIN_VIDEO_HEIGHT}p[/cyan]"
                )
            else:
                console.print("[yellow]Unable to determine source max height, skipping resolution gate[/yellow]")
        for client in clients:
            for format_item in current_formats:
                result = _attempt_download(
                    video_url=video_url,
                    video_title=title_for_log,
                    video_id=video_id,
                    client=client,
                    fmt=format_item,
                    download_video=download_video,
                    required_min_height=required_min_height,
                )
                if result["success"]:
                    add_downloaded_id(video_id, downloaded_ids)
                    success = True
                    success_count += 1
                    console.print(f"[green]Success: client={client}, fmt={format_item}[/green]")
                    if result.get("used_auth_fallback"):
                        console.print("[cyan]Retry without auth succeeded for this video[/cyan]")
                    break
                failure_line = f"client={client} fmt={format_item} type={result['error_type']} :: {result['error_message']}"
                attempt_failures.append(failure_line)
                console.print(f"[yellow]Failed: client={client}, fmt={format_item}, type={result['error_type']}[/yellow]")
                if not _should_continue_format_fallback(result["error_type"]):
                    break
            if success:
                break

        if not success:
            failure_count += 1
            console.print(f"[red]Failed to download {title_for_log}[/red]")
            if attempt_failures:
                console.print(f"[yellow]Last error:[/yellow] {attempt_failures[-1]}")
            if print_formats_on_fail:
                _print_formats_summary(video_url, clients)

    console.print("\n[bold magenta]=== Download Task Finished ===[/bold magenta]")
    console.print(f"Success: {success_count}")
    if failure_count > 0:
        raise RuntimeError(f"{failure_count} videos failed to download.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos or playlists with yt-dlp fallback logic and optional bgutil auto-start.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.step1_download
  python -m pipeline.step1_download --source "https://www.youtube.com/playlist?list=xxx"
  python -m pipeline.step1_download --force-redownload-id "abc123,def456"
  python -m pipeline.step1_download --fallback-clients "android,default,web"
  python -m pipeline.step1_download --fallback-formats "bestvideo+bestaudio/best,best"
""",
    )
    parser.add_argument("--source", default=VIDEO_URL, help="YouTube URL (single video or playlist)")
    parser.add_argument("--fmt", default=DEFAULT_FORMAT, help="Primary yt-dlp format selector")
    parser.add_argument("--download-video", action="store_true", default=DOWNLOAD_VIDEO, help="Download video files")
    parser.add_argument(
        "--force-redownload-id",
        type=str,
        default="",
        help="Comma-separated IDs to remove from download record before running",
    )
    parser.add_argument(
        "--fallback-clients",
        type=str,
        default=None,
        help="Comma-separated fallback clients, e.g. android,default,web",
    )
    parser.add_argument(
        "--fallback-formats",
        type=str,
        default=None,
        help="Comma-separated fallback formats. First wins.",
    )
    parser.add_argument(
        "--print-formats-on-fail",
        action="store_true",
        default=PRINT_FORMATS_ON_FAIL,
        help="Print available formats when all fallbacks fail",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to download_workdir from config.yaml.",
    )
    args = parser.parse_args()

    if args.workdir:
        global WORKDIR, DOWNLOAD_RECORD_FILE
        WORKDIR = args.workdir.resolve()
        DOWNLOAD_RECORD_FILE = WORKDIR / "downloaded_ids.json"

    if args.force_redownload_id:
        ids = [v.strip() for v in args.force_redownload_id.split(",") if v.strip()]
        if ids:
            existing = load_downloaded_ids()
            removed = 0
            for vid in ids:
                if vid in existing:
                    existing.discard(vid)
                    removed += 1
                _remove_existing_video_artifacts(WORKDIR, vid)
            if removed > 0:
                save_downloaded_ids(existing)
                console.print(f"[yellow]已从记录中移除 {removed} 个 ID: {ids}[/yellow]")

    try:
        ensure_bgutil_provider()
        download_videos_enhanced(
            args.source,
            fmt=args.fmt,
            download_video=args.download_video,
            fallback_clients_raw=args.fallback_clients,
            fallback_formats_raw=args.fallback_formats,
            print_formats_on_fail=args.print_formats_on_fail,
        )
        console.print(f"[bold green]下载流程完成。WORKDIR: {WORKDIR}[/bold green]")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
