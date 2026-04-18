from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

from gelai_translate.runtime import prepare_runtime


def _status(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": ok, "detail": detail}


def run_environment_checks(config_path: Path | str | None) -> dict[str, dict[str, Any]]:
    config_module, _ = prepare_runtime(config_path=config_path)
    ffmpeg_bin = str(getattr(config_module, "RENDER_FFMPEG_BIN", "ffmpeg"))
    ffprobe_bin = str(getattr(config_module, "RENDER_FFPROBE_BIN", "ffprobe"))
    hf_env = str(getattr(config_module, "ASR_HF_TOKEN_ENV", "HF_TOKEN"))

    checks: dict[str, dict[str, Any]] = {}
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    checks["GEMINI_API_KEY"] = _status(bool(gemini_key), "set" if gemini_key else "missing")

    hf_token = os.getenv(hf_env, "").strip()
    checks[hf_env] = _status(bool(hf_token), "set" if hf_token else "missing")

    ffmpeg_resolved = shutil.which(ffmpeg_bin)
    checks["ffmpeg"] = _status(bool(ffmpeg_resolved), ffmpeg_resolved or f"{ffmpeg_bin} not found")

    ffprobe_resolved = shutil.which(ffprobe_bin)
    checks["ffprobe"] = _status(bool(ffprobe_resolved), ffprobe_resolved or f"{ffprobe_bin} not found")

    torch_spec = importlib.util.find_spec("torch")
    checks["torch"] = _status(torch_spec is not None, "installed" if torch_spec else "missing")

    whisperx_spec = importlib.util.find_spec("whisperx")
    checks["whisperx"] = _status(whisperx_spec is not None, "installed" if whisperx_spec else "missing")

    pyannote_spec = importlib.util.find_spec("pyannote.audio")
    checks["pyannote.audio"] = _status(pyannote_spec is not None, "installed" if pyannote_spec else "missing")

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            detail = f"{props.name} ({props.total_memory / (1024 ** 3):.1f} GB)"
            checks["CUDA"] = _status(True, detail)
        else:
            checks["CUDA"] = _status(False, "not available")
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["CUDA"] = _status(False, f"probe failed: {exc}")

    return checks


def format_environment_checks(results: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for name, result in results.items():
        prefix = "OK" if result["ok"] else "MISSING"
        lines.append(f"[{prefix}] {name}: {result['detail']}")
    return "\n".join(lines)

