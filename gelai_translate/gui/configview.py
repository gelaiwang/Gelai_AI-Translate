from __future__ import annotations

from pathlib import Path
from typing import Any

from gelai_translate.runtime import prepare_runtime


def _stringify(path_value: object | None) -> str:
    if path_value is None:
        return ""
    return str(path_value)


def load_config_snapshot(config_path: Path | str | None) -> dict[str, Any]:
    config_module, _ = prepare_runtime(config_path=config_path)
    return {
        "config_path": _stringify(getattr(config_module, "CONFIG_YAML_PATH", "")),
        "workdir": _stringify(getattr(config_module, "WORKDIR", "")),
        "translation_service": str(getattr(config_module, "TRANSLATION_SERVICE", "")),
        "gemini_model": str(getattr(config_module, "GEMINI_MODEL_NAME", "")),
        "asr_model": str(getattr(config_module, "ASR_WHISPER_MODEL", "")),
        "asr_device": str(getattr(config_module, "ASR_DEVICE", "")),
        "asr_compute_type": str(getattr(config_module, "ASR_COMPUTE_TYPE", "")),
        "speaker_diarization": bool(getattr(config_module, "ASR_SPEAKER_DIARIZATION", False)),
        "use_vocal_separation": bool(getattr(config_module, "ASR_USE_VOCAL_SEPARATION", False)),
        "render_codec": str(getattr(config_module, "RENDER_VIDEO_CODEC", "")),
        "render_font_en": str(getattr(config_module, "FONT_FAMILY_EN", "")),
        "render_font_cn": str(getattr(config_module, "FONT_FAMILY_CN", "")),
        "render_output_suffix": str(getattr(config_module, "RENDER_OUTPUT_SUFFIX", "")),
        "hf_token_env": str(getattr(config_module, "ASR_HF_TOKEN_ENV", "HF_TOKEN")),
    }


def build_config_summary(snapshot: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Config: {snapshot.get('config_path', '') or '-'}",
            f"Workdir: {snapshot.get('workdir', '') or '-'}",
            f"Translation: {snapshot.get('translation_service', '-')}",
            f"Gemini model: {snapshot.get('gemini_model', '-')}",
            f"ASR: model={snapshot.get('asr_model', '-')}, device={snapshot.get('asr_device', '-')}, compute_type={snapshot.get('asr_compute_type', '-')}",
            f"Diarization: {snapshot.get('speaker_diarization', False)}",
            f"Vocal separation: {snapshot.get('use_vocal_separation', False)}",
            f"Render: codec={snapshot.get('render_codec', '-')}, output_suffix={snapshot.get('render_output_suffix', '-')}",
            f"Fonts: EN={snapshot.get('render_font_en', '-')}, CN={snapshot.get('render_font_cn', '-')}",
        ]
    )

