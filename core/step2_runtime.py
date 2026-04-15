from __future__ import annotations

import torch
import whisperx
from pyannote.audio import Pipeline

from config import (
    ASR_ALIGNMENT_LANGUAGE,
    ASR_BATCH_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_HF_TOKEN,
    ASR_HF_TOKEN_ENV,
    ASR_LANGUAGE,
    ASR_WHISPER_MODEL,
)


def detect_runtime_profile() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        gpu_name = props.name
    else:
        vram_gb = 0.0
        gpu_name = "CPU"

    device = ("cuda" if cuda_available else "cpu") if ASR_DEVICE == "auto" else ASR_DEVICE
    compute_type = ("float16" if device == "cuda" else "float32") if ASR_COMPUTE_TYPE == "auto" else ASR_COMPUTE_TYPE

    if ASR_BATCH_SIZE == "auto" or ASR_BATCH_SIZE is None:
        if device != "cuda":
            batch_size = 4
        elif ASR_WHISPER_MODEL == "large-v3":
            batch_size = 16 if vram_gb >= 20 else 8
        elif ASR_WHISPER_MODEL == "medium":
            batch_size = 16 if vram_gb >= 10 else 8
        else:
            batch_size = 8 if vram_gb >= 10 else 4
    else:
        batch_size = int(ASR_BATCH_SIZE)

    if device == "cuda":
        recommended_model = "small" if vram_gb < 8 else "medium" if vram_gb < 16 else "large-v3"
    else:
        recommended_model = "small"

    return {
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "recommended_model": recommended_model,
    }


def print_runtime_profile(console, profile: dict[str, object]) -> None:
    device = str(profile["device"])
    compute_type = str(profile["compute_type"])
    batch_size = int(profile["batch_size"])
    recommended_model = str(profile["recommended_model"])
    if device == "cuda":
        console.print(
            f"检测到设备：[bold green]CUDA[/bold green] "
            f"({profile['gpu_name']}, {float(profile['vram_gb']):.1f} GB VRAM)，"
            f"计算类型：{compute_type}，batch_size：{batch_size}"
        )
    else:
        console.print(
            f"检测到设备：[bold yellow]CPU[/bold yellow]，"
            f"计算类型：{compute_type}，batch_size：{batch_size}"
        )
    console.print(f"WhisperX 模型配置：{ASR_WHISPER_MODEL}")
    console.print(f"推荐模型档位：{recommended_model}")
    if ASR_WHISPER_MODEL != recommended_model:
        console.print(
            f"[yellow]当前模型 {ASR_WHISPER_MODEL} 与设备档位不完全匹配。"
            " 若遇到 OOM 或速度问题，可切换到推荐档位。[/yellow]"
        )


def require_hf_token(feature_name: str) -> str:
    token = ASR_HF_TOKEN
    if token:
        return token
    raise RuntimeError(
        f"{feature_name} requires a Hugging Face token. "
        f"Set {ASR_HF_TOKEN_ENV} in the environment or run `huggingface-cli login`, "
        "and accept the gated pyannote model terms first."
    )


def preload_models(console, logger) -> tuple[object, object, tuple[object, object], str, int]:
    console.print("\n[bold cyan]--- 正在预加载模型... ---[/bold cyan]")
    profile = detect_runtime_profile()
    device = str(profile["device"])
    compute_type = str(profile["compute_type"])
    batch_size = int(profile["batch_size"])
    print_runtime_profile(console, profile)

    console.print(f"  [cyan]1/3：正在加载 WhisperX ASR 模型（{ASR_WHISPER_MODEL}）...[/cyan]")
    try:
        asr_model = whisperx.load_model(
            ASR_WHISPER_MODEL,
            device,
            language=ASR_LANGUAGE,
            compute_type=compute_type,
        )
        console.print("    [green]WhisperX ASR 模型加载成功。[/green]")
    except Exception as e:
        console.print("    [bold red]WhisperX ASR 模型加载失败。[/bold red]")
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            console.print(
                "[bold yellow]提示：可以尝试把 asr.model_name 调低到 medium 或 small，"
                "或手动降低 asr.batch_size。[/bold yellow]"
            )
        logger.critical(f"关键步骤失败：WhisperX ASR 模型加载失败。错误：{e}")
        raise

    console.print("  [cyan]2/3：正在加载 Pyannote VAD 模型（pyannote/voice-activity-detection）...[/cyan]")
    try:
        vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=require_hf_token("Pyannote VAD"),
        )
        if device == "cuda":
            vad_pipeline.to(torch.device(device))
        console.print("    [green]Pyannote VAD 模型加载成功。[/green]")
    except Exception as e:
        console.print("    [bold red]Pyannote VAD 模型加载失败。[/bold red]")
        logger.critical(f"关键步骤失败：Pyannote VAD 模型加载失败。错误：{e}")
        console.print(
            f"    [bold yellow]提示：请确认已经设置环境变量 {ASR_HF_TOKEN_ENV} "
            "或执行 `huggingface-cli login`，并在 Hugging Face 上同意 pyannote 模型协议。[/bold yellow]"
        )
        raise

    console.print(f"  [cyan]3/3：正在加载 WhisperX 对齐模型（{ASR_ALIGNMENT_LANGUAGE}）...[/cyan]")
    try:
        align_model_tuple = whisperx.load_align_model(language_code=ASR_ALIGNMENT_LANGUAGE, device=device)
        console.print("    [green]WhisperX 对齐模型加载成功。[/green]")
    except Exception as e:
        console.print("    [bold red]WhisperX 对齐模型加载失败。[/bold red]")
        logger.critical(f"关键步骤失败：WhisperX 对齐模型加载失败。错误：{e}")
        raise

    console.print("[green]所有模型均已加载成功。[/green]")
    return asr_model, vad_pipeline, align_model_tuple, device, batch_size
