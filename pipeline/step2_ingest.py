# step2_ingest.py

# PyTorch 2.6+ compatibility shim.
# Some WhisperX / pyannote / speechbrain paths still expect weights_only=False.
import torch
import torch.serialization
import typing
import collections
from contextlib import contextmanager


@contextmanager
def legacy_torch_load_compatibility():
    """
    Scope the unsafe compatibility override to legacy model-loading code paths.
    This avoids leaving weights_only=False enabled for the entire process.
    """
    original_torch_load = torch.load
    original_serialization_load = getattr(torch.serialization, "load", None)

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    def patched_serialization_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_serialization_load(*args, **kwargs)

    torch.load = patched_torch_load
    if original_serialization_load is not None:
        torch.serialization.load = patched_serialization_load
    try:
        yield
    finally:
        torch.load = original_torch_load
        if original_serialization_load is not None:
            torch.serialization.load = original_serialization_load

try:
    import omegaconf
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.base import ContainerMetadata, Metadata, Node
    from omegaconf.nodes import ValueNode, AnyNode, IntegerNode, FloatNode, BooleanNode, StringNode
    _safe_classes = [
        DictConfig, ListConfig, ContainerMetadata, Metadata, Node,
        ValueNode, AnyNode, IntegerNode, FloatNode, BooleanNode, StringNode,
    ]
    # 尝试补充更多 omegaconf 类型
    for name in dir(omegaconf):
        obj = getattr(omegaconf, name)
        if isinstance(obj, type):
            _safe_classes.append(obj)
    torch.serialization.add_safe_globals(_safe_classes)
except Exception:
    pass

try:
    torch.serialization.add_safe_globals([
        typing.Any,
        list,
        dict,
        tuple,
        set,
        collections.defaultdict,
        collections.OrderedDict,
        collections.Counter,
    ])
except Exception:
    pass

import sys
import gc
import argparse
from pathlib import Path
import json
import logging
import time
from typing import Optional
from rich.console import Console

# 人声分离依赖
try:
    from audio_separator.separator import Separator
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False

# --- 全局初始化 ---
console = Console()

try:
    from core.audio_extract import extract_audio
    from core.audio_split import split_audio
    from core.step2_audio import run_vocal_separation_on_project
    from core.step2_asr import run_whisperx_on_project
    from core.step2_diarization import run_diarization_on_project
    from core.step2_inputs import (
        discover_input_videos,
        normalize_stem,
        prepare_project_inputs,
        select_videos_to_process,
    )
    from core.step2_runtime import preload_models
except ImportError:
    console.print("[bold red]错误：无法导入 core 模块（core/audio_extract.py 或 core/audio_split.py）。请确认项目结构和 PYTHONPATH 正确。[/bold red]")


    sys.exit(1)

# 动态导入外部库
import whisperx

# 从统一配置模块导入参数
try:
    from config import (
        WORKDIR as BASE_WORKDIR,
        ASR_LANGUAGE,
        ASR_USE_VOCAL_SEPARATION,
        ASR_SNR_THRESHOLD,
        ASR_SPEAKER_DIARIZATION,
        ASR_MIN_SPEAKERS,
        ASR_MAX_SPEAKERS,
        ASR_HF_TOKEN_ENV,
    )
except ImportError:
    console.print("[bold red]错误：无法从 config 模块导入配置。请检查 `config/settings.py`。[/bold red]")
    sys.exit(1)

# --- 日志配置 ---
BASE_WORKDIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_WORKDIR / "pipeline_errors.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- 辅助函数 ---

def run_with_retries(func, max_retries: int, retry_delay: int, description: str, *args, **kwargs):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            console.print(f"  [cyan]{description}（尝试 {attempt}/{max_retries}）...[/cyan]")
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            console.print(f"    [yellow]尝试 {attempt} 失败：{e.__class__.__name__}[/yellow]")
            if attempt < max_retries: time.sleep(retry_delay)
            else:
                console.print(f"    [red]所有重试均已失败。[/red]")
                raise last_exception


# --- SNR 估算函数：用于判断是否需要做人声分离 ---
# --- 主执行流程 ---
def main():
    global BASE_WORKDIR, LOG_FILE
    parser = argparse.ArgumentParser(description="视频导入流程（音频提取 + 人声分离 + ASR）")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="工作目录（默认使用 config.yaml 中的 workdir）")
    args = parser.parse_args()
    if args.workdir:
        BASE_WORKDIR = args.workdir.resolve()
        BASE_WORKDIR.mkdir(parents=True, exist_ok=True)
        LOG_FILE = BASE_WORKDIR / "pipeline_errors.log"

    console.print("[bold magenta]=== 视频导入流程启动 ===[/bold magenta]")
    console.print(f"工作目录（WORKDIR）：{BASE_WORKDIR}")

    max_retries, retry_delay = 3, 5

    video_exts = [".mp4", ".mov", ".mkv", ".flv", ".avi", ".webm"]
    console.print("\n[bold cyan]--- 模式：处理本地文件（扫描 WORKDIR） ---[/bold cyan]")
    initial_videos = discover_input_videos(console, BASE_WORKDIR, video_exts)
    if not initial_videos:
        console.print("\n[yellow]仍未找到任何可处理的视频文件。任务结束。[/yellow]")
        sys.exit(0)

    videos_to_process = select_videos_to_process(console, BASE_WORKDIR, initial_videos, ASR_SPEAKER_DIARIZATION)
    if not videos_to_process:
        console.print("\n[green]找到的视频均已处理完成，无需执行新任务。[/green]")
        sys.exit(0)

    console.print(f"\n[bold green]将处理 {len(videos_to_process)} 个新视频。[/bold green]")

    try:
        with legacy_torch_load_compatibility():
            asr_model, vad_pipeline, align_model_tuple, device, batch_size = preload_models(console, logger)
    except Exception:
        sys.exit(1)

    # 阶段 3：循环处理每个视频
    for i, video_path in enumerate(videos_to_process):
        console.print(f"\n{'='*80}")
        console.print(f"[bold magenta]处理视频 {i+1}/{len(videos_to_process)}：{video_path.name}[/bold magenta]")

        normalized_stem = normalize_stem(video_path.stem)
        try:
            project_dir, final_video_path = prepare_project_inputs(console, logger, video_path, BASE_WORKDIR)
        except Exception as e:
            logger.error(f"准备项目输入失败：{video_path}，错误：{e}")
            console.print("[bold red]准备项目输入失败，跳过该视频。[/bold red]")
            continue

        try:
            with legacy_torch_load_compatibility():
                # 步骤 1：提取音频
                wav_output_path = project_dir / f"{normalized_stem}.wav"
                if not wav_output_path.exists():
                    run_with_retries(
                        extract_audio,
                        max_retries,
                        retry_delay,
                        "步骤 1/5：提取音频",
                        final_video_path,
                        wav_output_path,
                        sr=16000
                    )

                # 步骤 2：切分音频
                chunks_dir = project_dir / "chunks"
                if not (chunks_dir.exists() and any(chunks_dir.iterdir())):
                     console.print("  [yellow]提示：音频切分会搜索静音片段。根据音频长度不同，这一步可能耗时较长，请耐心等待...[/yellow]")
                     run_with_retries(split_audio, max_retries, retry_delay, "步骤 2/5：切分音频", wav_output_path, out_dir=chunks_dir)

                if ASR_USE_VOCAL_SEPARATION:
                    console.print("  [cyan]步骤 3/5：人声分离...[/cyan]")
                    run_vocal_separation_on_project(project_dir, device, vad_pipeline, ASR_SNR_THRESHOLD)
                else:
                    console.print("  [dim]步骤 3/5：人声分离已禁用，跳过。[/dim]")

                # 步骤 4：运行 ASR
                run_with_retries(
                    run_whisperx_on_project,
                    max_retries,
                    retry_delay,
                    "步骤 4/5：语音识别",
                    project_dir,
                    asr_model,
                    vad_pipeline,
                    align_model_tuple,
                    device,
                    batch_size,
                )

                # 步骤 5：说话人识别
                if ASR_SPEAKER_DIARIZATION:
                    console.print("  [cyan]步骤 5/5：说话人识别...[/cyan]")
                    run_diarization_on_project(project_dir, device, ASR_MIN_SPEAKERS, ASR_MAX_SPEAKERS)
                else:
                    console.print("  [dim]步骤 5/5：说话人识别已禁用，跳过。[/dim]")

            console.print(f"[bold green]成功处理视频：{final_video_path.name}[/bold green]")
            # cleanup_wav_file(console, logger, project_dir)

        except Exception as e:
            logger.error(f"处理视频 {video_path.name} 的流水线时发生致命错误。错误：{e}", exc_info=True)
            console.print(f"[bold red]处理该视频时发生致命错误，已记录日志。将继续处理下一个视频。[/bold red]")
            continue

    console.print(f"\n{'='*80}")
    console.print("[bold green]所有任务处理完毕。[/bold green]")
    console.print(f"  如有错误，请检查日志文件：{LOG_FILE}")

if __name__ == "__main__":
    main()
