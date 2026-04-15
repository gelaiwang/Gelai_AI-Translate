from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Optional

import torch
import whisperx

from config import ASR_HF_TOKEN_ENV
from core.step2_audio import VOCALS_SUFFIX
from core.step2_runtime import require_hf_token


def run_diarization_on_project(console, logger, project_dir: Path, device: str, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> bool:
    chunk_dir = project_dir / "chunks"
    asr_dir = project_dir / "asr"
    if not asr_dir.exists():
        console.print("    [yellow]未找到 asr 目录，跳过说话人识别。[/yellow]")
        return False

    asr_json_list = sorted(asr_dir.glob("segment_*.json"))
    if not asr_json_list:
        console.print("    [dim]没有 ASR JSON 文件，跳过说话人识别。[/dim]")
        return True

    files_to_process = []
    for json_path in asr_json_list:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = data.get("segments", [])
            if segments and "speaker" not in segments[0]:
                files_to_process.append(json_path)
        except Exception:
            files_to_process.append(json_path)

    if not files_to_process:
        console.print("    [dim green]所有 ASR 结果都已包含 speaker 字段，跳过。[/dim green]")
        return True

    console.print(f"    [bold cyan]开始说话人识别（{len(files_to_process)} 个文件）...[/bold cyan]")
    diarize_model = None
    try:
        from whisperx.diarize import DiarizationPipeline

        console.print("      正在加载说话人识别模型（pyannote/speaker-diarization-3.1）...")
        diarize_model = DiarizationPipeline(
            use_auth_token=require_hf_token("Speaker diarization"),
            device=device,
        )
        console.print("      [green]说话人识别模型加载成功。[/green]")

        for idx, json_path in enumerate(files_to_process, start=1):
            stem = json_path.stem
            console.print(f"      [cyan]>[/cyan] 识别 {idx}/{len(files_to_process)}：{json_path.name}")
            vocals_path = chunk_dir / f"{stem}{VOCALS_SUFFIX}"
            wav_path = chunk_dir / f"{stem}.wav"
            if vocals_path.exists():
                audio_input = str(vocals_path)
            elif wav_path.exists():
                audio_input = str(wav_path)
            else:
                console.print(f"        [yellow]未找到对应音频，跳过：{stem}[/yellow]")
                continue

            try:
                diarize_kwargs = {}
                if min_speakers is not None:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers
                diarize_df = diarize_model(audio_input, **diarize_kwargs)
                with open(json_path, "r", encoding="utf-8") as f:
                    asr_result = json.load(f)
                result_with_speakers = whisperx.assign_word_speakers(diarize_df, asr_result)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result_with_speakers, f, indent=2, ensure_ascii=False)
                speakers = {seg.get("speaker") for seg in result_with_speakers.get("segments", []) if seg.get("speaker")}
                console.print(f"        [green]完成，识别到 {len(speakers)} 位说话人：{', '.join(sorted(speakers))}[/green]")
            except Exception as e:
                console.print(f"        [red]说话人识别失败：{e}[/red]")
                logger.error(f"Diarization 失败：{json_path.name}，错误：{e}")
                continue
    except Exception as e:
        console.print(f"    [red]说话人识别模型加载失败：{e}[/red]")
        logger.error(f"说话人识别模型加载失败：{e}")
        console.print(
            f"    [bold yellow]提示：请确认已经设置环境变量 {ASR_HF_TOKEN_ENV} "
            "并在 Hugging Face 上同意 pyannote/speaker-diarization-3.1 模型协议。[/bold yellow]"
        )
        return False
    finally:
        console.print("    [dim]释放说话人识别模型显存...[/dim]")
        if diarize_model is not None:
            del diarize_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.print("    [dim green]显存已释放。[/dim green]")

    return True
