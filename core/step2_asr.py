from __future__ import annotations

import json
from pathlib import Path

import whisperx

from config import ASR_LANGUAGE
from core.step2_audio import VOCALS_SUFFIX


def run_whisperx_on_project(console, logger, project_dir: Path, asr_model, vad_pipeline, align_model_tuple, device: str, batch_size: int = 16) -> bool:
    chunk_dir = project_dir / "chunks"
    asr_dir = project_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    wav_list = sorted(chunk_dir.glob("segment_*.wav"))

    total_chunks = len(wav_list)
    console.print(f"    [bold]发现 {total_chunks} 个音频块，开始处理...[/bold]")
    if not wav_list:
        return True

    align_model, align_metadata = align_model_tuple
    for i, wav_path in enumerate(wav_list, 1):
        output_asr_json_path = asr_dir / f"{wav_path.stem}.json"
        console.print(f"      [cyan]>[/cyan] [yellow]处理块 {i}/{total_chunks}：[/yellow] {wav_path.name}")
        if output_asr_json_path.exists():
            console.print("        [dim green]结果已存在，跳过。[/dim green]")
            continue

        vocals_path = chunk_dir / f"{wav_path.stem}{VOCALS_SUFFIX}"
        if vocals_path.exists():
            audio_input_path = vocals_path
            console.print(f"        [cyan]使用人声音频：{vocals_path.name}[/cyan]")
        else:
            audio_input_path = wav_path
            console.print(f"        [yellow]未找到人声文件，使用原始音频：{wav_path.name}[/yellow]")

        try:
            console.print("        [dim]1/4：正在执行 VAD 语音活动检测...[/dim]")
            speech_timeline = vad_pipeline(str(audio_input_path))
            if not speech_timeline.get_timeline().support():
                console.print("        [dim yellow]未检测到语音，写入空结果并跳过。[/dim yellow]")
                result_data = {"language": ASR_LANGUAGE, "segments": [], "word_segments": []}
            else:
                console.print("        [dim]检测到语音，继续处理。[/dim]")
                console.print("        [dim]2/4：正在加载音频数据...[/dim]")
                audio_data = whisperx.load_audio(str(audio_input_path))
                console.print("        [dim]3/4：正在使用 WhisperX 执行语音识别...[/dim]")
                transcription_result = asr_model.transcribe(audio_data, batch_size=batch_size, language=ASR_LANGUAGE)
                if transcription_result.get("segments"):
                    console.print("        [dim]4/4：正在执行词级对齐...[/dim]")
                    result_data = whisperx.align(
                        transcription_result["segments"],
                        align_model,
                        align_metadata,
                        audio_data,
                        device,
                        return_char_alignments=False,
                    )
                else:
                    console.print("        [dim yellow]未识别出任何片段，写入空结果。[/dim yellow]")
                    result_data = {"language": ASR_LANGUAGE, "segments": [], "word_segments": []}
            with open(output_asr_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
            console.print(f"        [green]结果已保存：{output_asr_json_path.name}[/green]")
        except Exception as e:
            logger.error(f"ASR 过程出错。文件：{wav_path.name}。错误：{e}")
            console.print(f"        [bold red]处理 {wav_path.name} 时发生错误：{e}[/bold red]")
            continue

    return True
