from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


def normalize_stem(stem: str) -> str:
    normalized = stem.replace("?", "-").replace(":", "-").replace("&", "and")
    normalized = re.sub(r"[\W_]+", "-", normalized)
    normalized = re.sub(r"--+", "-", normalized)
    normalized = normalized.lower().strip("-")
    max_len = 50
    if len(normalized) > max_len:
        hash_suffix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:6]
        truncated_stem = normalized[:max_len].strip("-")
        return f"{truncated_stem}-{hash_suffix}"
    return normalized


def discover_input_videos(console, workdir: Path, video_exts: list[str]) -> list[Path]:
    initial_videos = [f for f in workdir.iterdir() if f.is_file() and f.suffix.lower() in video_exts]
    if initial_videos:
        return initial_videos
    console.print("\n[yellow]根目录下未找到视频，尝试在一级子目录中寻找断点续跑任务...[/yellow]")
    for sub_dir in workdir.iterdir():
        if not sub_dir.is_dir():
            continue
        for candidate in sub_dir.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in video_exts:
                initial_videos.append(candidate)
    return initial_videos


def project_needs_diarization(asr_dir: Path) -> bool:
    for json_file in asr_dir.glob("segment_*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = data.get("segments", [])
            if segments and "speaker" not in segments[0]:
                return True
        except Exception:
            return True
    return False


def select_videos_to_process(console, workdir: Path, initial_videos: list[Path], speaker_diarization: bool) -> list[Path]:
    console.print("\n[bold cyan]--- 正在检查已完成任务... ---[/bold cyan]")
    videos_to_process: list[Path] = []
    for video_file in initial_videos:
        normalized_stem = normalize_stem(video_file.stem)
        project_dir = workdir / normalized_stem
        asr_dir = project_dir / "asr"
        if asr_dir.exists() and any(asr_dir.iterdir()):
            if speaker_diarization and project_needs_diarization(asr_dir):
                console.print(f"需要补跑说话人识别：{video_file.name}")
                videos_to_process.append(video_file)
                continue
            console.print(f"已跳过（找到已有输出）：{video_file.name}")
            continue
        videos_to_process.append(video_file)
    return videos_to_process


def cleanup_wav_file(console, logger, project_dir: Path) -> None:
    console.print("  [cyan]清理 .wav 临时文件...[/cyan]")
    for wav_file in project_dir.glob("*.wav"):
        try:
            wav_file.unlink()
            console.print(f"    - 已移除临时音频：{wav_file.name}")
        except OSError as e:
            logger.error(f"移除 {wav_file} 时出错：{e}")


def find_associated_thumbnail(video_file: Path) -> Path | None:
    for ext in [".webp", ".jpg", ".jpeg", ".png"]:
        candidate = video_file.with_suffix(ext)
        if candidate.exists():
            return candidate
    video_id_match = re.search(r"([A-Za-z0-9_-]{11})$", video_file.stem)
    if video_id_match:
        video_id = video_id_match.group(1)
        for thumbnail in video_file.parent.glob(f"*{video_id}*.webp"):
            if thumbnail.exists():
                return thumbnail
    return None


def find_associated_info_json(video_file: Path) -> Path | None:
    primary = video_file.with_suffix(".info.json")
    if primary.exists():
        return primary
    fallback = video_file.with_suffix(".json")
    if fallback.exists():
        return fallback
    return None


def move_sidecar_file(console, logger, source: Path, destination: Path, description: str) -> None:
    if destination.exists():
        console.print(f"  [dim]{description}已存在于目标位置：{destination.name}[/dim]")
        return
    console.print(f"  [cyan]发现{description}，移动到项目目录：{source.name} -> {destination.name}[/cyan]")
    try:
        shutil.move(str(source), str(destination))
    except Exception as move_error:
        console.print(f"  [yellow]移动{description}失败：{move_error}，尝试改为复制...[/yellow]")
        try:
            shutil.copy2(str(source), str(destination))
            source.unlink()
            console.print(f"  [green]通过复制方式成功移动{description}[/green]")
        except Exception as copy_error:
            console.print(f"  [red]复制{description}也失败：{copy_error}，跳过。[/red]")
            logger.error(f"移动/复制{description}失败：{source} -> {destination}，错误：{copy_error}")


def find_associated_vtt(video_path: Path) -> Path | None:
    for candidate in [video_path.with_suffix(".vtt"), video_path.parent / f"{video_path.stem}.en.vtt"]:
        if candidate.exists():
            return candidate
    return None


def prepare_project_inputs(console, logger, video_path: Path, workdir: Path) -> tuple[Path, Path]:
    is_root_video = video_path.parent == workdir
    normalized_stem = normalize_stem(video_path.stem)
    if is_root_video:
        project_dir = workdir / normalized_stem
        final_video_path = project_dir / f"{normalized_stem}{video_path.suffix}"
        console.print("  [dim]检测到根目录视频，按标准项目目录处理。[/dim]")
    else:
        project_dir = video_path.parent
        final_video_path = video_path
        console.print(f"  [dim]检测到子目录视频，复用现有项目目录：{project_dir.name}[/dim]")

    project_dir.mkdir(exist_ok=True)
    thumbnail_source = find_associated_thumbnail(video_path)
    info_json_source = find_associated_info_json(video_path)
    vtt_source = find_associated_vtt(video_path)

    if is_root_video and not final_video_path.exists():
        shutil.move(str(video_path), str(final_video_path))
    elif not final_video_path.exists():
        logger.error(f"视频文件路径异常：{video_path}")
        raise FileNotFoundError(f"Unexpected video path state: {video_path}")

    if info_json_source and info_json_source.exists():
        move_sidecar_file(console, logger, info_json_source, project_dir / info_json_source.name, " info.json")
    else:
        console.print("  [dim]未找到对应的 info.json 元数据文件，跳过。[/dim]")

    if vtt_source and vtt_source.exists():
        move_sidecar_file(console, logger, vtt_source, project_dir / f"[EN-AUTO]-{normalized_stem}.vtt", "关联 VTT 字幕")

    if thumbnail_source and thumbnail_source.exists():
        if thumbnail_source.suffix.lower() == ".webp":
            final_thumbnail_path = project_dir / f"{normalized_stem}.jpg"
            console.print(f"  [cyan]发现 WebP 缩略图，直接重命名为 JPG：{thumbnail_source.name} -> {final_thumbnail_path.name}[/cyan]")
        else:
            final_thumbnail_path = project_dir / f"{normalized_stem}{thumbnail_source.suffix}"
            console.print(f"  [cyan]发现并移动关联缩略图：{thumbnail_source.name} -> {final_thumbnail_path.name}[/cyan]")
        if final_thumbnail_path.exists():
            console.print(f"  [dim]关联缩略图已存在于目标位置：{final_thumbnail_path.name}[/dim]")
        else:
            shutil.move(str(thumbnail_source), str(final_thumbnail_path))
    else:
        console.print("  [dim]未找到对应缩略图文件，跳过。[/dim]")

    return project_dir, final_video_path
