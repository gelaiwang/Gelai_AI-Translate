from __future__ import annotations

import subprocess
import shutil
import json
from pathlib import Path


class AudioExtractionError(Exception):
    """音频提取相关的错误"""
    pass


def _check_ffmpeg_available() -> bool:
    """检查 ffmpeg 和 ffprobe 是否可用"""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _has_audio_stream(video_path: Path) -> bool:
    """
    使用 ffprobe 检测视频是否包含音频流
    
    Returns:
        True if video has audio stream, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30
        )
        if result.returncode != 0:
            return False
        
        probe_data = json.loads(result.stdout)
        return len(probe_data.get("streams", [])) > 0
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return False


def _validate_output_audio(output_path: Path, min_size_bytes: int = 1000) -> bool:
    """
    验证输出的音频文件是否有效
    
    Args:
        output_path: 输出文件路径
        min_size_bytes: 最小文件大小（字节），默认 1KB
    
    Returns:
        True if file is valid, False otherwise
    """
    if not output_path.exists():
        return False
    
    if output_path.stat().st_size < min_size_bytes:
        return False
    
    # 使用 ffprobe 验证音频文件格式
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(output_path)
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10
        )
        if result.returncode != 0:
            return False
        
        duration = float(result.stdout.strip())
        return duration > 0.1  # 至少 0.1 秒
    except (subprocess.TimeoutExpired, ValueError, Exception):
        return False


def extract_audio(video_path: Path, output_path: Path, sr: int = 16000) -> None:
    """
    Extract mono WAV audio from a video file using ffmpeg.

    Args:
        video_path: Input video path.
        output_path: Output wav path.
        sr: Sample rate (Hz), default 16000.
    
    Raises:
        AudioExtractionError: If extraction fails
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # 检查 ffmpeg 是否可用
    if not _check_ffmpeg_available():
        raise AudioExtractionError("ffmpeg 或 ffprobe 未安装或不在 PATH 中")
    
    # 检查输入文件
    if not video_path.exists():
        raise AudioExtractionError(f"输入视频文件不存在: {video_path}")
    
    # 检查视频是否包含音频流
    if not _has_audio_stream(video_path):
        raise AudioExtractionError(f"视频文件不包含音频流: {video_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=3600  # 1小时超时
        )
    except subprocess.CalledProcessError as e:
        raise AudioExtractionError(f"ffmpeg 提取音频失败: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("ffmpeg 提取音频超时（超过1小时）")
    
    # 验证输出文件
    if not _validate_output_audio(output_path):
        raise AudioExtractionError(f"提取的音频文件无效或为空: {output_path}")

