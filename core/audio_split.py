from __future__ import annotations
import json
import wave
import contextlib
from pathlib import Path
"""
Audio intelligent splitter (优化版):
1. 自适应静音阈值：根据音频平均 dBFS 动态计算
2. 最优静音点选择：在窗口内评分选择最佳切分点
3. 能量最小点备选：当无静音点时，选择能量最低的位置切分
4. 避免硬切分：确保不会在语音中间切断

依赖：pydub (pip install pydub)
"""


from typing import List, Dict, Tuple, Optional
from config import WORKDIR


def _calculate_adaptive_threshold(audio, base_offset: float = -15.0) -> float:
    """
    根据音频的平均 dBFS 动态计算静音阈值
    
    Args:
        audio: pydub AudioSegment
        base_offset: 相对于平均响度的偏移量（dB），越负越敏感
    
    Returns:
        自适应的静音阈值 (dB)
    """
    avg_dbfs = audio.dBFS
    # 限制阈值范围在 -50dB 到 -20dB 之间
    threshold = max(-50.0, min(-20.0, avg_dbfs + base_offset))
    return threshold


def _find_energy_minimum_point(audio, start_sec: float, end_sec: float, 
                                window_ms: int = 100) -> float:
    """
    在指定时间范围内找到能量最小的点，用作备选切分点
    
    Args:
        audio: pydub AudioSegment
        start_sec: 搜索起始时间（秒）
        end_sec: 搜索结束时间（秒）
        window_ms: 计算能量的窗口大小（毫秒）
    
    Returns:
        能量最小点的时间位置（秒）
    """
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    
    min_energy = float('inf')
    min_energy_pos = (start_sec + end_sec) / 2  # 默认中点
    
    # 每 50ms 采样一次
    step_ms = 50
    for pos_ms in range(start_ms, end_ms - window_ms, step_ms):
        chunk = audio[pos_ms:pos_ms + window_ms]
        if len(chunk) > 0:
            energy = chunk.dBFS if chunk.dBFS != float('-inf') else -100
            if energy < min_energy:
                min_energy = energy
                min_energy_pos = (pos_ms + window_ms / 2) / 1000.0
    
    return min_energy_pos


def _score_silence_point(silence_start: float, silence_end: float, 
                         target_pos: float, window_center: float) -> float:
    """
    为静音点评分，用于选择最优切分点
    
    评分考虑因素：
    1. 静音长度（越长越好）
    2. 与目标位置的距离（越近越好）
    
    Args:
        silence_start: 静音开始时间
        silence_end: 静音结束时间
        target_pos: 目标切分位置（max_segment_sec 边界）
        window_center: 搜索窗口中心
    
    Returns:
        评分值（越高越好）
    """
    silence_duration = silence_end - silence_start
    silence_center = (silence_start + silence_end) / 2
    
    # 静音长度得分：每秒静音 +10 分，最高 50 分
    duration_score = min(50, silence_duration * 10)
    
    # 距离得分：与目标位置的距离，每秒距离 -1 分
    distance = abs(silence_center - target_pos)
    distance_score = max(0, 30 - distance)
    
    return duration_score + distance_score


def _find_best_silence_point(silence_ranges: List[Tuple[float, float]], 
                             window_start: float, window_end: float,
                             target_pos: float) -> Optional[float]:
    """
    在指定窗口内找到最优的静音切分点
    
    Args:
        silence_ranges: 静音区间列表 [(start, end), ...]
        window_start: 搜索窗口起始时间
        window_end: 搜索窗口结束时间
        target_pos: 目标切分位置
    
    Returns:
        最优切分点时间，如果没有找到则返回 None
    """
    candidates = []
    
    for start_s, end_s in silence_ranges:
        # 检查静音区间是否与搜索窗口有交集
        if end_s < window_start or start_s > window_end:
            continue
        
        # 调整到窗口范围内
        effective_start = max(start_s, window_start)
        effective_end = min(end_s, window_end)
        
        if effective_end > effective_start:
            score = _score_silence_point(effective_start, effective_end, 
                                         target_pos, (window_start + window_end) / 2)
            # 切分点选择静音区间的中点或末端
            split_point = (effective_start + effective_end) / 2
            candidates.append((score, split_point))
    
    if not candidates:
        return None
    
    # 返回得分最高的切分点
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# -------------- 主逻辑 ------------------------------------------------------

def split_audio(
    wav_path: Path,
    out_dir: Path | None = None,
    max_segment_sec: int = 600,
    min_silence_sec: float = 0.5,  # 降低默认值以捕捉更短的停顿
    win_sec: float = 60.0,
    min_segment_sec: float = 10.0,  # 最小片段长度保护
) -> List[Dict]:
    """
    智能音频切分，返回切片信息列表
    
    Args:
        wav_path: 输入 WAV 文件路径
        out_dir: 输出目录
        max_segment_sec: 最大片段长度（秒）
        min_silence_sec: 最小静音长度（秒）
        win_sec: 搜索窗口大小（秒）
        min_segment_sec: 最小片段长度（秒），避免过短片段
    
    Returns:
        [{\"file\": \"segment_000.wav\", \"start\": 0, \"end\": 12.34}, ...]
    """
    from pydub import AudioSegment
    from pydub.silence import detect_silence

    out_dir = out_dir or WORKDIR / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)
    sr = audio.frame_rate
    duration = len(audio) / 1000.0

    # ==================== 自适应阈值计算 ====================
    silence_threshold_db = _calculate_adaptive_threshold(audio)
    min_silence_len_ms = int(min_silence_sec * 1000)
    
    print(f"📊 音频分析: 平均响度 {audio.dBFS:.1f} dBFS, 时长 {duration:.1f}s")
    print(f"🔧 自适应参数: 静音阈值 {silence_threshold_db:.1f}dB, 最小静音长度 {min_silence_len_ms}ms")
    
    # 检测静音区间
    silence_ranges_ms = detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_threshold_db
    )
    silence_ranges = [(start / 1000, end / 1000) for start, end in silence_ranges_ms]
    print(f"🔍 检测到 {len(silence_ranges)} 个静音区间")

    # ==================== 智能切分逻辑 ====================
    segments = []
    pos = 0.0

    while pos < duration:
        target_end = pos + max_segment_sec
        
        # 如果剩余时长不足，直接结束
        if target_end >= duration:
            segments.append((pos, duration))
            break
        
        # 定义搜索窗口
        window_start = max(pos + min_segment_sec, target_end - win_sec)
        window_end = min(duration, target_end + win_sec)
        
        # 策略 1: 寻找最优静音点
        split_at = _find_best_silence_point(
            silence_ranges, window_start, window_end, target_end
        )
        
        # 策略 2: 如果没有静音点，使用能量最小点
        if split_at is None:
            print(f"  ⚠️ 在 {window_start:.1f}s-{window_end:.1f}s 范围内未找到静音点，使用能量最低点")
            split_at = _find_energy_minimum_point(audio, window_start, window_end)
        
        # 确保切分点有效
        if split_at is None or split_at <= pos:
            split_at = target_end
        
        # 确保不超过音频时长
        split_at = min(split_at, duration)
        
        segments.append((pos, split_at))
        pos = split_at

    print(f"✂️ 切分为 {len(segments)} 个片段")

    # ==================== 写入文件 ====================
    import time
    mapping = []
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        params = wf.getparams()
        for idx, (start, end) in enumerate(segments):
            for attempt in range(3):
                try:
                    wf.setpos(int(start * sr))
                    frames_to_copy = int((end - start) * sr)
                    audio_bytes = wf.readframes(frames_to_copy)

                    seg_name = f"segment_{idx:03}.wav"
                    seg_path = out_dir / seg_name
                    with contextlib.closing(wave.open(str(seg_path), "wb")) as sw:
                        sw.setparams(params)
                        sw.writeframes(audio_bytes)

                    seg_duration = end - start
                    mapping.append({
                        "file": seg_name, 
                        "start": round(start, 3), 
                        "end": round(end, 3),
                        "duration": round(seg_duration, 3)
                    })
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        raise RuntimeError(f"写入 segment_{idx:03} 失败: {e}")

    (out_dir / "segments_map.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    return mapping


# ---------------------------------------------------------------------
# 🛠️  Helper: export segments to WAV files + mapping JSON
# ---------------------------------------------------------------------
def export_segments(audio_file: str, segments: List[tuple[float, float]], out_dir: str | Path):
    """
    Write each (start, end) slice as segment_000.wav … and save segments_map.json.

    Returns the mapping list for convenience.
    """
    from pydub import AudioSegment

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(audio_file)
    mapping = []
    for idx, (start_s, end_s) in enumerate(segments):
        seg = audio[int(start_s * 1000) : int(end_s * 1000)]
        name = f"segment_{idx:03d}.wav"
        seg.export(out_dir / name, format="wav")
        mapping.append(
            {"file": name, "start": round(start_s, 3), "end": round(end_s, 3)}
        )
    (out_dir / "segments_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2)
    )
    return mapping


# -------------- CLI ---------------------------------------------------------

# ---------------------------------------------------------------------
# 🎛️  CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to .wav file to split")
    args = parser.parse_args()

    from pathlib import Path
    audio_path = Path(args.audio).expanduser().resolve()
    target_len = 600
    win = 60
    out_dir = WORKDIR / "chunks"

    segs = split_audio(audio_path, max_segment_sec=target_len, win_sec=win, out_dir=out_dir)
    # 下面的 mapping 和 export_segments 调用是多余的，因为 split_audio 内部已经完成了文件写入
    # 但为保持原始结构，我们暂时保留它。
    # mapping = export_segments(audio_path, [(seg["start"], seg["end"]) for seg in segs], out_dir)
    print(f"✅ Done. {len(segs)} segments written to: {out_dir}")
