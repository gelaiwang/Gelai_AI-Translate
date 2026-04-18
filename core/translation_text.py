from __future__ import annotations

import re
from pathlib import Path

import srt  # type: ignore


DEFAULT_CONTEXT_BLOCK = (
    "* **视频主题**: `[根据字幕内容提炼主题、场景与主要讨论对象；若信息不足，不要臆造背景。]`\n"
    "* **演讲者风格**: `[优先从字幕中判断语气、密度、节奏与表达习惯；保持专业、准确、自然。]`\n"
    "* **目标观众**: `[面向希望快速理解原视频核心信息的中文观众；术语要统一，表达要清晰。]`\n"
)

ERROR_HINT_MAX_HISTORY = 3


def preprocess_lines_with_numbers(lines: list[str]) -> list[str]:
    numbered_lines = []
    for i, line in enumerate(lines):
        numbered_lines.append(f"{i+1}: {line}")
    return numbered_lines


def create_sliding_window_chunks(
    lines: list[str],
    window_size: int,
    overlap: int,
) -> list[tuple[int, int, list[str]]]:
    if window_size <= 0:
        window_size = 40
    if overlap < 0 or overlap >= window_size:
        overlap = 0

    step = window_size - overlap
    chunks = []
    start = 0
    total = len(lines)

    while start < total:
        end = min(start + window_size, total)
        chunks.append((start, end, lines[start:end]))
        if end >= total:
            break
        start += step

    return chunks


def clean_llm_output(text: str) -> str:
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    return text.strip()


def extract_lines_from_output(text: str) -> list[str]:
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+\s*[:\.]\s*(.*)", line)
        if match:
            line = match.group(1).strip()
        lines.append(line)
    return lines


def validate_line_count(lines: list[str], expected_count: int) -> tuple[bool, str]:
    if len(lines) == expected_count:
        return True, ""
    return False, f"行数不匹配: 期望 {expected_count}, 实际 {len(lines)}"


def merge_overlapping_translations(
    chunk_results: list[tuple[int, int, list[str]]],
    total_lines: int,
) -> dict[int, str]:
    final_result = {}
    line_sources = {}

    for chunk_idx, (start, end, lines) in enumerate(chunk_results):
        chunk_len = end - start
        safe_len = min(len(lines), chunk_len)

        for i in range(safe_len):
            content = lines[i]
            global_idx = start + i + 1

            if chunk_len > 1:
                centrality = 1.0 - abs(i - chunk_len / 2) / (chunk_len / 2)
            else:
                centrality = 1.0

            if global_idx not in line_sources or centrality > line_sources[global_idx][1]:
                line_sources[global_idx] = (chunk_idx, centrality)
                final_result[global_idx] = content

    return final_result


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def subtitle_to_lines(subtitle: srt.Subtitle) -> list[str]:
    lines = subtitle.content.splitlines()
    if not lines:
        return [""]
    return lines


def prepare_lines_for_batch(subtitles: list[srt.Subtitle]) -> tuple[list[str], list[int]]:
    flattened_lines: list[str] = []
    line_counts: list[int] = []
    for subtitle in subtitles:
        lines = subtitle_to_lines(subtitle)
        line_counts.append(len(lines))
        flattened_lines.extend(lines)
    return flattened_lines, line_counts


def chunk_subtitles_by_line_limit(
    subtitles: list[srt.Subtitle],
    max_lines_per_batch: int,
) -> list[list[srt.Subtitle]]:
    if max_lines_per_batch <= 0:
        max_lines_per_batch = 1

    batches: list[list[srt.Subtitle]] = []
    current_batch: list[srt.Subtitle] = []
    current_line_total = 0

    for subtitle in subtitles:
        line_count = len(subtitle_to_lines(subtitle))
        if current_batch and current_line_total + line_count > max_lines_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_line_total = 0

        current_batch.append(subtitle)
        current_line_total += line_count

        if current_line_total >= max_lines_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_line_total = 0

    if current_batch:
        batches.append(current_batch)

    return batches


def reconstruct_subtitles_from_lines(
    original_batch: list[srt.Subtitle],
    translated_text: str,
    line_counts: list[int],
) -> list[srt.Subtitle]:
    normalized_text = translated_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    translated_lines = normalized_text.split("\n")

    expected_total = sum(line_counts)
    while len(translated_lines) > expected_total and translated_lines and not any(translated_lines[expected_total:]):
        translated_lines = translated_lines[:expected_total]

    if len(translated_lines) != expected_total:
        raise ValueError(f"Translated line count mismatch. Expected {expected_total}, got {len(translated_lines)}.")

    rebuilt_subtitles: list[srt.Subtitle] = []
    cursor = 0
    for subtitle, count in zip(original_batch, line_counts):
        current_lines = translated_lines[cursor : cursor + count]
        cursor += count
        cleaned_lines = [line.strip() for line in current_lines]
        rebuilt_subtitles.append(
            srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content="\n".join(cleaned_lines),
            )
        )
    return rebuilt_subtitles


def write_plain_txt(subtitles: list[srt.Subtitle], txt_path: Path) -> None:
    with open(txt_path, "w", encoding="utf-8", newline="\n") as fp:
        for sub in subtitles:
            clean = re.sub(r"[\r\n]+", " ", sub.content).strip()
            if clean:
                fp.write(clean + "\n")


def sanitize_error_message(error: Exception | str) -> str:
    text = str(error).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > 280:
        text = text[:277] + "..."
    return text


def augment_prompt_with_hints(base_prompt: str, hints: list[str] | None) -> str:
    if not hints:
        return base_prompt
    trimmed = [hint for hint in hints if hint]
    if not trimmed:
        return base_prompt
    trimmed = trimmed[-ERROR_HINT_MAX_HISTORY:]
    seen: list[str] = []
    for hint in trimmed:
        if hint not in seen:
            seen.append(hint)
    if not seen:
        return base_prompt
    hint_lines = "\n".join(f"- {hint}" for hint in seen)
    return (
        f"{base_prompt}\n\n注意：上一轮翻译失败，出现以下问题，请保持原有输出格式并避免再次发生：\n{hint_lines}"
    )


def record_batch_error_hint(storage: dict[int, list[str]], batch_index: int, error: Exception | str) -> None:
    hint = sanitize_error_message(error)
    if not hint:
        return
    history = storage.setdefault(batch_index, [])
    if history and history[-1] == hint:
        return
    history.append(hint)
    if len(history) > ERROR_HINT_MAX_HISTORY * 2:
        del history[:-ERROR_HINT_MAX_HISTORY]
