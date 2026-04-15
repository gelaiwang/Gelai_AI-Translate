# _4_gpt_segment_translate.py

import re
import srt # type: ignore
import json
import time
import os
from datetime import timedelta
from pathlib import Path
import argparse
import logging
import shutil
import glob
from typing import Optional
import yaml

import config

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

# [修改] 导入 OpenAI 客户端
from openai import OpenAI

# --- Configuration ---
try:
    # 统一使用 config 包配置
    from config import (
        WORKDIR,
        TRANSLATION_SERVICE,
        GROK_API_KEY, GROK_MODEL_NAME,
        GEMINI_API_KEY, GEMINI_MODEL_NAME,
        VERTEX_MODEL_NAME, VERTEX_PROJECT,
        LOCAL_API_BASE_URL, LOCAL_API_KEY, LOCAL_MODEL_NAME,
        LOCAL_TIMEOUT, LOCAL_CONTEXT_LENGTH, LOCAL_SHOW_THINKING, LOCAL_TOP_P,
        LOCAL_ENABLE_TWO_STAGE, LOCAL_WINDOW_SIZE, LOCAL_OVERLAP,
        LOCAL_STAGE1_TEMPERATURE, LOCAL_STAGE2_TEMPERATURE,
        DEEPSEEK_API_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL_NAME,
        LLM_TEMPERATURE_TRANSLATE_CLOUD, LLM_TEMPERATURE_TRANSLATE_LOCAL,
        PROMPT_TEMPLATE_FILE,
        TRANSLATION_CONTEXT_TEMPLATE_FILE,
        TRANSLATION_CONTEXT_ENABLED,
        TRANSLATION_CONTEXT_FORCE_REGENERATE,
        TRANSLATION_CONTEXT_FILE_NAME,
        TRANSLATION_CONTEXT_SOURCE_MAX_CHARS,
    )
except ImportError:
    rprint(Panel("[bold red]错误: `config/settings.py` 未找到或缺少必要的配置变量。[/bold red]", title="配置错误", border_style="red"))
    exit(1)

from core.google_llm import (
    build_google_text_client,
    generate_google_text,
    is_google_service,
)

# --- Enhanced Configuration Validation (完整替换) ---
if not WORKDIR or not isinstance(WORKDIR, Path):
    rprint(Panel("[bold red]错误: 配置中的 `WORKDIR` 未正确配置或不是有效的 Path 对象。[/bold red]", title="配置错误", border_style="red"))
    exit(1)

SUPPORTED_SERVICES = ["grok", "gemini", "vertex", "local", "deepseek"]
if TRANSLATION_SERVICE not in SUPPORTED_SERVICES:
    rprint(Panel(f"[bold red]错误: 配置中的 `TRANSLATION_SERVICE` ('{TRANSLATION_SERVICE}') 无效. 请选择 {SUPPORTED_SERVICES}。[/bold red]", title="配置错误", border_style="red"))
    exit(1)

# 验证云端温度
if not isinstance(LLM_TEMPERATURE_TRANSLATE_CLOUD, (float, int)) or not (0 <= LLM_TEMPERATURE_TRANSLATE_CLOUD <= 2):
    LLM_TEMPERATURE_TRANSLATE_CLOUD = 0.5
    rprint(Panel(f"[yellow]警告: 配置中的 `LLM_TEMPERATURE_TRANSLATE_CLOUD` 未正确配置或超出范围. 已重置为默认值: {LLM_TEMPERATURE_TRANSLATE_CLOUD}[/yellow]", title="配置警告", border_style="yellow"))

# 验证本地温度
if not isinstance(LLM_TEMPERATURE_TRANSLATE_LOCAL, (float, int)) or not (0 <= LLM_TEMPERATURE_TRANSLATE_LOCAL <= 2):
    LLM_TEMPERATURE_TRANSLATE_LOCAL = 0.5
    rprint(Panel(f"[yellow]警告: 配置中的 `LLM_TEMPERATURE_TRANSLATE_LOCAL` 未正确配置或超出范围. 已重置为默认值: {LLM_TEMPERATURE_TRANSLATE_LOCAL}[/yellow]", title="配置警告", border_style="yellow"))

ACTIVE_API_KEY = None
ACTIVE_MODEL_NAME = None
SERVICE_DISPLAY_NAME = ""
ACTIVE_TEMPERATURE = LLM_TEMPERATURE_TRANSLATE_CLOUD  # 默认云端温度

grok_config_ok = GROK_API_KEY and GROK_MODEL_NAME
gemini_config_ok = GEMINI_API_KEY and GEMINI_MODEL_NAME
vertex_config_ok = VERTEX_PROJECT and VERTEX_MODEL_NAME
local_config_ok = LOCAL_API_BASE_URL and LOCAL_MODEL_NAME
deepseek_config_ok = DEEPSEEK_API_KEY and DEEPSEEK_MODEL_NAME

if TRANSLATION_SERVICE == "grok":
    SERVICE_DISPLAY_NAME = "Grok"
    if not grok_config_ok:
        rprint(Panel("[bold red]错误: 当 TRANSLATION_SERVICE 为 'grok' 时，`GROK_API_KEY` 或 `GROK_MODEL_NAME` 未设置。[/bold red]", title="配置错误", border_style="red"))
        exit(1)
    ACTIVE_API_KEY = GROK_API_KEY
    ACTIVE_MODEL_NAME = GROK_MODEL_NAME
elif TRANSLATION_SERVICE == "gemini":
    SERVICE_DISPLAY_NAME = "Gemini"
    if not gemini_config_ok:
        rprint(Panel("[bold red]错误: 当 TRANSLATION_SERVICE 为 'gemini' 时，`GEMINI_API_KEY` 或 `GEMINI_MODEL_NAME` 未设置。[/bold red]", title="配置错误", border_style="red"))
        exit(1)
    ACTIVE_API_KEY = GEMINI_API_KEY
    ACTIVE_MODEL_NAME = GEMINI_MODEL_NAME
    if not grok_config_ok:
        rprint(Panel("[yellow]警告: 未找到有效的 Grok 配置。Gemini 翻译失败时的 Grok 兜底功能将不可用。[/yellow]", title="配置警告", border_style="yellow"))
elif TRANSLATION_SERVICE == "vertex":
    SERVICE_DISPLAY_NAME = "Google Vertex AI"
    if not vertex_config_ok:
        rprint(Panel("[bold red]错误: 当 TRANSLATION_SERVICE 为 'vertex' 时，`GOOGLE_CLOUD_PROJECT` 或 `VERTEX_MODEL_NAME` 未设置。[/bold red]", title="配置错误", border_style="red"))
        exit(1)
    ACTIVE_API_KEY = VERTEX_PROJECT
    ACTIVE_MODEL_NAME = VERTEX_MODEL_NAME
    if not grok_config_ok:
        rprint(Panel("[yellow]警告: 未找到有效的 Grok 配置。Vertex 翻译失败时的 Grok 兜底功能将不可用。[/yellow]", title="配置警告", border_style="yellow"))
elif TRANSLATION_SERVICE == "local":
    SERVICE_DISPLAY_NAME = "Local LLM"
    if not local_config_ok:
        rprint(Panel("[bold red]错误: 当 TRANSLATION_SERVICE 为 'local' 时，`LOCAL_API_BASE_URL` 或 `LOCAL_MODEL_NAME` 未设置。[/bold red]", title="配置错误", border_style="red"))
        exit(1)
    ACTIVE_API_KEY = LOCAL_API_KEY
    ACTIVE_MODEL_NAME = LOCAL_MODEL_NAME
    ACTIVE_TEMPERATURE = LLM_TEMPERATURE_TRANSLATE_LOCAL  # 本地温度
elif TRANSLATION_SERVICE == "deepseek":
    SERVICE_DISPLAY_NAME = "DeepSeek"
    if not deepseek_config_ok:
        rprint(Panel("[bold red]错误: 当 TRANSLATION_SERVICE 为 'deepseek' 时，`DEEPSEEK_API_KEY` 或 `DEEPSEEK_MODEL_NAME` 未设置。[/bold red]", title="配置错误", border_style="red"))
        exit(1)
    ACTIVE_API_KEY = DEEPSEEK_API_KEY
    ACTIVE_MODEL_NAME = DEEPSEEK_MODEL_NAME
    if not grok_config_ok:
        rprint(Panel("[yellow]警告: 未找到有效的 Grok 配置。DeepSeek 翻译失败时的 Grok 兜底功能将不可用。[/yellow]", title="配置警告", border_style="yellow"))


LOG_FILE_PATH = WORKDIR / "translation_log.txt"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    filemode='a'
)
logger = logging.getLogger(__name__)

def load_prompt_template(file_path: Path, required_placeholders: Optional[list[str]] = None) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        for placeholder in required_placeholders or []:
            if placeholder not in template_content:
                rprint(Panel(f"[bold red]错误: 提示模板文件 '{file_path}'\n缺少必要的占位符 '{placeholder}'。[/bold red]", title="提示模板错误", border_style="red"))
                exit(1)
        return template_content
    except FileNotFoundError:
        rprint(Panel(f"[bold red]错误: 提示模板文件 '{file_path}' 未找到。[/bold red]", title="文件未找到", border_style="red"))
        exit(1)
    except Exception as e:
        rprint(Panel(f"[bold red]错误: 读取提示模板文件 '{file_path}' 时发生错误: {e}[/bold red]", title="文件读取错误", border_style="red"))
        exit(1)

prompt_template_content = load_prompt_template(
    PROMPT_TEMPLATE_FILE,
    required_placeholders=["{srt_content_for_llm}", "{video_context}"],
)
translation_context_template_content = load_prompt_template(
    TRANSLATION_CONTEXT_TEMPLATE_FILE,
    required_placeholders=["{metadata_block}", "{english_srt_excerpt}"],
)

# --- Two-Stage Local Translation Prompts ---
def _load_local_stage_prompts() -> tuple[str, str]:
    """Load stage1 and stage2 prompt templates for local two-stage translation."""
    base_dir = Path(__file__).resolve().parent.parent / "config" / "prompts"
    stage1_path = base_dir / "translate_local_stage1.txt"
    stage2_path = base_dir / "translate_local_stage2.txt"
    
    stage1_content = ""
    stage2_content = ""
    
    if stage1_path.exists():
        stage1_content = stage1_path.read_text(encoding="utf-8")
    if stage2_path.exists():
        stage2_content = stage2_path.read_text(encoding="utf-8")
    
    return stage1_content, stage2_content

LOCAL_STAGE1_PROMPT, LOCAL_STAGE2_PROMPT = _load_local_stage_prompts()

# --- Two-Stage Translation Helper Functions ---
# 使用行号模式替代标签模式，更简单可靠

def _preprocess_lines_with_numbers(lines: list[str]) -> list[str]:
    """为每行添加行号 1: 原文，便于LLM理解顺序。"""
    numbered_lines = []
    for i, line in enumerate(lines):
        numbered_lines.append(f"{i+1}: {line}")
    return numbered_lines

def _create_sliding_window_chunks(
    lines: list[str],
    window_size: int,
    overlap: int,
) -> list[tuple[int, int, list[str]]]:
    """
    滑动窗口分块，返回 (start_idx, end_idx, chunk_lines) 列表。
    - window_size: 每个 chunk 的行数
    - overlap: 重叠行数，步进 = window_size - overlap
    """
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

def _clean_llm_output(text: str) -> str:
    """清理LLM输出中的常见问题字符。"""
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    return text.strip()


def _normalize_publish_date(raw_value: object, ts_value: object) -> str:
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if re.fullmatch(r"\d{8}", cleaned):
            return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
            return cleaned
    if isinstance(ts_value, (int, float)):
        try:
            return time.strftime("%Y-%m-%d", time.gmtime(float(ts_value)))
        except (OverflowError, ValueError):
            return ""
    return ""


SERIES_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("lecture", re.compile(r"\blecture[-\s]*(\d+)\b", re.IGNORECASE)),
    ("episode", re.compile(r"\bepisode[-\s]*(\d+)\b", re.IGNORECASE)),
    ("ep", re.compile(r"\bep[-\s]*(\d+)\b", re.IGNORECASE)),
    ("part", re.compile(r"\bpart[-\s]*(\d+)\b", re.IGNORECASE)),
    ("video", re.compile(r"\bvideo[-\s]*(\d+)(?:[a-z])?\b", re.IGNORECASE)),
    ("year", re.compile(r"\b((?:19|20)\d{2})\b")),
]
SERIES_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "series_registry.yaml"


def _detect_series_from_title(title: str) -> tuple[bool, str, Optional[int]]:
    cleaned = (title or "").strip()
    if not cleaned:
        return False, "", None
    for marker, pattern in SERIES_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        try:
            episode_number = int(match.group(1))
        except (TypeError, ValueError):
            episode_number = None
        return True, marker, episode_number
    return False, "", None


def detect_series_marker(title: str) -> tuple[bool, str, Optional[int]]:
    """Public wrapper for lightweight series detection in pre-download stages."""
    return _detect_series_from_title(title)


def _load_project_info_metadata(project_dir: Optional[Path]) -> tuple[dict, Optional[Path]]:
    if not project_dir or not project_dir.exists():
        return {}, None

    candidates = sorted(
        list(project_dir.glob("*.info.json")) + list(project_dir.glob("*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data, candidate
    return {}, None


def _normalize_series_key(value: str) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"[’']", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _build_series_key_and_hint(
    title: str,
    channel: str,
    marker: str,
    data: dict,
    project_dir: Optional[Path] = None,
) -> tuple[str, str]:
    playlist_title = str(data.get("playlist_title") or data.get("playlist") or "").strip()
    if playlist_title:
        return _normalize_series_key(playlist_title), playlist_title

    for _, pattern in SERIES_PATTERNS:
        match = pattern.search(title or "")
        if not match:
            continue
        prefix = (title or "")[: match.start()].strip(" -|:()[]")
        suffix = (title or "")[match.end() :].strip(" -|:()[]")
        if prefix and len(prefix.split()) >= 2:
            return _normalize_series_key(prefix), prefix
        if suffix and len(suffix.split()) >= 2 and marker != "lecture":
            return _normalize_series_key(suffix), suffix
        break

    if project_dir and project_dir.parent:
        parent_hint = project_dir.parent.name.strip()
        if parent_hint and not re.fullmatch(r"\d{8}", parent_hint):
            normalized_parent = parent_hint.replace("_", " ").replace("-", " ").strip()
            if len(normalized_parent.split()) >= 2:
                return _normalize_series_key(normalized_parent), normalized_parent

    fallback_hint = f"{channel} {marker}".strip() or marker or "series"
    return _normalize_series_key(fallback_hint), fallback_hint


def _load_series_registry() -> dict:
    if not SERIES_REGISTRY_PATH.exists():
        return {}
    try:
        raw = yaml.safe_load(SERIES_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_series_registry(registry: dict) -> None:
    SERIES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERIES_REGISTRY_PATH.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def _generate_series_name_text(prompt: str) -> str:
    client_obj = _build_primary_llm_client(ACTIVE_MODEL_NAME)
    if client_obj is None:
        raise RuntimeError("系列命名 LLM 客户端未初始化。")

    temperature = min(0.3, max(float(ACTIVE_TEMPERATURE), 0.1))

    if TRANSLATION_SERVICE == "local":
        response = client_obj.chat.completions.create(
            model=ACTIVE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=LOCAL_TOP_P,
            extra_body={"think": not LOCAL_SHOW_THINKING, "num_ctx": LOCAL_CONTEXT_LENGTH},
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif TRANSLATION_SERVICE in {"grok", "deepseek"}:
        response = client_obj.chat.completions.create(
            model=ACTIVE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif is_google_service(TRANSLATION_SERVICE):
        text = generate_google_text(client_obj, prompt, temperature)
    else:
        raise RuntimeError(f"不支持的服务类型: {TRANSLATION_SERVICE}")
    return text.strip()


def _extract_series_name_from_response(text: str, fallback_hint: str) -> str:
    lines = [line.strip(" -\t") for line in (text or "").splitlines() if line.strip()]
    for line in lines:
        if "：" in line:
            _, value = line.split("：", 1)
            candidate = value.strip()
        elif ":" in line:
            _, value = line.split(":", 1)
            candidate = value.strip()
        else:
            candidate = line.strip()
        if candidate:
            candidate = re.sub(r"\s+", " ", candidate)
            candidate = re.sub(r"\bEP\d+\b", "", candidate, flags=re.IGNORECASE).strip(" -:")
            if candidate:
                return candidate
    return fallback_hint.strip() or "系列"


def _looks_too_english_series_name(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate:
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", candidate))
    ascii_count = len(re.findall(r"[A-Za-z]", candidate))
    return cjk_count == 0 and ascii_count >= 4


def _resolve_series_name(
    title: str,
    channel: str,
    marker: str,
    data: dict,
    column_hint: str = "",
    project_dir: Optional[Path] = None,
) -> tuple[str, str]:
    series_key, source_hint = _build_series_key_and_hint(title, channel, marker, data, project_dir)
    if not series_key:
        return "", ""

    registry = _load_series_registry()
    existing = registry.get(series_key)
    if isinstance(existing, dict):
        series_name = str(existing.get("series_name") or "").strip()
        if series_name and not _looks_too_english_series_name(series_name):
            return series_key, series_name

    prompt = (
        "你是内容团队的系列命名助手。请为一个长期视频系列生成稳定、简短、适合中文用户理解的系列名称。\n"
        "要求：\n"
        "1. 只输出系列名称本身，不要解释。\n"
        "2. 中文优先，长度尽量控制在 3 到 8 个字。\n"
        "3. 要像长期专栏名，不要把这一集的具体主题当作系列名。\n"
        "4. 不要输出 EP、期数、年份、括号说明。\n"
        "5. 同一系列后续视频会复用这个名称，所以必须稳定、通用。\n\n"
        "6. 如果线索是英文课程名或英文系列名，不要原样照抄英文，优先转成自然中文名称。\n"
        "7. 允许保留少量常见品牌缩写，如 YC、FSD、OpenAI，但整体仍应是中文专栏名。\n\n"
        "示例：\n"
        "- How to Start a Startup -> YC创业课\n"
        "- Berkshire Hathaway Shareholder Letters -> 伯克希尔股东信\n"
        "- Charlie Munger Archive -> 芒格精读\n\n"
        f"栏目参考：{column_hint or '待定'}\n"
        f"系列线索：{source_hint}\n"
        f"发布频道：{channel or '未知'}\n"
        f"原标题：{title or '未知'}\n\n"
        "系列名称："
    )

    try:
        generated_text = _generate_series_name_text(prompt)
        series_name = _extract_series_name_from_response(generated_text, source_hint)
        if _looks_too_english_series_name(series_name):
            retry_prompt = (
                "请把下面这个英文系列线索改写成自然、稳定、适合中文用户理解的系列名称。\n"
                "要求：\n"
                "1. 只输出中文系列名称，不要解释。\n"
                "2. 优先像中文专栏名，而不是直译整句英文。\n"
                "3. 可以保留少量常见缩写，如 YC、FSD、OpenAI。\n"
                "4. 不要输出英文原句，不要输出 EP、年份、括号。\n\n"
                "示例：\n"
                "- How to Start a Startup -> YC创业课\n"
                "- Berkshire Hathaway Shareholder Letters -> 伯克希尔股东信\n"
                "- Charlie Munger Archive -> 芒格精读\n\n"
                f"栏目参考：{column_hint or '待定'}\n"
                f"英文系列线索：{source_hint}\n"
                f"发布频道：{channel or '未知'}\n"
                f"原标题：{title or '未知'}\n\n"
                "中文系列名称："
            )
            generated_text = _generate_series_name_text(retry_prompt)
            series_name = _extract_series_name_from_response(generated_text, source_hint)
    except Exception:
        series_name = source_hint.strip() or "系列"

    now_str = time.strftime("%Y-%m-%d", time.localtime())
    registry[series_key] = {
        "series_name": series_name,
        "column": column_hint or "",
        "source_hint": source_hint,
        "created_at": now_str,
        "updated_at": now_str,
    }
    _save_series_registry(registry)
    return series_key, series_name


def ensure_series_registry_entry(
    *,
    title: str,
    channel: str = "",
    marker: str,
    playlist_title: str = "",
    column_hint: str = "",
) -> tuple[str, str]:
    """Resolve and persist a series name from lightweight playlist metadata."""
    return _resolve_series_name(
        title=title,
        channel=channel,
        marker=marker,
        data={"playlist_title": playlist_title},
        column_hint=column_hint,
        project_dir=None,
    )


def _build_prompt_metadata_block(project_dir: Optional[Path]) -> str:
    if not project_dir or not project_dir.exists():
        return ""

    data, candidate = _load_project_info_metadata(project_dir)
    if data and candidate:
        publish_date = _normalize_publish_date(
            data.get("upload_date") or data.get("release_date"),
            data.get("upload_date_timestamp") or data.get("timestamp"),
        )
        channel = str(data.get("channel") or data.get("uploader") or "").strip()
        title = str(data.get("title") or "").strip()
        is_series, series_marker, episode_number = _detect_series_from_title(title)
        column_hint = ""
        lowered = f"{title} {channel}".lower()
        if any(token in lowered for token in [
            "buffett", "dalio", "ray dalio", "bridgewater", "soros", "george soros",
            "howard marks", "ackman", "bill ackman", "fund manager", "hedge fund",
            "asset allocation", "valuation", "earnings", "portfolio", "macro",
            "market", "market update", "stocks", "equity", "bonds", "commodities", "fx"
        ]):
            column_hint = "投资市场观察"
        elif "fsd" in lowered:
            column_hint = "特斯拉与自动驾驶观察"
        elif any(token in lowered for token in ["munger", "berkshire", "shareholder", "yc", "startup", "charlie munger"]):
            column_hint = "商业经典精选"
        elif any(token in lowered for token in ["buffett", "jobs", "bezos", "cook", "musk", "ma yun", "lei jun", "ren zhengfei"]):
            column_hint = "商业巨头访谈"
        elif any(token in lowered for token in ["nvidia", "tsmc", "capex", "data center", "gpu", "semiconductor", "supply chain"]):
            column_hint = "产业趋势解读"
        elif any(token in lowered for token in ["altman", "hassabis", "a16z", "openai", "anthropic"]):
            column_hint = "AI前沿访谈"

        lines = [
            "【已知平台元数据（高优先级事实，优先于字幕推测）】",
            f"- 元数据来源：yt-dlp info.json（{candidate.name}）",
        ]
        if publish_date:
            lines.append(f"- 视频发布时间：{publish_date}")
            lines.append(
                "- 若标题或简介需要出现年份/日期，必须以这个发布时间为准；禁止改写成 2024、2025 等其他年份。"
            )
        else:
            lines.append("- 未解析到可靠发布时间；若无充分证据，请不要在标题中编造具体年份。")
        if channel:
            lines.append(f"- 发布频道：{channel}")
        if title:
            lines.append(f"- 平台原标题：{title}")
        if is_series:
            _, series_name = _resolve_series_name(title, channel, series_marker, data, column_hint, project_dir)
            lines.append("- 系列判定：是（由标题正则规则直接判定，不允许改判）")
            lines.append(f"- 系列标记：{series_marker}")
            if series_name:
                lines.append(f"- 系列名称：{series_name}（已从系列注册表固定，不允许改写）")
            if episode_number is not None:
                lines.append(f"- 集数：EP{episode_number:02d}")
            lines.append("- 标题生成规则：该视频必须使用系列标题模板 `【栏目】系列名称 EPxx：本集主题`，并严格复用给定系列名称。")
        else:
            lines.append("- 系列判定：否（由标题正则规则直接判定，不允许改判）")
            lines.append("- 标题生成规则：该视频使用普通单条内容标题模板，优先具体信息、明确问题或清晰观点，不强制使用因果判断句。")
        return "\n".join(lines)

    return ""

def _extract_lines_from_output(text: str) -> list[str]:
    """从纯文本输出中提取行。"""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line: continue
        # 移除可能存在的行号前缀 (如 "1: " 或 "1. ")
        match = re.match(r"^\d+\s*[:\.]\s*(.*)", line)
        if match:
            line = match.group(1).strip()
        lines.append(line)
    return lines

def _validate_line_count(lines: list[str], expected_count: int) -> tuple[bool, str]:
    if len(lines) == expected_count:
        return True, ""
    return False, f"行数不匹配: 期望 {expected_count}, 实际 {len(lines)}"

def _merge_overlapping_translations(
    chunk_results: list[tuple[int, int, list[str]]],
    total_lines: int,
) -> dict[int, str]:
    """
    合并重叠区域的翻译结果。
    """
    final_result = {}
    line_sources = {}  # index -> (chunk_idx, position_score)
    
    for chunk_idx, (start, end, lines) in enumerate(chunk_results):
        chunk_len = end - start
        # Ensure we don't go out of bounds if lines count mismatch (though validation should catch this)
        safe_len = min(len(lines), chunk_len)
        
        for i in range(safe_len):
            content = lines[i]
            global_idx = start + i + 1  # 1-based index
            
            # Centrality logic
            if chunk_len > 1:
                centrality = 1.0 - abs(i - chunk_len / 2) / (chunk_len / 2)
            else:
                centrality = 1.0
            
            if global_idx not in line_sources or centrality > line_sources[global_idx][1]:
                line_sources[global_idx] = (chunk_idx, centrality)
                final_result[global_idx] = content
                
    return final_result


DEFAULT_CONTEXT_BLOCK = (
    "* **视频主题**: `[系列内容聚焦全球顶级企业家、投资人与思想领袖的深度对谈，围绕企业战略、资本市场、科技创新、宏观趋势、风险管理与长期主义，提炼可复制的方法论与实践经验。]`\n"
    "* **演讲者风格**: `[主持人 Nicolai Tangen 提问直接、节奏干净，善于用数据与案例追问“怎么做”；兼具专业与幽默感，促使嘉宾落到可操作细节。]`\n"
    "* **目标观众**: `[投资者、企业管理者与创业者、金融与科技从业者、MBA/商学院学生，以及对全球商业与市场感兴趣的普通观众。]`\n"
)

def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

def _subtitle_to_lines(subtitle: srt.Subtitle) -> list[str]:
    """
    Split a subtitle's content into individual text lines (keeping order).
    Always returns at least one line so we can round-trip empty subtitles.
    """
    lines = subtitle.content.splitlines()
    if not lines:
        return [""]
    return lines

def _prepare_lines_for_batch(subtitles: list[srt.Subtitle]) -> tuple[list[str], list[int]]:
    """
    Flatten subtitles into a list of text lines for translation and remember
    how many lines belong to each subtitle so that we can reconstruct them.
    """
    flattened_lines: list[str] = []
    line_counts: list[int] = []
    for subtitle in subtitles:
        lines = _subtitle_to_lines(subtitle)
        line_counts.append(len(lines))
        flattened_lines.extend(lines)
    return flattened_lines, line_counts

def _chunk_subtitles_by_line_limit(
    subtitles: list[srt.Subtitle],
    max_lines_per_batch: int,
) -> list[list[srt.Subtitle]]:
    """
    Group subtitle objects into batches so that the total number of lines
    in each batch does not exceed `max_lines_per_batch` (except when a single
    subtitle already exceeds the limit, in which case it forms its own batch).
    """
    if max_lines_per_batch <= 0:
        max_lines_per_batch = 1

    batches: list[list[srt.Subtitle]] = []
    current_batch: list[srt.Subtitle] = []
    current_line_total = 0

    for subtitle in subtitles:
        line_count = len(_subtitle_to_lines(subtitle))
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

def _reconstruct_subtitles_from_lines(
    original_batch: list[srt.Subtitle],
    translated_text: str,
    line_counts: list[int],
) -> list[srt.Subtitle]:
    """
    Rebuild subtitles for the batch using translated plain-text lines.
    Validates line counts to ensure 1:1 mapping with the originals.
    """
    normalized_text = translated_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    translated_lines = normalized_text.split("\n")

    # Drop trailing empty lines if they exceed the expected total (common when the LLM appends a newline).
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
        # Keep intentional blank lines, but trim leading/trailing spaces that often appear in model outputs.
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


def _write_plain_txt(subtitles: list[srt.Subtitle], txt_path: Path) -> None:
    """
    Export subtitles to plain text (one merged line per subtitle).
    """
    with open(txt_path, "w", encoding="utf-8", newline="\n") as fp:
        for sub in subtitles:
            clean = re.sub(r"[\r\n]+", " ", sub.content).strip()
            if clean:
                fp.write(clean + "\n")

ERROR_HINT_MAX_HISTORY = 3

def _sanitize_error_message(error: Exception | str) -> str:
    text = str(error).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > 280:
        text = text[:277] + "..."
    return text

def _augment_prompt_with_hints(base_prompt: str, hints: list[str] | None) -> str:
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

def _record_batch_error_hint(storage: dict[int, list[str]], batch_index: int, error: Exception | str) -> None:
    hint = _sanitize_error_message(error)
    if not hint:
        return
    history = storage.setdefault(batch_index, [])
    if history and history[-1] == hint:
        return
    history.append(hint)
    if len(history) > ERROR_HINT_MAX_HISTORY * 2:
        del history[:-ERROR_HINT_MAX_HISTORY]

def _prepare_context_payload(subtitles: list[srt.Subtitle], max_chars: int = 12000) -> str:
    composed = srt.compose(subtitles).replace("\r\n", "\n").strip()
    if len(composed) <= max_chars:
        return composed
    half = max_chars // 2
    return f"{composed[:half]}\n...\n{composed[-half:]}"

def _validate_context_text(text: str) -> bool:
    if not text.strip():
        return False
    required_markers = ["【翻译风格基准", "【微型术语表"]
    return all(marker in text for marker in required_markers)


def _read_translation_context_file(context_path: Path) -> str:
    if not context_path.exists():
        return ""
    try:
        text = context_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text if _validate_context_text(text) else ""


def _generate_translation_context_text(prompt: str) -> str:
    client_obj = _build_primary_llm_client(ACTIVE_MODEL_NAME)
    if client_obj is None:
        raise RuntimeError("翻译上下文 LLM 客户端未初始化。")

    temperature = min(0.4, max(float(ACTIVE_TEMPERATURE), 0.1))

    if TRANSLATION_SERVICE == "local":
        response = client_obj.chat.completions.create(
            model=ACTIVE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=LOCAL_TOP_P,
            extra_body={"think": not LOCAL_SHOW_THINKING, "num_ctx": LOCAL_CONTEXT_LENGTH},
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif TRANSLATION_SERVICE in {"grok", "deepseek"}:
        response = client_obj.chat.completions.create(
            model=ACTIVE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif is_google_service(TRANSLATION_SERVICE):
        text = generate_google_text(client_obj, prompt, temperature)
    else:
        raise RuntimeError(f"不支持的服务类型: {TRANSLATION_SERVICE}")
    return text.strip()


def resolve_translation_context_text(
    subtitles: list[srt.Subtitle],
    project_dir: Optional[Path],
) -> str:
    context_path = None
    if project_dir is not None:
        context_path = project_dir / TRANSLATION_CONTEXT_FILE_NAME

    if context_path and context_path.exists() and not TRANSLATION_CONTEXT_FORCE_REGENERATE:
        existing = _read_translation_context_file(context_path)
        if existing:
            rprint(f"[green]检测到可复用的翻译上下文文件: {context_path.name}[/green]")
            return existing
        rprint(f"[yellow]现有 {context_path.name} 无效，将忽略并回退到配置逻辑。[/yellow]")

    if not TRANSLATION_CONTEXT_ENABLED:
        return DEFAULT_CONTEXT_BLOCK

    metadata_block = _build_prompt_metadata_block(project_dir).strip() or "无额外平台元数据。"
    english_excerpt = _prepare_context_payload(
        subtitles,
        max_chars=TRANSLATION_CONTEXT_SOURCE_MAX_CHARS,
    )
    prompt = translation_context_template_content.format(
        metadata_block=_escape_braces(metadata_block),
        english_srt_excerpt=_escape_braces(english_excerpt),
    )

    try:
        generated = _generate_translation_context_text(prompt)
    except Exception as exc:
        rprint(f"[yellow]生成 translation_context 失败，将回退到默认上下文：{exc}[/yellow]")
        return DEFAULT_CONTEXT_BLOCK

    if not _validate_context_text(generated):
        rprint("[yellow]生成的 translation_context 不符合预期结构，将回退到默认上下文。[/yellow]")
        return DEFAULT_CONTEXT_BLOCK

    if context_path is not None:
        try:
            context_path.write_text(generated.strip() + "\n", encoding="utf-8")
            rprint(f"[green]已生成翻译上下文文件: {context_path.name}[/green]")
        except OSError as exc:
            rprint(f"[yellow]写入 translation_context 文件失败，但将继续使用本次生成内容：{exc}[/yellow]")

    return generated

def _build_primary_llm_client(model_name: Optional[str] = None) -> object:
    target_model_name = model_name or ACTIVE_MODEL_NAME
    if not target_model_name:
        raise RuntimeError('LLM 模型名称未配置')
    if TRANSLATION_SERVICE != "vertex" and not ACTIVE_API_KEY:
        raise RuntimeError('LLM API key 未配置')
    if TRANSLATION_SERVICE == 'grok':
        return OpenAI(api_key=ACTIVE_API_KEY, base_url='https://api.x.ai/v1')
    if is_google_service(TRANSLATION_SERVICE):
        return build_google_text_client(TRANSLATION_SERVICE, target_model_name)
    if TRANSLATION_SERVICE == 'local':
        return OpenAI(base_url=LOCAL_API_BASE_URL, api_key=ACTIVE_API_KEY, timeout=LOCAL_TIMEOUT)
    if TRANSLATION_SERVICE == 'deepseek':
        return OpenAI(api_key=ACTIVE_API_KEY, base_url=DEEPSEEK_API_BASE_URL)
    raise RuntimeError(f'Unsupported translation service: {TRANSLATION_SERVICE}')

# --- Two-Stage Local Translation Main Function ---
def _translate_local_two_stage(
    english_srt_objects: list[srt.Subtitle],
    input_stem: str,
    raw_output_dir: Path,
    video_context_text: str,
) -> tuple[list[srt.Subtitle], list[dict], list[tuple[int, list[srt.Subtitle]]]]:
    """
    本地两阶段翻译流程 (无标签纯行模式)：
    1. 预处理：为每行添加行号
    2. 滑动窗口分块
    3. Stage1：保真翻译
    4. 校验 + 重试
    5. Stage2：口语润色
    6. 校验 + 合并
    7. 重建字幕
    """
    if not english_srt_objects:
        return [], [], []
    
    rprint(Panel(
        f"启用两阶段翻译模式 (window={LOCAL_WINDOW_SIZE}, overlap={LOCAL_OVERLAP})",
        title="Local LLM 两阶段翻译 (纯行模式)",
        border_style="cyan"
    ))
    
    # 初始化客户端
    try:
        llm_client = OpenAI(
            base_url=LOCAL_API_BASE_URL,
            api_key=LOCAL_API_KEY,
            timeout=LOCAL_TIMEOUT
        )
    except Exception as e:
        rprint(f"[bold red]初始化 Local LLM 客户端失败: {e}[/bold red]")
        return [], [], []
    
    # Step 0: 展平字幕为行列表
    all_lines: list[str] = []
    line_to_subtitle: list[tuple[int, int]] = []  # (subtitle_idx, line_idx_in_subtitle)
    
    for sub_idx, sub in enumerate(english_srt_objects):
        sub_lines = sub.content.splitlines() or [""]
        for line_idx, line in enumerate(sub_lines):
            all_lines.append(line)
            line_to_subtitle.append((sub_idx, line_idx))
    
    total_lines = len(all_lines)
    rprint(f"[cyan]共 {len(english_srt_objects)} 条字幕，展平为 {total_lines} 行[/cyan]")
    
    # Step 1: 添加行号 (Global numbering)
    numbered_lines = _preprocess_lines_with_numbers(all_lines)
    
    # Step 2: 滑动窗口分块
    chunks = _create_sliding_window_chunks(
        numbered_lines,
        LOCAL_WINDOW_SIZE,
        LOCAL_OVERLAP
    )
    rprint(f"[cyan]分为 {len(chunks)} 个翻译块[/cyan]")
    
    # Prepare context
    context_block = video_context_text.strip() or DEFAULT_CONTEXT_BLOCK
    escaped_context = _escape_braces(context_block)
    
    # Step 3 & 4: Stage1 翻译 + 校验
    raw_llm_responses: list[dict] = []
    chunk_results: list[tuple[int, int, list[str]]] = []
    failed_chunks: list[int] = []
    
    MAX_RETRIES = 3
    
    for chunk_idx, (start, end, chunk_lines) in enumerate(chunks):
        expected_count = end - start
        chunk_text = "\n".join(chunk_lines)
        
        rprint(f"🔄 Stage1 翻译块 #{chunk_idx + 1}/{len(chunks)} (行 {start+1}-{end})...")
        
        stage1_success = False
        stage1_lines = []
        
        for attempt in range(MAX_RETRIES):
            try:
                # Build Stage1 prompt
                if LOCAL_STAGE1_PROMPT:
                    prompt = LOCAL_STAGE1_PROMPT.format(
                        video_context=escaped_context,
                        srt_content_for_llm=chunk_text,
                    )
                else:
                    prompt = f"翻译以下英文（保持行数）：\n{chunk_text}"
                
                response = llm_client.chat.completions.create(
                    model=LOCAL_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LOCAL_STAGE1_TEMPERATURE,
                    top_p=LOCAL_TOP_P,
                    extra_body={
                        "think": not LOCAL_SHOW_THINKING,
                        "num_ctx": LOCAL_CONTEXT_LENGTH,
                    }
                )
                
                raw_output = response.choices[0].message.content.strip() if response.choices else ""
                cleaned_output = _clean_llm_output(raw_output)
                
                # Save raw output
                backup_path = raw_output_dir / f"{input_stem}_chunk_{chunk_idx+1:03d}_stage1_attempt{attempt+1}.txt"
                backup_path.write_text(cleaned_output, encoding="utf-8")
                raw_llm_responses.append({
                    "chunk": chunk_idx + 1,
                    "stage": 1,
                    "attempt": attempt + 1,
                    "file": str(backup_path)
                })
                
                # Validate
                extracted_lines = _extract_lines_from_output(cleaned_output)
                success, msg = _validate_line_count(extracted_lines, expected_count)
                
                if success:
                    stage1_lines = extracted_lines
                    stage1_success = True
                    rprint(f"  ✅ Stage1 块 #{chunk_idx + 1} 成功")
                    break
                else:
                    rprint(f"  [yellow]⚠️ Stage1 块 #{chunk_idx + 1} 校验失败 (尝试 {attempt+1}): {msg}[/yellow]")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                        
            except Exception as e:
                rprint(f"  [yellow]⚠️ Stage1 块 #{chunk_idx + 1} 异常 (尝试 {attempt+1}): {e}[/yellow]")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
        
        if not stage1_success:
            rprint(f"  [red]❌ Stage1 块 #{chunk_idx + 1} 最终失败，使用可用结果[/red]")
            # Use whatever we extracted, even if count mismatch
            stage1_lines = extracted_lines if 'extracted_lines' in locals() else []
            if len(stage1_lines) != expected_count:
                failed_chunks.append(chunk_idx)
                # Pad or truncate to match expected count
                if len(stage1_lines) < expected_count:
                    stage1_lines.extend([""] * (expected_count - len(stage1_lines)))
                else:
                    stage1_lines = stage1_lines[:expected_count]
        
        # Step 5: Stage2 润色
        stage2_lines = stage1_lines
        
        if LOCAL_STAGE2_PROMPT and stage1_success:
            rprint(f"🔄 Stage2 润色块 #{chunk_idx + 1}/{len(chunks)}...")
            
            # Re-number for Stage2 (Local numbering 1..N)
            stage2_input = []
            for i, line in enumerate(stage1_lines):
                stage2_input.append(f"{i+1}: {line}")
            stage2_input_text = "\n".join(stage2_input)
            
            for attempt in range(MAX_RETRIES):
                try:
                    prompt = LOCAL_STAGE2_PROMPT.format(
                        srt_content_for_llm=stage2_input_text,
                    )
                    
                    response = llm_client.chat.completions.create(
                        model=LOCAL_MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=LOCAL_STAGE2_TEMPERATURE,
                        top_p=LOCAL_TOP_P,
                        extra_body={
                            "think": not LOCAL_SHOW_THINKING,
                            "num_ctx": LOCAL_CONTEXT_LENGTH,
                        }
                    )
                    
                    raw_output = response.choices[0].message.content.strip() if response.choices else ""
                    cleaned_output = _clean_llm_output(raw_output)
                    
                    # Save raw output
                    backup_path = raw_output_dir / f"{input_stem}_chunk_{chunk_idx+1:03d}_stage2_attempt{attempt+1}.txt"
                    backup_path.write_text(cleaned_output, encoding="utf-8")
                    raw_llm_responses.append({
                        "chunk": chunk_idx + 1,
                        "stage": 2,
                        "attempt": attempt + 1,
                        "file": str(backup_path)
                    })
                     
                    extracted_lines = _extract_lines_from_output(cleaned_output)
                    success, msg = _validate_line_count(extracted_lines, expected_count)
                    
                    if success:
                        stage2_lines = extracted_lines
                        rprint(f"  ✅ Stage2 块 #{chunk_idx + 1} 成功")
                        break
                    else:
                        rprint(f"  [yellow]⚠️ Stage2 块 #{chunk_idx + 1} 校验失败: {msg}，使用 Stage1 结果[/yellow]")
                        break  # Use stage1 result
                        
                except Exception as e:
                    rprint(f"  [yellow]⚠️ Stage2 块 #{chunk_idx + 1} 异常: {e}，使用 Stage1 结果[/yellow]")
                    break
            
        chunk_results.append((start, end, stage2_lines))
        time.sleep(0.5)  # Rate limiting
    
    # Step 6: 合并重叠区域
    rprint("[cyan]合并翻译结果...[/cyan]")
    merged_translations = _merge_overlapping_translations(chunk_results, total_lines)
    
    # Step 7: 重建字幕
    rprint("[cyan]重建字幕结构...[/cyan]")
    rebuilt_subtitles: list[srt.Subtitle] = []
    
    # Group lines back to subtitles
    subtitle_contents: dict[int, list[str]] = {}
    for line_idx, (sub_idx, _) in enumerate(line_to_subtitle):
        tag_num = line_idx + 1
        content = merged_translations.get(tag_num, "")
        if sub_idx not in subtitle_contents:
            subtitle_contents[sub_idx] = []
        subtitle_contents[sub_idx].append(content)
    
    for sub_idx, sub in enumerate(english_srt_objects):
        lines = subtitle_contents.get(sub_idx, [""])
        rebuilt_subtitles.append(srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content="\n".join(lines),
        ))
    
    # 重新编号
    rebuilt_subtitles = [
        srt.Subtitle(index=idx, start=sub.start, end=sub.end, content=sub.content)
        for idx, sub in enumerate(rebuilt_subtitles, start=1)
    ]
    
    # Report
    success_count = len(rebuilt_subtitles)
    fail_info = [(i, []) for i in failed_chunks]  # Empty list as placeholder
    rprint(f"[green]✅ 两阶段翻译完成，成功 {success_count} 条字幕[/green]")
    
    return rebuilt_subtitles, raw_llm_responses, fail_info

def translate_srt_via_llm(
    english_srt_objects: list[srt.Subtitle],
    input_stem: str,
    batch_size: int,
    raw_output_dir: Path,
    video_context_text: str,
) -> tuple[list[srt.Subtitle], list[dict], list[tuple[int, list[srt.Subtitle]]]]:
    if not english_srt_objects:
        return [], [], []

    # 本地服务且启用两阶段翻译时，使用新流程
    if TRANSLATION_SERVICE == "local" and LOCAL_ENABLE_TWO_STAGE:
        return _translate_local_two_stage(
            english_srt_objects=english_srt_objects,
            input_stem=input_stem,
            raw_output_dir=raw_output_dir,
            video_context_text=video_context_text,
        )

    # 原有流程（云端服务或本地单阶段）
    context_block = video_context_text.strip() or DEFAULT_CONTEXT_BLOCK
    escaped_context_block = _escape_braces(context_block)

    try:
        llm_client = _build_primary_llm_client()
    except Exception as e:  # noqa: BLE001
        rprint(f"[bold red]初始化 {SERVICE_DISPLAY_NAME} 客户端失败: {e}[/bold red]")
        return [], [], []

    gemini_api_retry_model_name = "gemini-3-flash-preview"
    client_cache: dict[tuple[str, str], object] = {}
    client_cache[(TRANSLATION_SERVICE, ACTIVE_MODEL_NAME)] = llm_client

    def _build_client_for_provider(service_name: str, model_name: str) -> object:
        if service_name == "grok":
            if not GROK_API_KEY:
                raise RuntimeError("Grok API key 未配置")
            return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
        if service_name == "gemini":
            if not GEMINI_API_KEY:
                raise RuntimeError("Gemini API key 未配置")
            return build_google_text_client("gemini", model_name)
        if service_name == "vertex":
            if not VERTEX_PROJECT:
                raise RuntimeError("Vertex project 未配置")
            return build_google_text_client("vertex", model_name)
        if service_name == "local":
            if not LOCAL_API_BASE_URL:
                raise RuntimeError("Local API base URL 未配置")
            return OpenAI(base_url=LOCAL_API_BASE_URL, api_key=LOCAL_API_KEY, timeout=LOCAL_TIMEOUT)
        if service_name == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("DeepSeek API key 未配置")
            return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE_URL)
        raise RuntimeError(f"Unsupported translation service: {service_name}")

    def _resolve_client_for_provider(service_name: str, model_name: str):
        key = (service_name, model_name)
        if key not in client_cache:
            client_cache[key] = _build_client_for_provider(service_name, model_name)
        return client_cache[key]

    provider_plan: list[tuple[str, str, str]]
    max_retry_rounds: int
    if TRANSLATION_SERVICE == "vertex" and gemini_config_ok:
        provider_plan = [
            ("vertex", ACTIVE_MODEL_NAME, SERVICE_DISPLAY_NAME),
            ("gemini", gemini_api_retry_model_name, "Gemini API"),
        ]
        max_retry_rounds = 2
    else:
        provider_plan = [
            (TRANSLATION_SERVICE, ACTIVE_MODEL_NAME, SERVICE_DISPLAY_NAME),
        ]
        max_retry_rounds = 3

    def _sanitize_model_for_filename(model_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)

    def _short_stem_for_filename(stem: str, max_len: int = 48) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "batch"
        if len(sanitized) <= max_len:
            return sanitized
        return sanitized[:max_len].rstrip("._-") or "batch"

    successful_batches: dict[int, list[srt.Subtitle]] = {}
    raw_llm_responses: list[dict] = []
    batch_error_hints: dict[int, list[str]] = {}
    batch_line_cache: dict[int, tuple[list[str], list[int]]] = {}
    
    rprint(f"[cyan]...扫描目录 '{raw_output_dir.name}' 检查已完成的翻译批次...[/cyan]")
    existing_batch_files = glob.glob(str(raw_output_dir / f"{input_stem}_batch_*.srt"))
    loaded_count = 0
    for f_path in existing_batch_files:
        match = re.search(r"_batch_(\d+)", Path(f_path).name)
        if match:
            batch_index = int(match.group(1)) - 1
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    parsed_subs = list(srt.parse(content))
                    successful_batches[batch_index] = parsed_subs
                    loaded_count += 1
            except Exception as e:
                rprint(f"[yellow]警告: 加载已有批次文件 '{Path(f_path).name}' 失败: {e}，将重新翻译此批次。[/yellow]")

    if loaded_count > 0:
        rprint(f"[green]✅ 成功加载 {loaded_count} 个已存在的翻译批次。[/green]")
    else:
        rprint("[cyan]...未找到已完成的翻译批次，将从头开始。[/cyan]")

    max_lines_per_batch = max(1, batch_size)
    subtitle_batches = _chunk_subtitles_by_line_limit(english_srt_objects, max_lines_per_batch)
    all_batches: list[tuple[int, list[srt.Subtitle]]] = [
        (i, batch) for i, batch in enumerate(subtitle_batches)
    ]

    for idx, subtitles in all_batches:
        batch_line_cache[idx] = _prepare_lines_for_batch(subtitles)
    
    all_batches_to_process = [batch for batch in all_batches if batch[0] not in successful_batches]

    MAX_ATTEMPTS_PER_BATCH = 3

    if not all_batches_to_process:
        rprint(Panel(f"所有 {len(all_batches)} 个批次均已在之前的运行中翻译完成。直接进入整合步骤。", title=f"{SERVICE_DISPLAY_NAME} 翻译任务", border_style="green"))
    else:
        rprint(Panel(f"总共 {len(all_batches)} 个批次（单批不超过 {max_lines_per_batch} 行英文），需要处理 {len(all_batches_to_process)} 个新批次，送往 {SERVICE_DISPLAY_NAME} ({ACTIVE_MODEL_NAME}) 进行翻译...", title=f"{SERVICE_DISPLAY_NAME} 翻译任务", border_style="blue"))

    current_round = 1
    while current_round <= max_retry_rounds and all_batches_to_process:
        if current_round > 1:
            rprint(
                Panel(
                    f"开始第 {current_round}/{max_retry_rounds} 轮重试，处理剩余 {len(all_batches_to_process)} 个失败批次...",
                    title="重试流程",
                    style="bold yellow",
                )
            )

        round_failed_batches = all_batches_to_process

        for provider_service, provider_model_name, provider_display_name in provider_plan:
            if not round_failed_batches:
                break

            current_model_label = _sanitize_model_for_filename(provider_model_name)
            current_service_label = _sanitize_model_for_filename(provider_service)
            failed_in_provider: list[tuple[int, list[srt.Subtitle]]] = []

            for batch_index, batch_to_translate in round_failed_batches:
                if not batch_to_translate:
                    continue

                batch_lines, line_counts = batch_line_cache.get(batch_index, _prepare_lines_for_batch(batch_to_translate))
                batch_line_cache[batch_index] = (batch_lines, line_counts)
                lines_payload = "\n".join(batch_lines)
                escaped_batch_text = _escape_braces(lines_payload)
                base_prompt = prompt_template_content.format(
                    srt_content_for_llm=escaped_batch_text,
                    video_context=escaped_context_block,
                )
                expected_line_count = len(batch_lines)

                rprint(
                    f"🌐 正在翻译批次 #{batch_index + 1} (字幕 {len(batch_to_translate)} 条 / 英文行 {len(batch_lines)}) "
                    f"使用 {provider_display_name} / {provider_model_name}..."
                )

                batch_successfully_processed = False
                resource_exhausted = False
                for attempt in range(MAX_ATTEMPTS_PER_BATCH):
                    try:
                        current_client = _resolve_client_for_provider(provider_service, provider_model_name)
                        current_prompt = _augment_prompt_with_hints(
                            base_prompt, batch_error_hints.get(batch_index)
                        )
                        translated_plain_text = ""
                        if provider_service == "local":
                            if not current_client:
                                raise Exception(f"{provider_display_name} LLM client is not initialized.")
                            response = current_client.chat.completions.create(
                                model=provider_model_name,
                                messages=[{"role": "user", "content": current_prompt}],
                                temperature=ACTIVE_TEMPERATURE,
                                top_p=LOCAL_TOP_P,
                                extra_body={
                                    "think": not LOCAL_SHOW_THINKING,
                                    "num_ctx": LOCAL_CONTEXT_LENGTH,
                                }
                            )
                            translated_plain_text = response.choices[0].message.content.strip() if response.choices else ""
                        elif provider_service in {"grok", "deepseek"}:
                            if not current_client:
                                raise Exception(f"{provider_display_name} LLM client is not initialized.")
                            response = current_client.chat.completions.create(
                                model=provider_model_name,
                                messages=[{"role": "user", "content": current_prompt}],
                                temperature=ACTIVE_TEMPERATURE,
                            )
                            translated_plain_text = response.choices[0].message.content.strip() if response.choices else ""
                        elif is_google_service(provider_service):
                            if not current_client:
                                raise Exception(f"{provider_display_name} LLM client is not initialized.")
                            translated_plain_text = generate_google_text(current_client, current_prompt, ACTIVE_TEMPERATURE)

                        short_input_stem = _short_stem_for_filename(input_stem)
                        backup_path = raw_output_dir / (
                            f"{short_input_stem}_b{batch_index+1:03d}_{current_service_label}_{current_model_label}_"
                            f"r{current_round}_a{attempt+1}.txt"
                        )
                        with open(backup_path, "w", encoding="utf-8") as f:
                            f.write(translated_plain_text)
                        raw_llm_responses.append(
                            {
                                "batch_index": batch_index,
                                "attempt": attempt + 1,
                                "service": provider_service,
                                "model": provider_model_name,
                                "raw_text_file": str(backup_path),
                            }
                        )

                        if not translated_plain_text or not translated_plain_text.strip():
                            raise ValueError("LLM returned empty content.")
                        if expected_line_count == 0:
                            raise ValueError("No lines detected in batch payload.")

                        current_batch_translated_subs = _reconstruct_subtitles_from_lines(
                            batch_to_translate,
                            translated_plain_text,
                            line_counts,
                        )

                        successful_batches[batch_index] = current_batch_translated_subs
                        batch_srt_path = raw_output_dir / f"{input_stem}_batch_{batch_index+1:03d}.srt"
                        with open(batch_srt_path, "w", encoding="utf-8") as f:
                            f.write(srt.compose(current_batch_translated_subs).replace("\r\n", "\n"))
                        batch_successfully_processed = True
                        batch_error_hints.pop(batch_index, None)
                        rprint(
                            f"✅ 批次 #{batch_index + 1}, 第 {attempt+1} 次尝试"
                            f"({provider_display_name} / {provider_model_name}) 翻译成功！"
                        )
                        time.sleep(1)
                        break

                    except Exception as e:
                        resource_exhausted = "RESOURCE_EXHAUSTED" in str(e)
                        error_msg = (
                            f"批次 #{batch_index + 1}, 第 {attempt+1}/{MAX_ATTEMPTS_PER_BATCH} 次尝试"
                            f"({provider_display_name} / {provider_model_name}) 失败: {e}"
                        )
                        rprint(f"[yellow]⚠️ {error_msg}[/yellow]")
                        _record_batch_error_hint(batch_error_hints, batch_index, e)
                        if resource_exhausted:
                            rprint("[yellow]⚠️ 发现 Vertex 资源耗尽，立即切换后续提供方[/yellow]")
                            resource_exhausted = True
                            short_input_stem = _short_stem_for_filename(input_stem)
                            failed_input_path = raw_output_dir / (
                                f"{short_input_stem}_b{batch_index+1:03d}_input_failed_{current_service_label}.txt"
                            )
                            with open(failed_input_path, "w", encoding="utf-8") as f:
                                f.write(lines_payload)
                            break
                        if attempt < MAX_ATTEMPTS_PER_BATCH - 1:
                            wait_time = 2 + attempt
                            time.sleep(wait_time)
                            rprint(f"    {wait_time}秒后重试...")
                        else:
                            final_error_msg = (
                                f"批次 #{batch_index+1} 在 {MAX_ATTEMPTS_PER_BATCH} 次尝试后仍失败"
                                f"（当前模型: {provider_model_name}）。将其移至下一阶段重试。"
                            )
                            rprint(f"[red]❌ {final_error_msg}[/red]")
                            short_input_stem = _short_stem_for_filename(input_stem)
                            failed_input_path = raw_output_dir / (
                                f"{short_input_stem}_b{batch_index+1:03d}_input_failed_{current_service_label}.txt"
                            )
                            with open(failed_input_path, "w", encoding="utf-8") as f:
                                f.write(lines_payload)

                if not batch_successfully_processed:
                    failed_in_provider.append((batch_index, batch_to_translate))

            round_failed_batches = failed_in_provider

        all_batches_to_process = round_failed_batches
        current_round += 1

    permanently_failed_batches = all_batches_to_process

    if TRANSLATION_SERVICE in {"gemini", "vertex", "deepseek"} and permanently_failed_batches:
        if not grok_config_ok:
            rprint(Panel("[yellow]翻译后仍有失败批次，但 Grok 配置缺失，无法启动兜底程序。[/yellow]", title="Grok 兜底跳过", border_style="yellow"))
        else:
            rprint(Panel(f"...主服务仍失败 {len(permanently_failed_batches)} 个批次，启动 Grok 作为最终兜底...", title="Fallback to Grok", style="bold red"))
            
            try:
                grok_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
                still_failed_after_fallback = []
                MAX_FALLBACK_ATTEMPTS = 2

                for batch_index, batch_to_translate in permanently_failed_batches:
                    rprint(f" G-Fallback 正在尝试翻译批次 #{batch_index + 1} 使用 Grok...")

                    batch_lines, line_counts = batch_line_cache.get(batch_index, _prepare_lines_for_batch(batch_to_translate))
                    batch_line_cache[batch_index] = (batch_lines, line_counts)
                    lines_payload = "\n".join(batch_lines)
                    escaped_batch_text = _escape_braces(lines_payload)
                    base_prompt = prompt_template_content.format(
                        srt_content_for_llm=escaped_batch_text,
                        video_context=escaped_context_block,
                    )

                    fallback_success = False
                    for attempt in range(MAX_FALLBACK_ATTEMPTS):
                        try:
                            current_prompt = _augment_prompt_with_hints(
                                base_prompt, batch_error_hints.get(batch_index)
                            )
                            response = grok_client.chat.completions.create(
                                model=GROK_MODEL_NAME,
                                messages=[{"role": "user", "content": current_prompt}],
                                temperature=LLM_TEMPERATURE_TRANSLATE_CLOUD,
                            )
                            translated_plain_text = response.choices[0].message.content.strip() if response.choices else ""

                            backup_path = raw_output_dir / f"{input_stem}_batch_{batch_index+1:03d}_GrokFallback_attempt_{attempt+1}.txt"
                            with open(backup_path, "w", encoding="utf-8") as f:
                                f.write(translated_plain_text)
                            raw_llm_responses.append({"batch_index": batch_index, "attempt": f"grok_fallback_{attempt+1}", "raw_text_file": str(backup_path)})

                            if not translated_plain_text or not translated_plain_text.strip():
                                raise ValueError("Grok returned empty content.")

                            current_batch_translated_subs = _reconstruct_subtitles_from_lines(
                                batch_to_translate,
                                translated_plain_text,
                                line_counts,
                            )

                            successful_batches[batch_index] = current_batch_translated_subs
                            batch_srt_path = raw_output_dir / f"{input_stem}_batch_{batch_index+1:03d}.srt"
                            with open(batch_srt_path, "w", encoding="utf-8") as f:
                                f.write(srt.compose(current_batch_translated_subs).replace("\r\n", "\n"))
                            fallback_success = True
                            batch_error_hints.pop(batch_index, None)
                            rprint(f"✅ G-Fallback: 批次 #{batch_index + 1} 使用 Grok 翻译成功！")
                            time.sleep(1)
                            break
                        except Exception as e:
                            error_msg = f"G-Fallback: 批次 #{batch_index + 1}, 第 {attempt+1}/{MAX_FALLBACK_ATTEMPTS} 次尝试(Grok) 失败: {e}"
                            rprint(f"[yellow]⚠️ {error_msg}[/yellow]")
                            _record_batch_error_hint(batch_error_hints, batch_index, e)
                            if attempt < MAX_FALLBACK_ATTEMPTS - 1:
                                time.sleep(2)

                    if not fallback_success:
                        still_failed_after_fallback.append((batch_index, batch_to_translate))

                permanently_failed_batches = still_failed_after_fallback

            except Exception as e:
                rprint(f"[bold red]初始化 Grok 兜底客户端失败: {e}。兜底程序中止。[/bold red]")


    rprint("[cyan]...翻译和重试流程结束，开始整合最终结果...[/cyan]")
    final_translated_subs: list[srt.Subtitle] = []
    
    sorted_successful_indices = sorted(successful_batches.keys())
    
    if len(sorted_successful_indices) != len(all_batches):
        if not permanently_failed_batches:
            rprint(f"[bold yellow]⚠️ 警告: 最终成功批次数 ({len(sorted_successful_indices)}) 与总批次数 ({len(all_batches)}) 不符。可能有些批次文件丢失或损坏。[/bold yellow]")

    for index in sorted_successful_indices:
        final_translated_subs.extend(successful_batches[index])
    
    final_translated_subs = [srt.Subtitle(index=idx, start=sub.start, end=sub.end, content=sub.content) for idx, sub in enumerate(final_translated_subs, start=1)]
    
    return final_translated_subs, raw_llm_responses, permanently_failed_batches


def validate_llm_translation(
    original_english_subs: list[srt.Subtitle],
    translated_chinese_subs: list[srt.Subtitle] | None,
    input_stem: str
) -> bool:
    rprint(Panel(f"开始验证翻译结果 (仅中文)...", title="翻译验证", border_style="magenta"))

    if translated_chinese_subs is None:
        error_msg = f"主要错误: 翻译步骤未能成功返回字幕列表 (结果为 None)。验证中止。"
        rprint(f"[bold red]{error_msg}[/bold red]")
        return False

    if not isinstance(translated_chinese_subs, list):
        error_msg = (f"主要错误: 翻译步骤返回的不是列表类型 (类型: {type(translated_chinese_subs)}). 验证中止。")
        rprint(f"[bold red]{error_msg}[/bold red]")
        return False

    overall_valid = True
    issues_found = 0

    if len(original_english_subs) != len(translated_chinese_subs):
        overall_valid = False
        issues_found += 1
        rprint(f"[bold yellow]⚠️ 主要警告: 原始字幕数 ({len(original_english_subs)}) 与最终翻译后字幕数 ({len(translated_chinese_subs)}) 不匹配！这可能是因为有批次永久失败。[/bold yellow]")

    validation_table = Table(title=f"翻译验证详情（仅显示问题项）")
    validation_table.add_column("字幕号 (原)", style="dim")
    validation_table.add_column("问题类型")
    validation_table.add_column("详情")

    original_subs_map = {(sub.start, sub.end): sub for sub in original_english_subs}
    
    for i, translated_sub in enumerate(translated_chinese_subs):
        original_sub = original_subs_map.get((translated_sub.start, translated_sub.end))
        current_sub_issues = []

        if original_sub is None:
            issue = f"时间戳无法匹配到任何原始字幕: 译 {translated_sub.start}-->{translated_sub.end}"
            current_sub_issues.append(issue)
        else:
            content_lines = translated_sub.content.strip().split('\n')
            original_content_lines_count = len(original_sub.content.strip().split('\n'))
            expected_lines = original_content_lines_count if original_content_lines_count > 1 else 1

            if len(content_lines) != expected_lines:
                issue = (f"行数错误: 期望 {expected_lines} 行，得到 {len(content_lines)} 行。 "
                         f"内容: '{translated_sub.content[:80].replace(chr(10), '<NL>')}{'...' if len(translated_sub.content) > 80 else ''}'")
                current_sub_issues.append(issue)
            elif not any(line.strip() for line in content_lines):
                issue = "中文翻译缺失或所有行均为空。"
                current_sub_issues.append(issue)

        if current_sub_issues:
            issues_found += len(current_sub_issues)
            overall_valid = False
            for issue_desc in current_sub_issues:
                 validation_table.add_row(
                     str(original_sub.index if original_sub else f"未知(译{i+1})"),
                     issue_desc.split(":")[0],
                     issue_desc.split(":",1)[1].strip() if ":" in issue_desc else issue_desc
                 )

    if issues_found > 0:
        rprint(validation_table)
        rprint(f"\n[bold yellow]⚠️ 验证发现 {issues_found} 个问题。[/bold yellow]")
    else:
        rprint(f"[bold green]✅ 所有成功翻译的字幕均通过基本格式和内容验证！[/bold green]")
    return overall_valid


def generate_and_translate_srt(input_stem: str, batch_size: int, project_dir: Optional[Path] = None):
    service_display = SERVICE_DISPLAY_NAME or ("Local LLM" if TRANSLATION_SERVICE == "local" else TRANSLATION_SERVICE.title())
    rprint(Panel(f"SRT 处理 → {service_display} 翻译流程开启 for: {input_stem}", title="主流程", style="bold green"))

    if project_dir is None:
        project_dir = WORKDIR / input_stem
    output_dir = project_dir / "output"
    raw_output_dir = output_dir / f"{TRANSLATION_SERVICE}_raw_batches"

    rprint(f"[cyan]步骤 0: 准备项目临时目录 (支持断点续传): {raw_output_dir}[/cyan]")
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    rprint("[green]✅ 临时目录已就绪。[/green]")

    rprint("[cyan]步骤 1: 加载已生成的英文 SRT 文件...[/cyan]")
    english_srt_path = project_dir / f"[EN]-{input_stem}.srt"

    if not english_srt_path.exists():
        rprint(f"[red]失败: 找不到预期的英文 SRT 文件: {english_srt_path}[/red]")
        return

    try:
        with open(english_srt_path, "r", encoding="utf-8") as f:
            english_srt_objects = list(srt.parse(f.read()))
        if not english_srt_objects:
            rprint("[yellow]警告: 英文SRT文件为空，无需翻译。[/yellow]")
            return
        rprint(f"[green]✅ 已从文件 {english_srt_path} 加载 {len(english_srt_objects)} 条英文字幕。[/green]")
    except Exception as e:
        rprint(f"[red]加载英文 SRT 文件 '{english_srt_path}' 失败: {e}[/red]")
        return

    video_context_text = resolve_translation_context_text(
        subtitles=english_srt_objects,
        project_dir=project_dir,
    )

    rprint(Panel(f"步骤 2: 使用 {service_display} 进行翻译...", title="翻译步骤", style="bold blue"))
    
    chinese_srt_objects, raw_llm_responses_data, permanently_failed_batches = translate_srt_via_llm(
        english_srt_objects=english_srt_objects,
        input_stem=input_stem,
        batch_size=batch_size,
        raw_output_dir=raw_output_dir,
        video_context_text=video_context_text,
    )
    if permanently_failed_batches:
        failed_indices = [item[0] for item in permanently_failed_batches]
        error_msg = f"项目 {input_stem} 有 {len(permanently_failed_batches)} 个批次在所有尝试（包括兜底）后永久失败。失败的批次索引: {[i+1 for i in failed_indices]}"
        logger.warning(error_msg)
        rprint(f"[bold red]永久失败警告: {error_msg}[/bold red]")

    if not chinese_srt_objects:
        rprint(f"[bold red]❌ 失败: 经过所有尝试后，未能成功翻译任何字幕。无法生成中文字幕文件。[/bold red]")
        return
    
    rprint(f"[green]✅ 翻译完成，共成功翻译 {len(chinese_srt_objects)} / {len(english_srt_objects)} 条中文字幕。[/green]")

    rprint(Panel(f"步骤 3: 验证翻译结果...", title="验证步骤", style="bold magenta"))
    validation_ok = validate_llm_translation(
        original_english_subs=english_srt_objects,
        translated_chinese_subs=chinese_srt_objects,
        input_stem=input_stem
    )

    if permanently_failed_batches:
        rprint("[bold red]❌ 检测到永久失败批次，拒绝写入不完整的中文字幕文件。[/bold red]")
        return

    if not validation_ok:
        rprint("[bold red]❌ 翻译验证未通过，拒绝写入中文字幕文件。[/bold red]")
        return
    
    chinese_srt_path = project_dir / f"[CN]-{input_stem}.srt"
    chinese_txt_path = project_dir / f"[CN]-{input_stem}.txt"
    
    try:
        composed_content = srt.compose(chinese_srt_objects).replace('\r\n', '\n')
        with open(chinese_srt_path, "w", encoding="utf-8", newline='\n') as f:
            f.write(composed_content)
        _write_plain_txt(chinese_srt_objects, chinese_txt_path)
        rprint(f"[bold green]🎉 最终中文字幕已保存至: {chinese_srt_path} ({len(chinese_srt_objects)} 条字幕)[/bold green]")
        rprint(f"[green]• 中文TXT已写入: {chinese_txt_path}[/green]")
    except Exception as e:
        rprint(f"[red]写入中文字幕文件失败: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description=f"使用 LLM API 将英文 SRT 翻译为中文 SRT。")
    parser.add_argument("--stem", required=True, help="正在处理的文件的基本名称（stem），用于命名SRT文件。")
    parser.add_argument("--batch_size", type=int, default=getattr(config, 'DEFAULT_BATCH_SIZE', 50), help="每次请求允许的英文字幕最大行数。")

    args = parser.parse_args()

    global SERVICE_DISPLAY_NAME
    if TRANSLATION_SERVICE == "grok": SERVICE_DISPLAY_NAME = "Grok"
    elif TRANSLATION_SERVICE == "gemini": SERVICE_DISPLAY_NAME = "Gemini"
    elif TRANSLATION_SERVICE == "vertex": SERVICE_DISPLAY_NAME = "Google Vertex AI"
    elif TRANSLATION_SERVICE == "local": SERVICE_DISPLAY_NAME = "Local LLM"
    
    desc_text = f"使用 {SERVICE_DISPLAY_NAME} API 将英文 SRT 翻译为中文 SRT。"
    if TRANSLATION_SERVICE in {'gemini', 'vertex'} and grok_config_ok:
        desc_text += " 如果主服务失败，将使用 Grok 作为兜底。"
    parser.description = desc_text
    
    try:
        import openai
    except ImportError as e:
        lib_name = str(e).split("'")[1]
        rprint(f"[bold red]错误: 必要的库 `{lib_name}` 未安装。请运行 `pip install {lib_name}` 来安装它。[/bold red]")
        exit(1)

    generate_and_translate_srt(args.stem, args.batch_size)

if __name__ == "__main__":
    main()


