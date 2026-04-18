from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

import srt  # type: ignore
import yaml


SERIES_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("lecture", re.compile(r"\blecture[-\s]*(\d+)\b", re.IGNORECASE)),
    ("episode", re.compile(r"\bepisode[-\s]*(\d+)\b", re.IGNORECASE)),
    ("ep", re.compile(r"\bep[-\s]*(\d+)\b", re.IGNORECASE)),
    ("part", re.compile(r"\bpart[-\s]*(\d+)\b", re.IGNORECASE)),
    ("video", re.compile(r"\bvideo[-\s]*(\d+)(?:[a-z])?\b", re.IGNORECASE)),
    ("year", re.compile(r"\b((?:19|20)\d{2})\b")),
]
SERIES_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "series_registry.yaml"


def normalize_publish_date(raw_value: object, ts_value: object) -> str:
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


def detect_series_from_title(title: str) -> tuple[bool, str, Optional[int]]:
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
    return detect_series_from_title(title)


def load_project_info_metadata(project_dir: Optional[Path]) -> tuple[dict, Optional[Path]]:
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


def normalize_series_key(value: str) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"[’']", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def build_series_key_and_hint(
    title: str,
    channel: str,
    marker: str,
    data: dict,
    project_dir: Optional[Path] = None,
) -> tuple[str, str]:
    playlist_title = str(data.get("playlist_title") or data.get("playlist") or "").strip()
    if playlist_title:
        return normalize_series_key(playlist_title), playlist_title

    for _, pattern in SERIES_PATTERNS:
        match = pattern.search(title or "")
        if not match:
            continue
        prefix = (title or "")[: match.start()].strip(" -|:()[]")
        suffix = (title or "")[match.end() :].strip(" -|:()[]")
        if prefix and len(prefix.split()) >= 2:
            return normalize_series_key(prefix), prefix
        if suffix and len(suffix.split()) >= 2 and marker != "lecture":
            return normalize_series_key(suffix), suffix
        break

    if project_dir and project_dir.parent:
        parent_hint = project_dir.parent.name.strip()
        if parent_hint and not re.fullmatch(r"\d{8}", parent_hint):
            normalized_parent = parent_hint.replace("_", " ").replace("-", " ").strip()
            if len(normalized_parent.split()) >= 2:
                return normalize_series_key(normalized_parent), normalized_parent

    fallback_hint = f"{channel} {marker}".strip() or marker or "series"
    return normalize_series_key(fallback_hint), fallback_hint


def load_series_registry() -> dict:
    if not SERIES_REGISTRY_PATH.exists():
        return {}
    try:
        raw = yaml.safe_load(SERIES_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_series_registry(registry: dict) -> None:
    SERIES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERIES_REGISTRY_PATH.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def extract_series_name_from_response(text: str, fallback_hint: str) -> str:
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


def looks_too_english_series_name(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate:
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", candidate))
    ascii_count = len(re.findall(r"[A-Za-z]", candidate))
    return cjk_count == 0 and ascii_count >= 4


def resolve_series_name(
    title: str,
    channel: str,
    marker: str,
    data: dict,
    generate_series_name_text: Callable[[str], str],
    column_hint: str = "",
    project_dir: Optional[Path] = None,
) -> tuple[str, str]:
    series_key, source_hint = build_series_key_and_hint(title, channel, marker, data, project_dir)
    if not series_key:
        return "", ""

    registry = load_series_registry()
    existing = registry.get(series_key)
    if isinstance(existing, dict):
        series_name = str(existing.get("series_name") or "").strip()
        if series_name and not looks_too_english_series_name(series_name):
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
        generated_text = generate_series_name_text(prompt)
        series_name = extract_series_name_from_response(generated_text, source_hint)
        if looks_too_english_series_name(series_name):
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
            generated_text = generate_series_name_text(retry_prompt)
            series_name = extract_series_name_from_response(generated_text, source_hint)
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
    save_series_registry(registry)
    return series_key, series_name


def ensure_series_registry_entry(
    *,
    title: str,
    marker: str,
    generate_series_name_text: Callable[[str], str],
    channel: str = "",
    playlist_title: str = "",
    column_hint: str = "",
) -> tuple[str, str]:
    return resolve_series_name(
        title=title,
        channel=channel,
        marker=marker,
        data={"playlist_title": playlist_title},
        generate_series_name_text=generate_series_name_text,
        column_hint=column_hint,
        project_dir=None,
    )


def build_prompt_metadata_block(
    project_dir: Optional[Path],
    generate_series_name_text: Callable[[str], str],
) -> str:
    if not project_dir or not project_dir.exists():
        return ""

    data, candidate = load_project_info_metadata(project_dir)
    if data and candidate:
        publish_date = normalize_publish_date(
            data.get("upload_date") or data.get("release_date"),
            data.get("upload_date_timestamp") or data.get("timestamp"),
        )
        channel = str(data.get("channel") or data.get("uploader") or "").strip()
        title = str(data.get("title") or "").strip()
        is_series, series_marker, episode_number = detect_series_from_title(title)
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
            _, series_name = resolve_series_name(title, channel, series_marker, data, generate_series_name_text, column_hint, project_dir)
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


def prepare_context_payload(subtitles: list[srt.Subtitle], max_chars: int = 12000) -> str:
    composed = srt.compose(subtitles).replace("\r\n", "\n").strip()
    if len(composed) <= max_chars:
        return composed
    half = max_chars // 2
    return f"{composed[:half]}\n...\n{composed[-half:]}"


def validate_context_text(text: str) -> bool:
    if not text.strip():
        return False
    required_markers = ["【翻译风格基准", "【微型术语表"]
    return all(marker in text for marker in required_markers)


def read_translation_context_file(context_path: Path) -> str:
    if not context_path.exists():
        return ""
    try:
        text = context_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text if validate_context_text(text) else ""


def resolve_translation_context_text(
    subtitles: list[srt.Subtitle],
    project_dir: Optional[Path],
    *,
    file_name: str,
    force_regenerate: bool,
    enabled: bool,
    default_context: str,
    source_max_chars: int,
    template_content: str,
    escape_braces: Callable[[str], str],
    generate_translation_context_text: Callable[[str], str],
    generate_series_name_text: Callable[[str], str],
    print_info: Callable[[str], None],
    print_warning: Callable[[str], None],
) -> str:
    context_path = None
    if project_dir is not None:
        context_path = project_dir / file_name

    if context_path and context_path.exists() and not force_regenerate:
        existing = read_translation_context_file(context_path)
        if existing:
            print_info(f"检测到可复用的翻译上下文文件: {context_path.name}")
            return existing
        print_warning(f"现有 {context_path.name} 无效，将忽略并回退到配置逻辑。")

    if not enabled:
        return default_context

    metadata_block = build_prompt_metadata_block(project_dir, generate_series_name_text).strip() or "无额外平台元数据。"
    english_excerpt = prepare_context_payload(subtitles, max_chars=source_max_chars)
    prompt = template_content.format(
        metadata_block=escape_braces(metadata_block),
        english_srt_excerpt=escape_braces(english_excerpt),
    )

    try:
        generated = generate_translation_context_text(prompt)
    except Exception as exc:
        print_warning(f"生成 translation_context 失败，将回退到默认上下文：{exc}")
        return default_context

    if not validate_context_text(generated):
        print_warning("生成的 translation_context 不符合预期结构，将回退到默认上下文。")
        return default_context

    if context_path is not None:
        try:
            context_path.write_text(generated.strip() + "\n", encoding="utf-8")
            print_info(f"已生成翻译上下文文件: {context_path.name}")
        except OSError as exc:
            print_warning(f"写入 translation_context 文件失败，但将继续使用本次生成内容：{exc}")

    return generated
