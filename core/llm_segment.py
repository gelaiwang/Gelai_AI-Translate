# _3_2_llm_segment.py

import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, List
import difflib

try:
    import google.generativeai as genai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    genai = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None  # type: ignore

# 断句服务：rule / gemini / grok / deepseek / local
from config import (
    WORKDIR,
    GEMINI_API_KEY,
    LLM_TEMPERATURE_SEGMENTATION,
    GEMINI_MODEL_SEGMENTATION,
    VERTEX_MODEL_SEGMENTATION,
    GROK_API_KEY,
    GROK_MODEL_NAME,
    LOCAL_API_BASE_URL,
    LOCAL_API_KEY,
    LOCAL_MODEL_SEGMENTATION,
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL_SEGMENTATION,
    SEGMENTATION_SERVICE,
    PAUSE_THRESHOLD,
    MIN_SEGMENT_LENGTH,
    MAX_SEGMENT_LENGTH,
)
from core.google_llm import build_google_text_client, generate_google_text
from core.validate_segments import normalize_text

# --- 日志记录设定 ---
LOG_FILE_PATH = WORKDIR / "llm_segment.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
for handler in list(logger.handlers):
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)
        handler.close()
_file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
_file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
logger.addHandler(_file_handler)


# --- 提示词模板 ---
SPLIT_TEXT_PROMPT = """你是一位专业的英文视频字幕编辑专家，专注于自然语言处理和文本切分。你的核心任务是将一段完整的英文文本，智能地拆分成适合屏幕阅读的短句。

【核心原则与优先级】
1.  **专注原文句子的断句（最最高优先级）**: 严禁擅自添加或删减任何内容，自我发挥添加补足删除原文句子内容，专注原文句子的断句。
2.  **语义完整性（最高优先级）**: 任何拆分都不能破坏句子的核心语义结构。必须避免拆开固定短语、专有名词、以及紧密连接的修饰成分和被修饰对象（例如，绝不能将 "The book that I read" 拆分成 "The book" 和 "that I read"）。
3.  **句长限制（高优先级）**: 每个短句的长度严格限制在60个英文字符以内（含空格和标点）。如果为了保证语义完整性确实无法在50字符内断句，长度可略微放宽，但绝不能超过65个字符。
4.  **自然断点（一般优先级）**: 在满足以上两个原则的前提下，优先选择在最自然的位置断句，以提升观众的阅读流畅度。
5.  **绝对不要**将缩写词展开。例如，如果原文是 "it's"，输出必须是 "it's"，绝不能是 "it is"。如果原文是 "they're"，输出必须是 "they're"，绝不能是 "they are"。
    保持所有原始标点符号不变。
    你的输出，当所有分句的文本（"text"字段）拼接在一起时，必须与我提供的原始输入文本在字符级别上完全一致。


【断句规则】
1.  **强制断句**: 遇到以下标点符号时，必须在此处结束当前句子：
    * 句号 `.`
    * 问号 `?`
    * 感叹号 `!`
    * 以及全角对应的 `。` `？` `！`

2.  **优先断点选择**: 当一个句子过长需要被拆分时，请按以下顺序寻找最佳断点：
    * **首选**：在引导独立从句的连词**之前**断句，例如 `and`, `but`, `so`, `or`, `because`。
    * **次选**：在引导从句的词**之前**断句，例如 `which`, `when`, `where`, `who`, `if`, `although`。但请务必遵循【核心原则1】，确保不会破坏语义。
    * **再次**：在能形成完整语义单元的逗号 `,` 处断句。避免在简短的插入语（如 `However,`）后立即断句。

3.  **禁止断句**: 严格禁止在以下情况中进行断句：
    * 固定搭配或短语内部 (e.g., "in order to", "such as", "as well as")。
    * 专有名词中间 (e.g., "New York City", "Dr. Smith")。


【输出格式】
请严格保留原始英文文本及标点符号，不进行拼写修改或内容删减。
输出必须是标准的JSON结构，格式如下：
{{
    "sentences": [
        {{
            "text": "第一个句子",
            "index": 0
        }},
        {{
            "text": "第二个句子",
            "index": 1
        }}
    ]
}}

【示例】
输入文本: "It's a very long sentence that demonstrates how the segmentation should work, because it includes multiple clauses and exceeds the character limit."
期望输出:
{{
    "sentences": [
        {{
            "text": "It's a very long sentence",
            "index": 0
        }},
        {{
            "text": "that demonstrates how the segmentation should work,",
            "index": 1
        }},
        {{
            "text": "because it includes multiple clauses",
            "index": 2
        }},
        {{
            "text": "and exceeds the character limit.",
            "index": 3
        }}
    ]
}}

现在，请处理以下文本内容：
{text}

"""

HARD_END_PUNCTUATION = {".", "?", "!", "\u3002", "\uFF1F", "\uFF01"}
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "rev.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "no.", "nos.", "fig.", "figs.", "approx.", "dept.", "est.", "inc.", "ltd.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    "a.m.", "p.m.", "u.s.", "u.k.", "u.n.", "p.s.", "ph.d.", "m.d.", "al.", "cf.", "ed.", "vol.",
}
CLOSING_PUNCTUATION = {'"', "'", ")", "]", "}", "»", "”", "’", "》", "」", "』", "）", "】"}
OPENING_PUNCTUATION = {'"', "'", "(", "[", "{", "«", "“", "‘", "《", "「", "『", "（", "【"}
ELLIPSIS_MIN_LENGTH = 3


def _extend_with_following_spaces(text: str, index: int) -> int:
    """Consume连续空白，确保保留原始间距。"""
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _get_previous_word(text: str, index: int) -> str:
    """获取 index 之前的单词，用于判定引导词。"""
    i = index - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    end = i + 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    start = i + 1
    return text[start:end]


def _next_non_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _previous_non_space(text: str, index: int) -> int:
    while index >= 0 and text[index].isspace():
        index -= 1
    return index


def _is_decimal_point(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    left = _previous_non_space(text, index - 1)
    right = _next_non_space(text, index + 1)
    if left == -1 or right >= len(text):
        return False
    left_char = text[left]
    right_char = text[right]
    if not left_char.isdigit():
        return False
    if right_char.isdigit():
        return True
    if right_char in {",", "/"}:
        lookahead = _next_non_space(text, right + 1)
        return lookahead < len(text) and text[lookahead].isdigit()
    return False


def _ellipsis_span_end(text: str, index: int) -> int:
    if text[index] != ".":
        return index + 1
    j = index
    dot_count = 0
    while j < len(text) and text[j] == ".":
        dot_count += 1
        j += 1
    if dot_count >= ELLIPSIS_MIN_LENGTH:
        return j
    return index + 1


def _extract_token_for_abbreviation(text: str, index: int) -> str:
    end = index + 1
    start = index
    while start >= 0 and text[start] in CLOSING_PUNCTUATION:
        start -= 1
    end = start + 1
    while start >= 0 and not text[start].isspace():
        start -= 1
    token = text[start + 1:end]
    token = token.strip()
    return token.lstrip("".join(OPENING_PUNCTUATION))


def _is_known_abbreviation(text: str, index: int) -> bool:
    token = _extract_token_for_abbreviation(text, index).lower()
    if not token:
        return False
    if token in ABBREVIATIONS:
        return True
    if token.endswith(".") and token[:-1] in ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[a-z]\.){2,}", token):
        return True
    return False


def _count_words(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len([w for w in re.split(r"\s+", stripped) if w])


def _is_initialism(text: str, index: int) -> bool:
    if text[index] != "." or index == 0:
        return False
    prev_char = text[index - 1]
    if not prev_char.isalpha():
        return False
    next_index = _next_non_space(text, index + 1)
    if next_index >= len(text):
        return False
    next_char = text[next_index]
    if not next_char.isalpha():
        return False
    lookahead = next_index + 1
    if lookahead < len(text) and text[lookahead] == ".":
        return True
    return False


def _looks_like_sentence_continuation(text: str, index: int) -> bool:
    next_index = _next_non_space(text, index + 1)
    if next_index >= len(text):
        return False
    next_char = text[next_index]
    if next_char.islower():
        return True
    if next_char in OPENING_PUNCTUATION:
        probe = _next_non_space(text, next_index + 1)
        if probe < len(text) and text[probe].islower():
            return True
    return False


def _consume_trailing_closers(text: str, index: int) -> int:
    while index < len(text) and text[index] in CLOSING_PUNCTUATION:
        index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _starts_new_sentence_after_period(text: str, index: int) -> bool:
    """
    针对字幕/口语文本，识别 '. he/.\" she/.\" then' 等小写开头但实际上是新句子的场景。
    """
    next_index = _next_non_space(text, index + 1)
    if next_index >= len(text):
        return False

    # 跳过开引号
    if text[next_index] in OPENING_PUNCTUATION:
        next_index = _next_non_space(text, next_index + 1)
        if next_index >= len(text):
            return False

    # 读取后面的连续字母作为一个单词
    end = next_index
    while end < len(text) and text[end].isalpha():
        end += 1
    word = text[next_index:end].lower()

    # 常见的对话/连接词开头，尽管是小写，也可以视为“新句子”
    return word in {"i", "he", "she", "we", "they", "it", "then", "so", "but", "and"}


def _is_sentence_terminator(text: str, index: int) -> bool:
    char = text[index]
    if char in {"?", "!", "。", "？", "！"}:
        return True
    if char != ".":
        return False
    if _is_decimal_point(text, index):
        return False
    if _is_known_abbreviation(text, index) or _is_initialism(text, index):
        return False
    # 如果看起来像“句子继续”，但后面符合对话/连接词开头模式，则仍然视为句末
    if _looks_like_sentence_continuation(text, index) and not _starts_new_sentence_after_period(text, index):
        return False
    return True


def _is_numeric_comma(text: str, index: int) -> bool:
    left = index - 1
    right = index + 1
    if left < 0 or right >= len(text):
        return False
    return text[left].isdigit() and text[right].isdigit()


def _is_time_colon(text: str, index: int) -> bool:
    left = index - 1
    right = index + 1
    if left < 0 or right >= len(text):
        return False
    if not text[left].isdigit() or not text[right].isdigit():
        return False
    left_start = left
    while left_start - 1 >= 0 and text[left_start - 1].isdigit():
        left_start -= 1
    right_end = right
    while right_end + 1 < len(text) and text[right_end + 1].isdigit():
        right_end += 1
    return 1 <= (left - left_start + 1) <= 2 and 1 <= (right_end - right + 1) <= 2


def _split_by_hard_punctuation(text: str) -> List[str]:
    segments: List[str] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char in HARD_END_PUNCTUATION and _is_sentence_terminator(text, i):
            if char == ".":
                i = _ellipsis_span_end(text, i)
            else:
                i += 1
            boundary = _consume_trailing_closers(text, i)
            segments.append(text[start:boundary])
            start = boundary
            i = boundary
        else:
            i += 1

    if start < len(text):
        segments.append(text[start:])

    return [seg for seg in segments if seg]


def _split_at_pauses(text: str, words: List[Dict], threshold: float = None) -> List[str]:
    """
    在停顿位置切分文本
    
    Args:
        text: 原始文本
        words: ASR 词列表（含时间戳）
        threshold: 停顿阈值
    
    Returns:
        切分后的文本片段列表
    """
    if not words or not text:
        return [text] if text else []
    
    pause_indices = detect_pauses(words, threshold)
    if not pause_indices:
        return [text]
    
    # 构建词到文本位置的映射并在停顿处切分
    segments = []
    text_lower = text.lower()
    cursor = 0
    last_cut = 0
    
    for i, word in enumerate(words):
        word_text = (word.get("word") or word.get("text", "") or "").strip()
        if not word_text:
            continue
        
        # 在文本中查找该词
        word_clean = re.sub(r"[^\w\s]", "", word_text.lower()).strip()
        if not word_clean:
            continue
        
        search_text = text_lower[cursor:]
        word_pos = -1
        match_len = 0
        
        for pattern in (word_text.lower(), word_clean):
            # 使用正则表达式确保词边界匹配，避免匹配到词的中间
            # 例如：避免在 "things" 中匹配到 "thin"
            boundary_pattern = r'(?<![a-zA-Z])' + re.escape(pattern) + r'(?![a-zA-Z])'
            match = re.search(boundary_pattern, search_text, re.IGNORECASE)
            if match:
                word_pos = cursor + match.start()
                match_len = len(pattern)
                break
        
        if word_pos == -1:
            continue
        
        # 找到词的结束位置（包含尾随标点）
        word_end = word_pos + match_len
        while word_end < len(text) and not text[word_end].isspace() and not text[word_end].isalnum():
            word_end += 1
        
        cursor = word_end
        
        # 如果这个词后面应该切分
        if i in pause_indices:
            segment = text[last_cut:word_end].strip()
            if segment:
                segments.append(segment)
            last_cut = word_end
            # 跳过后续空格
            while last_cut < len(text) and text[last_cut].isspace():
                last_cut += 1
    
    # 添加剩余文本
    if last_cut < len(text):
        tail = text[last_cut:].strip()
        if tail:
            segments.append(tail)
    
    return segments if segments else [text]


def _split_at_conjunctions(text: str, max_len: int = 65) -> List[str]:
    """
    在连词处切分长文本
    
    切分优先级：
    1. 首选：and, but, so, or, because 之前
    2. 次选：which, when, where, who, if, although 之前
    3. 再次：逗号后
    
    Args:
        text: 原始文本
        max_len: 最大长度
    
    Returns:
        切分后的文本片段列表
    """
    text = text.strip()
    if len(text) <= max_len:
        return [text]
    
    # 定义连词模式
    primary_conj = re.compile(r'\s+(and|but|so|or|because)\s+', re.IGNORECASE)
    secondary_conj = re.compile(r'\s+(which|when|where|who|whom|if|although|though|while|since|that)\s+', re.IGNORECASE)
    
    segments = []
    remaining = text
    
    while len(remaining) > max_len:
        best_split = -1
        best_priority = 999
        
        # 在 max_len 范围内查找最佳切分点
        search_window = remaining[:max_len + 20]  # 稍微放宽搜索范围
        
        # 优先级 1: 主要连词
        for match in primary_conj.finditer(search_window):
            split_pos = match.start()
            if 15 <= split_pos <= max_len:  # 确保切分点合理
                if best_priority > 1 or split_pos > best_split:
                    best_split = split_pos
                    best_priority = 1
        
        # 优先级 2: 次要连词
        if best_split == -1:
            for match in secondary_conj.finditer(search_window):
                split_pos = match.start()
                if 15 <= split_pos <= max_len:
                    if best_priority > 2 or split_pos > best_split:
                        best_split = split_pos
                        best_priority = 2
        
        # 优先级 3: 逗号
        if best_split == -1:
            comma_pos = -1
            for i, ch in enumerate(search_window):
                if ch == ',' and 15 <= i <= max_len:
                    comma_pos = i + 1  # 逗号后切分
            if comma_pos > 0:
                best_split = comma_pos
                best_priority = 3
        
        # 优先级 4: 最后一个空格（强制切分）
        if best_split == -1:
            space_pos = remaining[:max_len].rfind(' ')
            if space_pos > 10:
                best_split = space_pos + 1
            else:
                # 无法切分，强制在 max_len 处切
                best_split = max_len
        
        # 执行切分
        segment = remaining[:best_split].strip()
        if segment:
            segments.append(segment)
        remaining = remaining[best_split:].strip()
    
    # 添加剩余部分
    if remaining:
        segments.append(remaining)
    
    return segments


def split_text_with_rules(text: str, words: List[Dict] = None, segment_id: int = 0) -> List[Dict[str, object]]:
    """
    规则断句：多层切分
    
    Layer 1: 硬标点分句（. ? ! ;）
    Layer 2: 停顿检测断句（利用 ASR 词级时间戳）
    Layer 3: 连词规则断句
    Layer 4: 强制空格切分兜底
    
    Args:
        text: 原始文本
        words: ASR 词列表（可选）
        segment_id: 片段 ID
    
    Returns:
        断句结果列表
    """
    source_text = (text or "").strip()
    if not source_text:
        return []
    
    # Layer 1: 硬标点分句
    sentences_l1 = _split_by_hard_punctuation(source_text)
    print(f"[Rules-V2] Layer1 硬标点分句: {len(sentences_l1)} 个句子 (segment {segment_id})")
    
    # Layer 2: 停顿检测断句
    sentences_l2 = []
    for sent in sentences_l1:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(sent) <= MAX_SEGMENT_LENGTH:
            sentences_l2.append(sent)
        elif words:
            # 尝试用停顿切分
            paused = _split_at_pauses(sent, words)
            sentences_l2.extend(paused)
        else:
            sentences_l2.append(sent)
    
    print(f"[Rules-V2] Layer2 停顿断句: {len(sentences_l2)} 个句子")
    
    # Layer 3: 连词规则断句
    sentences_l3 = []
    for sent in sentences_l2:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(sent) <= MAX_SEGMENT_LENGTH:
            sentences_l3.append(sent)
        else:
            # 用连词规则切分
            conj_split = _split_at_conjunctions(sent, MAX_SEGMENT_LENGTH)
            sentences_l3.extend(conj_split)
    
    print(f"[Rules-V2] Layer3 连词断句: {len(sentences_l3)} 个句子")
    
    # Layer 4: 强制切分（兜底）
    final_segments = []
    for sent in sentences_l3:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(sent) <= MAX_SEGMENT_LENGTH:
            final_segments.append(sent)
        else:
            # 强制按空格切分
            forced = _split_by_max_chars(sent, MAX_SEGMENT_LENGTH)
            final_segments.extend(forced)
    
    # 后处理
    final_segments = _fix_comma_at_start(final_segments)
    final_segments = _merge_single_word_lines(final_segments)
    
    print(f"✅ Rules-V2 断句完成 (segment {segment_id}): {len(final_segments)} 条句子")
    return [{"text": s, "index": idx} for idx, s in enumerate(final_segments)]



def _fix_comma_at_start(lines: List[str]) -> List[str]:
    fixed_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i > 0 and line.startswith((",", ".", "!", "?", ";", ":")):
            match = re.match(r"^([,\.\!\?\;\:]+)\s*(.*)", line)
            if match:
                punct = match.group(1)
                content = match.group(2)
                if fixed_lines:
                    fixed_lines[-1] = fixed_lines[-1].rstrip() + punct
                if content:
                    fixed_lines.append(content)
                continue
        fixed_lines.append(line)
    return fixed_lines


def _merge_single_word_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    for line in lines:
        word_count = len(re.findall(r"\b\w+\b", line))
        if word_count == 1 and merged:
            merged[-1] = (merged[-1].rstrip() + " " + line.strip()).strip()
        else:
            merged.append(line)
    return merged

def detect_pauses(words: List[Dict], threshold: float = None) -> List[int]:
    """
    检测词间停顿，返回应在停顿后断句的词索引
    
    Args:
        words: ASR 词列表，每个词包含 start, end 时间戳
        threshold: 停顿阈值（秒），None 则使用配置值
    
    Returns:
        pause_indices: 应在该词之后插入停顿标记的索引列表
    """
    if threshold is None:
        threshold = PAUSE_THRESHOLD
    
    pause_indices = []
    for i in range(len(words) - 1):
        try:
            current_end = float(words[i].get("end", 0))
            next_start = float(words[i + 1].get("start", 0))
            gap = next_start - current_end
            if gap >= threshold:
                pause_indices.append(i)
        except (ValueError, TypeError):
            continue
    return pause_indices


def _split_by_max_chars(text: str, max_chars: int) -> List[str]:
    """
    Layer 4 兜底切分：在句子中间找词边界切分
    
    策略：找到句子中间位置附近的空格，尽量让切分后的两部分长度相近
    """
    if not text:
        return []
    max_chars = max(1, int(max_chars))
    remaining = (text or "").strip()
    parts: List[str] = []
    
    while len(remaining) > max_chars:
        # 目标：在中间位置附近找词边界
        mid_point = len(remaining) // 2
        
        # 向两边搜索最近的空格
        left_space = remaining.rfind(" ", 0, mid_point + 1)
        right_space = remaining.find(" ", mid_point)
        
        # 选择离中点更近的空格
        if left_space == -1 and right_space == -1:
            # 没有空格，强制在中间切
            cut = mid_point
        elif left_space == -1:
            cut = right_space
        elif right_space == -1:
            cut = left_space
        else:
            # 选择离中点更近的那个
            if (mid_point - left_space) <= (right_space - mid_point):
                cut = left_space
            else:
                cut = right_space
        
        # 确保 cut 不会太靠近边缘（至少保留 10 个字符）
        if cut < 10:
            cut = remaining.find(" ", 10)
            if cut == -1:
                cut = mid_point
        
        piece = remaining[:cut].strip()
        if piece:
            parts.append(piece)
        remaining = remaining[cut:].strip()
    
    if remaining:
        parts.append(remaining)
    return parts


def load_transcription_data(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_segments(segments: list, output_path: Path):
    if not isinstance(segments, list):
        print(f"⚠️ Warning: Segments for {output_path.name} is not a list. Saving as empty.")
        segments = []
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"sentences": segments}, f, ensure_ascii=False, indent=2)


def split_text_dispatcher(text: str, segment_id: int, words: List[Dict] = None) -> List[Dict[str, object]]:
    """
    断句分发器：支持 rule / gemini / grok / deepseek / local
    
    Args:
        text: 原始文本
        segment_id: 片段 ID
        words: ASR 词列表（可选，rule 模式会用来做停顿断句）
    
    Returns:
        断句结果列表
    """
    service = (SEGMENTATION_SERVICE or "").strip().lower()
    
    if service == "rule":
        segments = split_text_with_rules(text, words=words, segment_id=segment_id)
    elif service in {"gemini", "vertex", "grok", "deepseek", "local"}:
        segments = split_text_with_llm(text, segment_id)
    else:
        raise ValueError(f"Unknown SEGMENTATION_SERVICE: {SEGMENTATION_SERVICE}")

    for idx, segment in enumerate(segments):
        segment.setdefault("index", idx)
    return segments


def split_text_with_llm(text: str, segment_id: int) -> list:
    max_retries = 3
    retry_delay = 5
    raw_response_text = ""
    prompt = SPLIT_TEXT_PROMPT.format(text=text)
    normalized_source_text = normalize_text(text)

    seg_service = (SEGMENTATION_SERVICE or "").strip().lower()
    if seg_service == "gemini" and genai is None:
        msg = "SEGMENTATION_SERVICE=gemini 但未安装 google-generativeai，无法进行 LLM 分句。"
        print(f"Warning: {msg}")
        logger.error(msg)
        return []
    if seg_service in {"grok", "local", "deepseek"} and OpenAI is None:
        msg = f"SEGMENTATION_SERVICE={seg_service} 但未安装 openai，无法进行 LLM 分句。"
        print(f"Warning: {msg}")
        logger.error(msg)
        return []
    if seg_service == "deepseek" and not DEEPSEEK_API_KEY:
        msg = "SEGMENTATION_SERVICE=deepseek 但未配置 DEEPSEEK_API_KEY，无法进行 LLM 分句。"
        print(f"Warning: {msg}")
        logger.error(msg)
        return []

    for attempt in range(max_retries):
        try:
            if seg_service == "gemini":
                print(f"⚙️ (尝试 {attempt + 1}/{max_retries}) 正在处理 segment {segment_id} (Gemini)...")
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL_SEGMENTATION, generation_config={"temperature": LLM_TEMPERATURE_SEGMENTATION})
                response = model.generate_content(prompt)
                try:
                    raw_response_text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    raw_response_text = getattr(response, "text", "")
            if seg_service == "vertex":
                print(f"⚙️ (尝试 {attempt + 1}/{max_retries}) 正在处理 segment {segment_id} (Vertex AI)...")
                model = build_google_text_client("vertex", VERTEX_MODEL_SEGMENTATION)
                raw_response_text = generate_google_text(model, prompt, LLM_TEMPERATURE_SEGMENTATION)
            elif seg_service == "grok":
                print(f"⚙️ (尝试 {attempt + 1}/{max_retries}) 正在处理 segment {segment_id} (Grok)...")
                client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
                response = client.chat.completions.create(model=GROK_MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=LLM_TEMPERATURE_SEGMENTATION)
                raw_response_text = response.choices[0].message.content.strip() if response.choices else ""
            elif seg_service == "local":
                print(f"[Local LLM] Attempt {attempt + 1}/{max_retries}: processing segment {segment_id} (local segmentation service)...")
                client = OpenAI(base_url=LOCAL_API_BASE_URL, api_key=LOCAL_API_KEY)
                response = client.chat.completions.create(model=LOCAL_MODEL_SEGMENTATION, messages=[{"role": "user", "content": prompt}], temperature=LLM_TEMPERATURE_SEGMENTATION)
                raw_response_text = response.choices[0].message.content.strip() if response.choices else ""
            elif seg_service == "deepseek":
                print(f"⚙️ (尝试 {attempt + 1}/{max_retries}) 正在处理 segment {segment_id} (DeepSeek)...")
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE_URL)
                response = client.chat.completions.create(model=DEEPSEEK_MODEL_SEGMENTATION, messages=[{"role": "user", "content": prompt}], temperature=LLM_TEMPERATURE_SEGMENTATION)
                raw_response_text = response.choices[0].message.content.strip() if response.choices else ""
            else:
                raise ValueError(f"未知的 SEGMENTATION_SERVICE 配置: {SEGMENTATION_SERVICE}")

            raw = raw_response_text.strip()
            def _clean_json(t: str) -> str:
                t = re.sub(r'```[a-zA-Z]*', '', t)
                t = t.replace('```', '')
                t = re.sub(r',\s*([\]}])', r'\1', t)
                return t

            start_index = raw.find('{')
            end_index = raw.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = _clean_json(raw[start_index:end_index + 1])
                result = json.loads(json_str)

                if "sentences" not in result:
                    raise ValueError("JSON解析成功但缺少 'sentences' 字段")

                sentences = result.get("sentences", [])
                segmented_text = "".join([s.get("text", "") for s in sentences])
                normalized_segmented_text = normalize_text(segmented_text)

                similarity_ratio = difflib.SequenceMatcher(
                    None,
                    normalized_source_text,
                    normalized_segmented_text
                ).ratio()

                if similarity_ratio > 0.993:
                    print(f"✅ 内容校验通过 (相似度: {similarity_ratio:.2%}, 尝试 {attempt + 1}/{max_retries})。")
                    print(f"✅ 成功从 LLM 获取并解析了 {len(sentences)} 条句子。")
                    return sentences
                else:
                    error_message = (
                        f"内容校验失败: 文本相似度 ({similarity_ratio:.2%}) 未达到 99.3% 的阈值。"
                    )
                    print(f"❌ {error_message} (尝试 {attempt + 1}/{max_retries})")
                    raise ValueError(error_message)
            else:
                raise ValueError(f"LLM 回应中找不到有效的 JSON 对象。")

        except Exception as e:
            error_message = f"處理 segment {segment_id} 时发生错误: {e}"
            print(f"❌ {error_message}")
            if attempt < max_retries - 1:
                print(f"🕒 将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                log_entry = (
                    f"在对 segment {segment_id} 进行 {max_retries} 次尝试后最终失败。\n"
                    f"错误详情: {e}\n"
                    f"LLM 原始回应: {raw_response_text if raw_response_text else '无'}\n"
                    f"-----------------------------------------"
                )
                logger.error(log_entry)
                print(f"❌ 所有重试均失败。详细错误已记录至 {LOG_FILE_PATH}")

    return []


def _group_by_speaker(segments: list) -> list:
    """
    Layer 0: 将 ASR segments 按连续相同 speaker 分组。
    
    输入: [{text, speaker, words, ...}, ...]
    输出: [(speaker, combined_text, combined_words), ...]
    """
    groups = []
    current_speaker = None
    current_texts = []
    current_words = []

    for seg in segments:
        speaker = seg.get("speaker")
        text = (seg.get("text") or "").strip()
        words = seg.get("words", [])

        if speaker != current_speaker and current_texts:
            # 说话人切换，保存上一组
            groups.append((current_speaker, " ".join(current_texts), current_words))
            current_texts = []
            current_words = []

        current_speaker = speaker
        if text:
            current_texts.append(text)
        current_words.extend(words)

    # 保存最后一组
    if current_texts:
        groups.append((current_speaker, " ".join(current_texts), current_words))

    return groups


def process_audio_segment(segment_id: int, asr_dir: Path, output_dir: Path) -> bool:
    json_path = asr_dir / f"segment_{segment_id:03d}.json"
    output_path = output_dir / f"segment_{segment_id:03d}_segments.json"
    
    if output_path.exists():
        print(f"⏩ 跳过 segment {segment_id}: 输出文件已存在")
        return True
    
    try:
        transcription_data = load_transcription_data(json_path)
        asr_segments = transcription_data.get("segments", [])
        
        # --- Layer 0: 检查是否有 speaker 标签 ---
        has_speaker = any(seg.get("speaker") for seg in asr_segments)
        
        if has_speaker:
            # 按连续相同说话人分组，每组独立断句
            speaker_groups = _group_by_speaker(asr_segments)
            print(f"🗣️ Layer 0: 检测到 {len(set(g[0] for g in speaker_groups))} 位说话人, "
                  f"{len(speaker_groups)} 个说话人轮次 (segment {segment_id})")
            
            all_sentences = []
            global_index = 0
            for speaker, group_text, group_words in speaker_groups:
                group_text = group_text.strip()
                if not group_text:
                    continue
                
                sentences = split_text_dispatcher(group_text, segment_id, words=group_words)
                for sent in sentences:
                    sent["index"] = global_index
                    if speaker:
                        sent["speaker"] = speaker
                    global_index += 1
                all_sentences.extend(sentences)
            
            if all_sentences:
                save_segments(all_sentences, output_path)
                print(f"✅ 已处理 segment {segment_id}: {len(all_sentences)} 条句子已储存至 {output_path.name}")
                return True
            else:
                print(f"⚠️ 已处理 segment {segment_id}, 但未生成有效内容。")
                save_segments([], output_path)
                return True
        
        # --- 无 speaker 标签：走原有逻辑 ---
        text = transcription_data.get("text", "").strip()
        if not text and asr_segments:
            segment_texts = [seg.get("text", "").strip() for seg in asr_segments if seg.get("text", "").strip()]
            text = " ".join(segment_texts)

        if not text:
            print(f"⚠️ 警告: 在 {json_path.name} 中未找到有效文本，将生成空文件。")
            save_segments([], output_path)
            return True
        
        # 提取词级时间戳（用于 rule 模式停顿断句）
        words = []
        if transcription_data.get("word_segments"):
            words = transcription_data["word_segments"]
        elif transcription_data.get("words"):
            words = transcription_data["words"]
        else:
            # 从 segments 中提取 words
            for seg in asr_segments:
                words.extend(seg.get("words", []))
        
        segments = split_text_dispatcher(text, segment_id, words=words)
        
        if segments:
            save_segments(segments, output_path)
            print(f"✅ 已处理 segment {segment_id}: {len(segments)} 条句子已储存至 {output_path.name}")
            return True
        else:
            print(f"⚠️ 已处理 segment {segment_id}, 但LLM最终未能生成有效且匹配的内容。此切片处理失败。")
            return False
        
    except Exception as e:
        error_msg = f"🔥🔥🔥 处理 segment {segment_id} 时发生严重错误: {e}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return False

def main():
    asr_dir = WORKDIR / "asr"
    segments_dir = WORKDIR / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(asr_dir.glob("segment_*.json"))
    
    if not json_files:
        print(f"在 {asr_dir} 中未找到 'segment_*.json' 文件")
        return
        
    print(f"找到 {len(json_files)} 个文件待处理。")
    
    for json_file in json_files:
        try:
            segment_id = int(json_file.stem.split("_")[1])
            process_audio_segment(segment_id, asr_dir, segments_dir)
        except (IndexError, ValueError):
            print(f"无法从档名解析 segment ID: {json_file.name}。已跳过。")

if __name__ == "__main__":
    main()
