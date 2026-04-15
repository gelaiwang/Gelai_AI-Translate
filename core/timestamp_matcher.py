# _3_3_match_timestamps.py
# 重写版本：采用 DTW 全局对齐 + 锚点检测 + 后处理优化
# 参考主流工具(aeneas, gentle, WhisperX)的对齐策略

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import srt

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy 未安装，将使用贪婪匹配替代匈牙利算法")

try:
    from config import WORKDIR
except ImportError:
    print("警告：未找到 config 包，将使用脚本所在目录作为工作目录。")
    WORKDIR = Path(__file__).parent


# ============================================================================
# 数据模型 (Data Models)
# ============================================================================

@dataclass
class AlignmentResult:
    """单个句子的对齐结果"""
    start_time: float
    end_time: float
    confidence: float  # 0.0-1.0
    method: str  # "dtw", "anchor", "hungarian", "greedy", "fallback"
    matched_word_indices: List[int] = field(default_factory=list)
    
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class Anchor:
    """锚点：高置信度的句子-ASR词对应关系"""
    sentence_idx: int
    word_idx: int
    token: str
    confidence: float


@dataclass
class AlignmentContext:
    """对齐上下文信息"""
    tokens: List[str]  # ASR词的规范化token列表
    word_frequency: Counter  # 全局词频统计
    avg_duration: float  # 平均词时长
    total_words: int


@dataclass
class SentenceInfo:
    """句子信息"""
    index: int
    text: str
    tokens: List[str]
    rare_tokens: Set[str]  # 稀有词集合


# ============================================================================
# 预处理模块 (Preprocessing)
# ============================================================================

def normalize_token(token: str) -> str:
    """规范化单个token: Unicode规范化 + 小写 + 移除标点"""
    if not token:
        return ""
    token = unicodedata.normalize("NFKC", token)
    token = token.lower()
    # 保留撇号用于英语缩写 (don't, it's)
    token = re.sub(r"[^a-z0-9']+", "", token)
    # 移除首尾撇号
    token = token.strip("'")
    return token


def tokenize_sentence(sentence: str) -> List[str]:
    """将句子分词为规范化token列表"""
    # 处理常见缩写和连字符
    sentence = sentence.replace("'", "'").replace("`", "'")
    sentence = re.sub(r"(?<=\w)[-–—]+(?=\w)", "", sentence)  # 移除词内连字符
    # 提取英文单词
    tokens = re.findall(r"[A-Za-z0-9']+", sentence)
    # 规范化
    normalized = [normalize_token(tok) for tok in tokens]
    return [tok for tok in normalized if tok]


def extract_asr_words(transcription_data: Dict) -> List[Dict]:
    """从ASR JSON中提取词列表"""
    if not transcription_data:
        return []
    # WhisperX 格式
    if transcription_data.get("word_segments"):
        return transcription_data["word_segments"]
    # 通用 Whisper 格式
    if transcription_data.get("words"):
        return transcription_data["words"]
    # segments 嵌套格式
    flat_words: List[Dict] = []
    for seg in transcription_data.get("segments", []):
        flat_words.extend(seg.get("words", []))
    return flat_words


def clean_asr_words(words: List[Dict]) -> List[Dict]:
    """清理和规范化ASR词列表"""
    cleaned: List[Dict] = []
    for w in words:
        # 获取原始文本
        raw_text = w.get("word") or w.get("text")
        if raw_text is None:
            continue
        
        # 获取时间戳
        try:
            start = float(w["start"])
            end = float(w["end"])
        except (TypeError, ValueError, KeyError):
            continue
        
        if end < start:
            continue
        
        # 规范化token
        normalized = normalize_token(str(raw_text))
        if not normalized:
            continue
        
        # 构建清理后的词条目
        new_word = dict(w)
        new_word["text"] = str(raw_text)
        new_word["start"] = start
        new_word["end"] = end
        new_word["normalized"] = normalized
        new_word["duration"] = max(end - start, 0.01)
        # ASR置信度 (如果有)
        new_word["confidence"] = float(w.get("probability", w.get("confidence", 0.8)))
        cleaned.append(new_word)
    
    return cleaned


def build_alignment_context(words: List[Dict]) -> AlignmentContext:
    """构建对齐上下文"""
    tokens = [w.get("normalized", "") for w in words]
    frequency = Counter(tok for tok in tokens if tok)
    
    total_duration = sum(w.get("duration", 0.3) for w in words)
    avg_duration = total_duration / len(words) if words else 0.3
    
    return AlignmentContext(
        tokens=tokens,
        word_frequency=frequency,
        avg_duration=max(avg_duration, 0.05),
        total_words=len(words)
    )


def build_sentence_info(sentences: List[str], context: AlignmentContext) -> List[SentenceInfo]:
    """构建句子信息列表，识别稀有词"""
    result = []
    for idx, text in enumerate(sentences):
        tokens = tokenize_sentence(text)
        # 稀有词 = 在ASR中出现频率 <= 2 的词
        rare_tokens = {
            tok for tok in tokens 
            if context.word_frequency.get(tok, 0) <= 2 and len(tok) >= 3
        }
        result.append(SentenceInfo(
            index=idx,
            text=text,
            tokens=tokens,
            rare_tokens=rare_tokens
        ))
    return result


# ============================================================================
# 相似度计算 (Similarity)
# ============================================================================

def token_similarity(a: str, b: str) -> float:
    """计算两个token的相似度 (0.0-1.0)"""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    
    # 短词：要求精确匹配或包含关系
    if len(a) <= 3 or len(b) <= 3:
        if a in b or b in a:
            return 0.7
        return 0.0
    
    # 长词：使用序列匹配
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return ratio


def sentence_span_similarity(sentence_tokens: List[str], asr_tokens: List[str]) -> float:
    """计算句子tokens与ASR tokens跨度的相似度"""
    if not sentence_tokens or not asr_tokens:
        return 0.0
    
    # 基于token覆盖率
    matched = 0
    asr_counter = Counter(asr_tokens)
    for tok in sentence_tokens:
        if asr_counter.get(tok, 0) > 0:
            matched += 1
            asr_counter[tok] -= 1
    
    coverage = matched / len(sentence_tokens)
    
    # 长度匹配惩罚
    length_ratio = min(len(sentence_tokens), len(asr_tokens)) / max(len(sentence_tokens), len(asr_tokens))
    
    return coverage * 0.7 + length_ratio * 0.3


# ============================================================================
# 锚点检测 (Anchor Detection)
# ============================================================================

def detect_anchors(
    sentence_infos: List[SentenceInfo],
    words: List[Dict],
    context: AlignmentContext,
    min_confidence: float = 0.85
) -> List[Anchor]:
    """检测高置信度锚点"""
    anchors = []
    
    for sent_info in sentence_infos:
        if not sent_info.rare_tokens:
            continue
        
        # 对每个稀有词查找唯一匹配
        for rare_token in sent_info.rare_tokens:
            # 在ASR词中查找匹配
            matches = []
            for word_idx, word in enumerate(words):
                asr_token = word.get("normalized", "")
                sim = token_similarity(rare_token, asr_token)
                if sim >= 0.85:
                    word_conf = word.get("confidence", 0.8)
                    matches.append((word_idx, sim * word_conf))
            
            # 只有唯一高置信度匹配才作为锚点
            if len(matches) == 1 and matches[0][1] >= min_confidence:
                word_idx, conf = matches[0]
                anchors.append(Anchor(
                    sentence_idx=sent_info.index,
                    word_idx=word_idx,
                    token=rare_token,
                    confidence=conf
                ))
                break  # 每个句子只需要一个锚点
    
    # 按句子索引排序并验证单调性
    anchors.sort(key=lambda a: (a.sentence_idx, a.word_idx))
    
    # 移除不满足单调性的锚点
    valid_anchors = []
    last_word_idx = -1
    for anchor in anchors:
        if anchor.word_idx > last_word_idx:
            valid_anchors.append(anchor)
            last_word_idx = anchor.word_idx
    
    return valid_anchors


def partition_by_anchors(
    sentence_infos: List[SentenceInfo],
    words: List[Dict],
    anchors: List[Anchor]
) -> List[Tuple[range, range]]:
    """将对齐问题按锚点分割成子问题
    
    Returns:
        List of (sentence_range, word_range) tuples
    """
    if not sentence_infos or not words:
        return []
    
    if not anchors:
        # 无锚点，整体作为一个分区
        return [(range(len(sentence_infos)), range(len(words)))]
    
    partitions = []
    
    # 第一个分区: 开始 -> 第一个锚点
    first_anchor = anchors[0]
    if first_anchor.sentence_idx > 0 or first_anchor.word_idx > 0:
        sent_range = range(0, first_anchor.sentence_idx + 1)
        word_range = range(0, first_anchor.word_idx + 1)
        partitions.append((sent_range, word_range))
    
    # 中间分区: 锚点 -> 锚点
    for i in range(len(anchors) - 1):
        curr = anchors[i]
        next_anchor = anchors[i + 1]
        sent_range = range(curr.sentence_idx, next_anchor.sentence_idx + 1)
        word_range = range(curr.word_idx, next_anchor.word_idx + 1)
        partitions.append((sent_range, word_range))
    
    # 最后一个分区: 最后一个锚点 -> 结束
    last_anchor = anchors[-1]
    if last_anchor.sentence_idx < len(sentence_infos) - 1 or last_anchor.word_idx < len(words) - 1:
        sent_range = range(last_anchor.sentence_idx, len(sentence_infos))
        word_range = range(last_anchor.word_idx, len(words))
        partitions.append((sent_range, word_range))
    
    # 如果只有锚点没有分区，创建一个覆盖所有的分区
    if not partitions:
        partitions = [(range(len(sentence_infos)), range(len(words)))]
    
    return partitions


# ============================================================================
# DTW 全局对齐 (DTW Global Alignment)
# ============================================================================

def build_cost_matrix(
    sentence_infos: List[SentenceInfo],
    words: List[Dict],
    context: AlignmentContext
) -> List[List[float]]:
    """构建DTW代价矩阵
    
    C[i][j] = 将第i个句子的起始对齐到第j个ASR词的代价
    """
    n_sentences = len(sentence_infos)
    n_words = len(words)
    
    if n_sentences == 0 or n_words == 0:
        return []
    
    # 初始化代价矩阵 (高代价 = 不匹配)
    INF = 1e9
    cost_matrix = [[INF] * n_words for _ in range(n_sentences)]
    
    for i, sent_info in enumerate(sentence_infos):
        if not sent_info.tokens:
            continue
        
        sent_len = len(sent_info.tokens)
        
        for j in range(n_words):
            # 计算从位置j开始匹配句子的代价
            # 估计句子需要的词数范围
            min_span = max(1, sent_len - 3)
            max_span = min(n_words - j, sent_len + 8)
            
            best_cost = INF
            for span_len in range(min_span, max_span + 1):
                end_idx = j + span_len
                if end_idx > n_words:
                    break
                
                # 提取ASR tokens
                asr_tokens = [words[k].get("normalized", "") for k in range(j, end_idx)]
                
                # 计算相似度
                similarity = sentence_span_similarity(sent_info.tokens, asr_tokens)
                
                # 代价 = 1 - 相似度 (越相似代价越低)
                cost = 1.0 - similarity
                
                # 长度惩罚
                length_diff = abs(span_len - sent_len)
                cost += length_diff * 0.02
                
                best_cost = min(best_cost, cost)
            
            cost_matrix[i][j] = best_cost
    
    return cost_matrix


def dtw_align_partition(
    sentence_infos: List[SentenceInfo],
    words: List[Dict],
    context: AlignmentContext
) -> List[Tuple[int, int, int]]:
    """对单个分区执行DTW对齐
    
    Returns:
        List of (sentence_idx, start_word_idx, end_word_idx)
    """
    n_sentences = len(sentence_infos)
    n_words = len(words)
    
    if n_sentences == 0 or n_words == 0:
        return []
    
    if n_sentences == 1:
        # 只有一个句子，直接覆盖整个词范围
        return [(0, 0, n_words - 1)]
    
    # 构建代价矩阵
    cost_matrix = build_cost_matrix(sentence_infos, words, context)
    
    # DP: dp[i][j] = 将前i个句子对齐到前j个词的最小代价
    INF = 1e9
    dp = [[INF] * (n_words + 1) for _ in range(n_sentences + 1)]
    parent = [[(-1, -1)] * (n_words + 1) for _ in range(n_sentences + 1)]
    
    dp[0][0] = 0
    
    for i in range(1, n_sentences + 1):
        sent_info = sentence_infos[i - 1]
        sent_len = len(sent_info.tokens) if sent_info.tokens else 1
        
        # 估计句子需要的词数范围
        min_span = max(1, sent_len - 2)
        max_span = min(n_words, sent_len + 10)
        
        for j in range(1, n_words + 1):
            # 尝试不同的起始位置
            for prev_j in range(max(0, j - max_span), j - min_span + 1):
                if dp[i - 1][prev_j] >= INF:
                    continue
                
                # 句子i从prev_j开始，到j-1结束
                start_word = prev_j
                if start_word >= n_words:
                    continue
                
                cost = cost_matrix[i - 1][start_word] if start_word < len(cost_matrix[i - 1]) else INF
                
                # 跳跃惩罚 (如果跳过了词)
                skip_penalty = 0
                if i > 1 and prev_j > 0:
                    # 检查是否跳过了词
                    pass  # 暂不添加跳跃惩罚
                
                total_cost = dp[i - 1][prev_j] + cost + skip_penalty
                
                if total_cost < dp[i][j]:
                    dp[i][j] = total_cost
                    parent[i][j] = (i - 1, prev_j)
    
    # 回溯找最优路径
    # 从最后一行找最小代价的结束位置
    best_j = n_words
    best_cost = INF
    for j in range(1, n_words + 1):
        if dp[n_sentences][j] < best_cost:
            best_cost = dp[n_sentences][j]
            best_j = j
    
    # 回溯
    path = []
    i, j = n_sentences, best_j
    while i > 0:
        prev_i, prev_j = parent[i][j]
        if prev_i >= 0:
            # 句子 i-1 从 prev_j 开始，到 j-1 结束
            start_word = prev_j
            end_word = j - 1
            path.append((i - 1, start_word, end_word))
        i, j = prev_i, prev_j
    
    path.reverse()
    return path


# ============================================================================
# 匈牙利词匹配 (Hungarian Word Matching)
# ============================================================================

def hungarian_match(
    sentence_tokens: List[str],
    asr_words: List[Dict]
) -> List[Tuple[int, int]]:
    """使用匈牙利算法找到最优词匹配
    
    Returns:
        List of (sentence_token_idx, asr_word_idx) pairs
    """
    if not sentence_tokens or not asr_words:
        return []
    
    n_sent = len(sentence_tokens)
    n_asr = len(asr_words)
    
    # 构建相似度矩阵
    sim_matrix = []
    for i, sent_tok in enumerate(sentence_tokens):
        row = []
        for j, asr_word in enumerate(asr_words):
            asr_tok = asr_word.get("normalized", "")
            sim = token_similarity(sent_tok, asr_tok)
            row.append(sim)
        sim_matrix.append(row)
    
    if HAS_SCIPY:
        # 使用scipy的匈牙利算法 (需要代价矩阵，所以取负)
        import numpy as np
        cost_matrix = np.array([[-sim for sim in row] for row in sim_matrix])
        
        # 如果矩阵不是方阵，需要填充
        max_dim = max(n_sent, n_asr)
        padded = np.zeros((max_dim, max_dim))
        padded[:n_sent, :n_asr] = cost_matrix
        
        row_ind, col_ind = linear_sum_assignment(padded)
        
        # 过滤有效匹配
        matches = []
        for r, c in zip(row_ind, col_ind):
            if r < n_sent and c < n_asr and sim_matrix[r][c] > 0.5:
                matches.append((r, c))
        
        return sorted(matches, key=lambda x: x[1])
    
    else:
        # 贪婪匹配作为fallback
        return greedy_match(sentence_tokens, asr_words)


def greedy_match(
    sentence_tokens: List[str],
    asr_words: List[Dict]
) -> List[Tuple[int, int]]:
    """贪婪词匹配 (scipy不可用时的fallback)"""
    if not sentence_tokens or not asr_words:
        return []
    
    matches = []
    used_asr = set()
    
    for sent_idx, sent_tok in enumerate(sentence_tokens):
        best_asr_idx = -1
        best_sim = 0.5  # 最低阈值
        
        for asr_idx, asr_word in enumerate(asr_words):
            if asr_idx in used_asr:
                continue
            asr_tok = asr_word.get("normalized", "")
            sim = token_similarity(sent_tok, asr_tok)
            if sim > best_sim:
                best_sim = sim
                best_asr_idx = asr_idx
        
        if best_asr_idx >= 0:
            matches.append((sent_idx, best_asr_idx))
            used_asr.add(best_asr_idx)
    
    return sorted(matches, key=lambda x: x[1])


# ============================================================================
# 边界精调 (Boundary Refinement)
# ============================================================================

def refine_timestamps(
    sentence_info: SentenceInfo,
    words: List[Dict],
    start_word_idx: int,
    end_word_idx: int,
    prev_end_time: float,
    next_start_time: Optional[float],
    context: AlignmentContext
) -> Tuple[float, float, float]:
    """精调句子的起止时间戳
    
    Returns:
        (start_time, end_time, confidence)
    """
    if not words or start_word_idx < 0 or end_word_idx < 0:
        return prev_end_time, prev_end_time + 1.0, 0.0
    
    # 确保索引有效
    start_word_idx = max(0, min(start_word_idx, len(words) - 1))
    end_word_idx = max(start_word_idx, min(end_word_idx, len(words) - 1))
    
    # 基础时间
    raw_start = words[start_word_idx]["start"]
    raw_end = words[end_word_idx]["end"]
    
    # 确保单调性
    start_time = max(raw_start, prev_end_time)
    
    # 确保不超过下一句开始
    if next_start_time is not None:
        end_time = min(raw_end, next_start_time)
    else:
        end_time = raw_end
    
    # 确保 end > start
    if end_time <= start_time:
        expected_duration = context.avg_duration * max(len(sentence_info.tokens), 1)
        end_time = start_time + expected_duration
    
    # 计算置信度
    span_words = words[start_word_idx:end_word_idx + 1]
    if span_words:
        # 基于词匹配计算置信度
        word_matches = hungarian_match(
            sentence_info.tokens,
            span_words
        )
        match_ratio = len(word_matches) / max(len(sentence_info.tokens), 1)
        avg_word_conf = sum(w.get("confidence", 0.8) for w in span_words) / len(span_words)
        confidence = match_ratio * 0.6 + avg_word_conf * 0.4
    else:
        confidence = 0.3
    
    return start_time, end_time, confidence


# ============================================================================
# 后处理 (Post-processing)
# ============================================================================

def fill_gaps(
    results: List[AlignmentResult],
    words: List[Dict],
    max_gap: float = 0.5
) -> List[AlignmentResult]:
    """填充字幕之间的静音间隙"""
    if len(results) < 2:
        return results
    
    filled = [results[0]]
    
    for i in range(1, len(results)):
        prev = filled[-1]
        curr = results[i]
        
        gap = curr.start_time - prev.end_time
        
        if 0 < gap <= max_gap:
            # 小间隙：将前一个字幕延长到下一个开始
            prev.end_time = curr.start_time
        
        filled.append(curr)
    
    return filled


def resolve_overlaps(results: List[AlignmentResult]) -> List[AlignmentResult]:
    """消除字幕重叠"""
    if len(results) < 2:
        return results
    
    resolved = [results[0]]
    
    for i in range(1, len(results)):
        prev = resolved[-1]
        curr = results[i]
        
        if curr.start_time < prev.end_time:
            # 有重叠，取中点
            midpoint = (prev.end_time + curr.start_time) / 2
            prev.end_time = midpoint
            curr.start_time = midpoint
        
        resolved.append(curr)
    
    return resolved


def smooth_boundaries(
    results: List[AlignmentResult],
    min_duration: float = 0.3
) -> List[AlignmentResult]:
    """平滑边界，确保最小时长"""
    for result in results:
        if result.duration() < min_duration:
            result.end_time = result.start_time + min_duration
    
    return results


def apply_post_processing(
    results: List[AlignmentResult],
    words: List[Dict]
) -> List[AlignmentResult]:
    """应用所有后处理步骤"""
    results = fill_gaps(results, words)
    results = resolve_overlaps(results)
    results = smooth_boundaries(results)
    return results


# ============================================================================
# 主对齐流程 (Main Alignment)
# ============================================================================

def align_sentences_to_words(
    sentences: List[str],
    words: List[Dict]
) -> List[AlignmentResult]:
    """将句子列表对齐到ASR词列表
    
    这是核心对齐函数，采用以下策略：
    1. 构建对齐上下文
    2. 检测锚点
    3. 按锚点分区
    4. 对每个分区执行DTW对齐
    5. 精调边界
    6. 后处理优化
    """
    if not sentences or not words:
        return []
    
    # 1. 构建上下文
    context = build_alignment_context(words)
    sentence_infos = build_sentence_info(sentences, context)
    
    # 2. 检测锚点
    anchors = detect_anchors(sentence_infos, words, context)
    print(f"  检测到 {len(anchors)} 个锚点")
    
    # 3. 按锚点分区
    partitions = partition_by_anchors(sentence_infos, words, anchors)
    print(f"  分割为 {len(partitions)} 个分区")
    
    # 4. 对每个分区执行对齐
    all_alignments: List[Tuple[int, int, int]] = []  # (sentence_idx, start_word, end_word)
    
    for sent_range, word_range in partitions:
        # 提取分区数据
        partition_sentences = [sentence_infos[i] for i in sent_range]
        partition_words = [words[i] for i in word_range]
        
        if not partition_sentences or not partition_words:
            continue
        
        # 执行DTW对齐
        partition_alignments = dtw_align_partition(
            partition_sentences,
            partition_words,
            context
        )
        
        # 转换回全局索引
        for local_sent_idx, local_start, local_end in partition_alignments:
            global_sent_idx = sent_range.start + local_sent_idx
            global_start = word_range.start + local_start
            global_end = word_range.start + local_end
            all_alignments.append((global_sent_idx, global_start, global_end))
    
    # 按句子索引排序
    all_alignments.sort(key=lambda x: x[0])
    
    # 5. 精调边界并构建结果
    results: List[AlignmentResult] = []
    prev_end_time = 0.0
    
    for i, (sent_idx, start_word, end_word) in enumerate(all_alignments):
        if sent_idx >= len(sentence_infos):
            continue
        
        sent_info = sentence_infos[sent_idx]
        
        # 获取下一个句子的开始时间（如果有）
        next_start_time = None
        if i + 1 < len(all_alignments):
            next_start_word = all_alignments[i + 1][1]
            if next_start_word < len(words):
                next_start_time = words[next_start_word]["start"]
        
        # 精调时间戳
        start_time, end_time, confidence = refine_timestamps(
            sent_info,
            words,
            start_word,
            end_word,
            prev_end_time,
            next_start_time,
            context
        )
        
        results.append(AlignmentResult(
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            method="dtw",
            matched_word_indices=list(range(start_word, end_word + 1))
        ))
        
        prev_end_time = end_time
    
    # 处理未对齐的句子（fallback）
    aligned_sent_indices = {a[0] for a in all_alignments}
    for sent_idx, sent_info in enumerate(sentence_infos):
        if sent_idx not in aligned_sent_indices:
            # 使用线性插值
            expected_duration = context.avg_duration * max(len(sent_info.tokens), 1)
            start_time = prev_end_time
            end_time = start_time + expected_duration
            
            results.append(AlignmentResult(
                start_time=start_time,
                end_time=end_time,
                confidence=0.2,
                method="fallback",
                matched_word_indices=[]
            ))
            
            prev_end_time = end_time
    
    # 按开始时间排序
    results.sort(key=lambda r: r.start_time)
    
    # 6. 后处理
    results = apply_post_processing(results, words)
    
    return results


# ============================================================================
# 文件处理 (File Processing)
# ============================================================================

def load_json(path: Path) -> Dict:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_segments(json_path: Path) -> tuple:
    """加载分段句子数据
    
    Returns:
        (sentences, speakers): 文本列表和对应的说话人标签列表
        speakers 可能为空列表（无说话人数据时）
    """
    data = load_json(json_path)
    sentences_raw = data.get("sentences", [])
    
    # 兼容字典格式和纯字符串格式
    sentences = []
    speakers = []
    has_speaker = False
    for item in sentences_raw:
        if isinstance(item, dict):
            text = item.get("text", "").strip()
            speaker = item.get("speaker")
            if speaker:
                has_speaker = True
        else:
            text = str(item).strip() if item else ""
            speaker = None
        if text:
            sentences.append(text)
            speakers.append(speaker)
    
    return sentences, speakers if has_speaker else []


def process_segment(
    seg_stem: str,
    project_dir: Path,
    offset: float,
    output_subs: List[srt.Subtitle],
    output_speakers: Optional[List] = None
):
    """处理单个音频段落"""
    seg_json = project_dir / "segments" / f"{seg_stem}_segments.json"
    asr_json = project_dir / "asr" / f"{seg_stem}.json"
    
    if not seg_json.exists():
        print(f"⚠️ 缺少分句文件: {seg_json}")
        return
    if not asr_json.exists():
        print(f"⚠️ 缺少ASR文件: {asr_json}")
        return
    
    # 加载数据
    sentences, speakers = load_segments(seg_json)
    asr_data = load_json(asr_json)
    words = clean_asr_words(extract_asr_words(asr_data))
    
    if not sentences:
        print(f"⚠️ 无句子数据: {seg_stem}")
        return
    if not words:
        print(f"⚠️ ASR词为空: {seg_stem}")
        return
    
    print(f"  处理段落 {seg_stem}: {len(sentences)} 句, {len(words)} 词")
    
    # 执行对齐
    alignments = align_sentences_to_words(sentences, words)
    
    # 构建字幕（SRT 内容保持纯净，speaker 信息单独存储）
    for sent_idx, (sentence, alignment) in enumerate(zip(sentences, alignments)):
        start = timedelta(seconds=alignment.start_time + offset)
        end = timedelta(seconds=alignment.end_time + offset)
        
        sub = srt.Subtitle(
            index=len(output_subs) + 1,
            start=start,
            end=end,
            content=sentence.strip()
        )
        output_subs.append(sub)
        
        # 记录对应的 speaker（用于 TXT 输出）
        if output_speakers is not None:
            speaker = speakers[sent_idx] if speakers and sent_idx < len(speakers) else None
            output_speakers.append(speaker)
        
        # 低置信度警告
        if alignment.confidence < 0.4:
            print(f"    🔄 低置信度 ({alignment.confidence:.2f}): {sentence[:50]}...")


def create_english_srt(stem: str, project_dir: Path | None = None):
    """???????????????????????RT"""
    project_dir = project_dir if project_dir is not None else WORKDIR / stem
    
    if not project_dir.exists():
        print(f"❌ 项目目录不存在: {project_dir}")
        return
    
    # 输出路径
    out_srt = project_dir / f"[EN]-{stem}.srt"
    out_txt = project_dir / f"[EN]-{stem}.txt"
    
    if out_srt.exists() and out_txt.exists():
        print(f"跳过: {out_srt.name} 已存在")
        return
    
    # 加载segments_map
    map_path = project_dir / "chunks" / "segments_map.json"
    if not map_path.exists():
        print(f"❌ 未找到 segments_map.json: {map_path}")
        return
    
    segment_map = load_json(map_path)
    if not isinstance(segment_map, list):
        print(f"❌ segments_map.json 格式错误")
        return
    
    # 按时间排序
    segment_map = sorted(segment_map, key=lambda m: float(m.get("start", 0.0)))
    
    print(f"\n=== 时间戳匹配 (DTW): {stem} ===")
    print(f"  段落数: {len(segment_map)}")
    
    # 处理每个段落
    all_subs: List[srt.Subtitle] = []
    all_speakers: List = []
    
    for meta in segment_map:
        seg_file = meta.get("file")
        offset = float(meta.get("start", 0.0))
        
        if not seg_file:
            continue
        
        seg_stem = Path(seg_file).stem
        process_segment(seg_stem, project_dir, offset, all_subs, all_speakers)
    
    if not all_subs:
        print("❌ 未生成任何字幕")
        return
    
    # 重新编号
    for idx, sub in enumerate(all_subs, 1):
        sub.index = idx
    
    # 写入 SRT（纯净，不含 speaker 标签）
    out_srt.write_text(srt.compose(all_subs), encoding="utf-8")
    print(f"✅ 写入 SRT: {out_srt}")
    
    # 写入 TXT（含 speaker 标签，仅在说话人切换时标注）
    txt_lines = []
    last_speaker = None
    for idx, sub in enumerate(all_subs):
        line = sub.content
        speaker = all_speakers[idx] if idx < len(all_speakers) else None
        if speaker and speaker != last_speaker:
            line = f"[{speaker}] {line}"
            last_speaker = speaker
        txt_lines.append(line)
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"✅ 写入 TXT: {out_txt}")
    
    # 保存 speakers.json（供中文 TXT 生成时使用）
    has_any_speaker = any(s for s in all_speakers if s)
    if has_any_speaker:
        speakers_json_path = project_dir / "speakers.json"
        with open(speakers_json_path, "w", encoding="utf-8") as f:
            json.dump(all_speakers, f, ensure_ascii=False)
        print(f"✅ 写入 speakers.json: {speakers_json_path}")
    
    print(f"✅ 完成，共 {len(all_subs)} 条字幕")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="时间戳匹配 (DTW全局对齐版本)"
    )
    parser.add_argument(
        "--stem",
        required=True,
        help="项目名称 (用于生成 [EN]-<stem>.srt)"
    )
    args = parser.parse_args()
    
    create_english_srt(args.stem)


if __name__ == "__main__":
    main()
