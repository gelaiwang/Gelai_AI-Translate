# _4_gpt_segment_translate.py

import re
import srt # type: ignore
import time
import os
from pathlib import Path
import argparse
import logging
import glob
from typing import Optional

import config

from rich import print as rprint
from rich.panel import Panel

# [修改] 导入 OpenAI 客户端
from openai import OpenAI

# --- Configuration ---
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

from core.google_llm import (
    build_google_text_client,
    generate_google_text,
    is_google_service,
)
from core.translation_context import (
    detect_series_marker,
    ensure_series_registry_entry,
    resolve_translation_context_text,
)
from core.translation_text import (
    DEFAULT_CONTEXT_BLOCK,
    augment_prompt_with_hints,
    chunk_subtitles_by_line_limit,
    clean_llm_output,
    create_sliding_window_chunks,
    escape_braces,
    extract_lines_from_output,
    merge_overlapping_translations,
    prepare_lines_for_batch,
    preprocess_lines_with_numbers,
    reconstruct_subtitles_from_lines,
    record_batch_error_hint,
    subtitle_to_lines,
    validate_line_count,
    write_plain_txt,
)
from core.translation_runtime import (
    build_primary_llm_client,
    determine_provider_plan,
    resolve_client_for_provider,
    sanitize_model_for_filename,
    short_stem_for_filename,
)
from core.translation_batches import execute_batch_translation
from core.translation_prompts import load_local_stage_prompts, load_prompt_template
from core.translation_validation import validate_llm_translation

SUPPORTED_SERVICES = ["grok", "gemini", "vertex", "local", "deepseek"]
ACTIVE_API_KEY: str | None = None
ACTIVE_MODEL_NAME: str | None = None
SERVICE_DISPLAY_NAME = ""
ACTIVE_TEMPERATURE = 0.5
grok_config_ok = False
gemini_config_ok = False
vertex_config_ok = False
local_config_ok = False
deepseek_config_ok = False
_RUNTIME_INITIALIZED = False

LOG_FILE_PATH = (WORKDIR if isinstance(WORKDIR, Path) else Path.cwd()) / "translation_log.txt"
logger = logging.getLogger(__name__)


def _normalize_temperature(value: object, *, default: float, label: str) -> float:
    if isinstance(value, (float, int)) and 0 <= float(value) <= 2:
        return float(value)
    rprint(
        Panel(
            f"[yellow]警告: 配置中的 `{label}` 未正确配置或超出范围. 已重置为默认值: {default}[/yellow]",
            title="配置警告",
            border_style="yellow",
        )
    )
    return default


def _configure_logger(log_path: Path) -> None:
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_path
    logger.setLevel(logging.WARNING)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _ensure_runtime_initialized() -> None:
    global ACTIVE_API_KEY, ACTIVE_MODEL_NAME, SERVICE_DISPLAY_NAME, ACTIVE_TEMPERATURE
    global grok_config_ok, gemini_config_ok, vertex_config_ok, local_config_ok, deepseek_config_ok
    global _RUNTIME_INITIALIZED

    if _RUNTIME_INITIALIZED:
        return

    if not WORKDIR or not isinstance(WORKDIR, Path):
        raise RuntimeError("配置中的 `WORKDIR` 未正确配置或不是有效的 Path 对象。")
    if TRANSLATION_SERVICE not in SUPPORTED_SERVICES:
        raise RuntimeError(
            f"配置中的 `TRANSLATION_SERVICE` ('{TRANSLATION_SERVICE}') 无效. 请选择 {SUPPORTED_SERVICES}。"
        )

    cloud_temperature = _normalize_temperature(
        LLM_TEMPERATURE_TRANSLATE_CLOUD,
        default=0.5,
        label="LLM_TEMPERATURE_TRANSLATE_CLOUD",
    )
    local_temperature = _normalize_temperature(
        LLM_TEMPERATURE_TRANSLATE_LOCAL,
        default=0.5,
        label="LLM_TEMPERATURE_TRANSLATE_LOCAL",
    )

    grok_config_ok = bool(GROK_API_KEY and GROK_MODEL_NAME)
    gemini_config_ok = bool(GEMINI_API_KEY and GEMINI_MODEL_NAME)
    vertex_config_ok = bool(VERTEX_PROJECT and VERTEX_MODEL_NAME)
    local_config_ok = bool(LOCAL_API_BASE_URL and LOCAL_MODEL_NAME)
    deepseek_config_ok = bool(DEEPSEEK_API_KEY and DEEPSEEK_MODEL_NAME)

    ACTIVE_API_KEY = None
    ACTIVE_MODEL_NAME = None
    SERVICE_DISPLAY_NAME = ""
    ACTIVE_TEMPERATURE = cloud_temperature

    if TRANSLATION_SERVICE == "grok":
        SERVICE_DISPLAY_NAME = "Grok"
        if not grok_config_ok:
            raise RuntimeError("当 TRANSLATION_SERVICE 为 'grok' 时，`GROK_API_KEY` 或 `GROK_MODEL_NAME` 未设置。")
        ACTIVE_API_KEY = GROK_API_KEY
        ACTIVE_MODEL_NAME = GROK_MODEL_NAME
    elif TRANSLATION_SERVICE == "gemini":
        SERVICE_DISPLAY_NAME = "Gemini"
        if not gemini_config_ok:
            raise RuntimeError("当 TRANSLATION_SERVICE 为 'gemini' 时，`GEMINI_API_KEY` 或 `GEMINI_MODEL_NAME` 未设置。")
        ACTIVE_API_KEY = GEMINI_API_KEY
        ACTIVE_MODEL_NAME = GEMINI_MODEL_NAME
        if not grok_config_ok:
            rprint(Panel("[yellow]警告: 未找到有效的 Grok 配置。Gemini 翻译失败时的 Grok 兜底功能将不可用。[/yellow]", title="配置警告", border_style="yellow"))
    elif TRANSLATION_SERVICE == "vertex":
        SERVICE_DISPLAY_NAME = "Google Vertex AI"
        if not vertex_config_ok:
            raise RuntimeError("当 TRANSLATION_SERVICE 为 'vertex' 时，`GOOGLE_CLOUD_PROJECT` 或 `VERTEX_MODEL_NAME` 未设置。")
        ACTIVE_API_KEY = VERTEX_PROJECT
        ACTIVE_MODEL_NAME = VERTEX_MODEL_NAME
        if not grok_config_ok:
            rprint(Panel("[yellow]警告: 未找到有效的 Grok 配置。Vertex 翻译失败时的 Grok 兜底功能将不可用。[/yellow]", title="配置警告", border_style="yellow"))
    elif TRANSLATION_SERVICE == "local":
        SERVICE_DISPLAY_NAME = "Local LLM"
        if not local_config_ok:
            raise RuntimeError("当 TRANSLATION_SERVICE 为 'local' 时，`LOCAL_API_BASE_URL` 或 `LOCAL_MODEL_NAME` 未设置。")
        ACTIVE_API_KEY = LOCAL_API_KEY
        ACTIVE_MODEL_NAME = LOCAL_MODEL_NAME
        ACTIVE_TEMPERATURE = local_temperature
    elif TRANSLATION_SERVICE == "deepseek":
        SERVICE_DISPLAY_NAME = "DeepSeek"
        if not deepseek_config_ok:
            raise RuntimeError("当 TRANSLATION_SERVICE 为 'deepseek' 时，`DEEPSEEK_API_KEY` 或 `DEEPSEEK_MODEL_NAME` 未设置。")
        ACTIVE_API_KEY = DEEPSEEK_API_KEY
        ACTIVE_MODEL_NAME = DEEPSEEK_MODEL_NAME
        if not grok_config_ok:
            rprint(Panel("[yellow]警告: 未找到有效的 Grok 配置。DeepSeek 翻译失败时的 Grok 兜底功能将不可用。[/yellow]", title="配置警告", border_style="yellow"))

    _configure_logger(WORKDIR / "translation_log.txt")
    _RUNTIME_INITIALIZED = True


def _active_model_name() -> str:
    _ensure_runtime_initialized()
    if not ACTIVE_MODEL_NAME:
        raise RuntimeError("LLM 模型名称未初始化。")
    return ACTIVE_MODEL_NAME


def _build_primary_client(model_name: str | None = None) -> object:
    _ensure_runtime_initialized()
    return build_primary_llm_client(
        translation_service=TRANSLATION_SERVICE,
        active_api_key=ACTIVE_API_KEY,
        active_model_name=_active_model_name(),
        local_api_base_url=LOCAL_API_BASE_URL,
        local_timeout=LOCAL_TIMEOUT,
        deepseek_api_base_url=DEEPSEEK_API_BASE_URL,
        model_name=model_name,
    )

prompt_template_content = load_prompt_template(
    PROMPT_TEMPLATE_FILE,
    required_placeholders=["{srt_content_for_llm}", "{video_context}"],
)
translation_context_template_content = load_prompt_template(
    TRANSLATION_CONTEXT_TEMPLATE_FILE,
    required_placeholders=["{metadata_block}", "{english_srt_excerpt}"],
)

LOCAL_STAGE1_PROMPT, LOCAL_STAGE2_PROMPT = load_local_stage_prompts(
    Path(__file__).resolve().parent.parent / "config" / "prompts"
)

# --- Two-Stage Translation Helper Functions ---
# 使用行号模式替代标签模式，更简单可靠
def _generate_series_name_text(prompt: str) -> str:
    model_name = _active_model_name()
    client_obj = _build_primary_client(model_name)

    temperature = min(0.3, max(float(ACTIVE_TEMPERATURE), 0.1))

    if TRANSLATION_SERVICE == "local":
        response = client_obj.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=LOCAL_TOP_P,
            extra_body={"think": not LOCAL_SHOW_THINKING, "num_ctx": LOCAL_CONTEXT_LENGTH},
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif TRANSLATION_SERVICE in {"grok", "deepseek"}:
        response = client_obj.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif is_google_service(TRANSLATION_SERVICE):
        text = generate_google_text(client_obj, prompt, temperature)
    else:
        raise RuntimeError(f"不支持的服务类型: {TRANSLATION_SERVICE}")
    return text.strip()

def _generate_translation_context_text(prompt: str) -> str:
    model_name = _active_model_name()
    client_obj = _build_primary_client(model_name)

    temperature = min(0.4, max(float(ACTIVE_TEMPERATURE), 0.1))

    if TRANSLATION_SERVICE == "local":
        response = client_obj.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=LOCAL_TOP_P,
            extra_body={"think": not LOCAL_SHOW_THINKING, "num_ctx": LOCAL_CONTEXT_LENGTH},
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif TRANSLATION_SERVICE in {"grok", "deepseek"}:
        response = client_obj.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
    elif is_google_service(TRANSLATION_SERVICE):
        text = generate_google_text(client_obj, prompt, temperature)
    else:
        raise RuntimeError(f"不支持的服务类型: {TRANSLATION_SERVICE}")
    return text.strip()


def _create_openai_text(
    client: object,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float | None,
    show_thinking: bool | None,
    context_length: int | None,
) -> str:
    extra_body = None
    if top_p is not None and show_thinking is not None and context_length is not None:
        extra_body = {"think": not show_thinking, "num_ctx": context_length}
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip() if response.choices else ""


def _create_grok_client() -> object:
    return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

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
    _ensure_runtime_initialized()
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
    numbered_lines = preprocess_lines_with_numbers(all_lines)
    
    # Step 2: 滑动窗口分块
    chunks = create_sliding_window_chunks(
        numbered_lines,
        LOCAL_WINDOW_SIZE,
        LOCAL_OVERLAP
    )
    rprint(f"[cyan]分为 {len(chunks)} 个翻译块[/cyan]")
    
    # Prepare context
    context_block = video_context_text.strip() or DEFAULT_CONTEXT_BLOCK
    escaped_context = escape_braces(context_block)
    
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
        stage1_lines: list[str] = []
        last_stage1_lines: list[str] = []
        
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
                cleaned_output = clean_llm_output(raw_output)
                
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
                extracted_lines = extract_lines_from_output(cleaned_output)
                last_stage1_lines = extracted_lines
                success, msg = validate_line_count(extracted_lines, expected_count)
                
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
            stage1_lines = list(last_stage1_lines)
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
                    cleaned_output = clean_llm_output(raw_output)
                    
                    # Save raw output
                    backup_path = raw_output_dir / f"{input_stem}_chunk_{chunk_idx+1:03d}_stage2_attempt{attempt+1}.txt"
                    backup_path.write_text(cleaned_output, encoding="utf-8")
                    raw_llm_responses.append({
                        "chunk": chunk_idx + 1,
                        "stage": 2,
                        "attempt": attempt + 1,
                        "file": str(backup_path)
                    })
                     
                    extracted_lines = extract_lines_from_output(cleaned_output)
                    success, msg = validate_line_count(extracted_lines, expected_count)

                    if success:
                        stage2_lines = extracted_lines
                        rprint(f"  ✅ Stage2 块 #{chunk_idx + 1} 成功")
                        break
                    else:
                        rprint(
                            f"  [yellow]⚠️ Stage2 块 #{chunk_idx + 1} 校验失败 "
                            f"(尝试 {attempt+1}): {msg}[/yellow]"
                        )
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(1)
                        else:
                            rprint(f"  [yellow]⚠️ Stage2 块 #{chunk_idx + 1} 最终失败，使用 Stage1 结果[/yellow]")

                except Exception as e:
                    rprint(
                        f"  [yellow]⚠️ Stage2 块 #{chunk_idx + 1} 异常 "
                        f"(尝试 {attempt+1}): {e}[/yellow]"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2)
                    else:
                        rprint(f"  [yellow]⚠️ Stage2 块 #{chunk_idx + 1} 最终失败，使用 Stage1 结果[/yellow]")
            
        chunk_results.append((start, end, stage2_lines))
        time.sleep(0.5)  # Rate limiting
    
    # Step 6: 合并重叠区域
    rprint("[cyan]合并翻译结果...[/cyan]")
    merged_translations = merge_overlapping_translations(chunk_results, total_lines)
    
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
    _ensure_runtime_initialized()
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

    try:
        llm_client = build_primary_llm_client(
            translation_service=TRANSLATION_SERVICE,
            active_api_key=ACTIVE_API_KEY,
            active_model_name=ACTIVE_MODEL_NAME,
            local_api_base_url=LOCAL_API_BASE_URL,
            local_timeout=LOCAL_TIMEOUT,
            deepseek_api_base_url=DEEPSEEK_API_BASE_URL,
        )
    except Exception as e:  # noqa: BLE001
        rprint(f"[bold red]初始化 {SERVICE_DISPLAY_NAME} 客户端失败: {e}[/bold red]")
        return [], [], []

    client_cache: dict[tuple[str, str], object] = {}
    client_cache[(TRANSLATION_SERVICE, ACTIVE_MODEL_NAME)] = llm_client
    provider_plan, max_retry_rounds = determine_provider_plan(
        translation_service=TRANSLATION_SERVICE,
        active_model_name=ACTIVE_MODEL_NAME,
        service_display_name=SERVICE_DISPLAY_NAME,
        gemini_config_ok=gemini_config_ok,
    )

    return execute_batch_translation(
        english_srt_objects=english_srt_objects,
        input_stem=input_stem,
        batch_size=batch_size,
        raw_output_dir=raw_output_dir,
        video_context_text=video_context_text,
        translation_service=TRANSLATION_SERVICE,
        active_model_name=ACTIVE_MODEL_NAME,
        service_display_name=SERVICE_DISPLAY_NAME,
        active_temperature=ACTIVE_TEMPERATURE,
        local_top_p=LOCAL_TOP_P,
        local_show_thinking=LOCAL_SHOW_THINKING,
        local_context_length=LOCAL_CONTEXT_LENGTH,
        prompt_template_content=prompt_template_content,
        default_context_block=DEFAULT_CONTEXT_BLOCK,
        client_cache=client_cache,
        provider_plan=provider_plan,
        max_retry_rounds=max_retry_rounds,
        grok_config_ok=grok_config_ok,
        grok_model_name=GROK_MODEL_NAME,
        llm_temperature_translate_cloud=LLM_TEMPERATURE_TRANSLATE_CLOUD,
        resolve_client_for_provider=resolve_client_for_provider,
        sanitize_model_for_filename=sanitize_model_for_filename,
        short_stem_for_filename=short_stem_for_filename,
        generate_google_text=generate_google_text,
        print_info=lambda message: rprint(f"[cyan]{message}[/cyan]"),
        print_warning=lambda message: rprint(f"[yellow]{message}[/yellow]"),
        print_error=lambda message: rprint(f"[red]{message}[/red]"),
        print_panel=rprint,
        create_grok_client=_create_grok_client,
        create_openai_text=_create_openai_text,
    )


def generate_and_translate_srt(input_stem: str, batch_size: int, project_dir: Optional[Path] = None):
    _ensure_runtime_initialized()
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
        file_name=TRANSLATION_CONTEXT_FILE_NAME,
        force_regenerate=TRANSLATION_CONTEXT_FORCE_REGENERATE,
        enabled=TRANSLATION_CONTEXT_ENABLED,
        default_context=DEFAULT_CONTEXT_BLOCK,
        source_max_chars=TRANSLATION_CONTEXT_SOURCE_MAX_CHARS,
        template_content=translation_context_template_content,
        escape_braces=escape_braces,
        generate_translation_context_text=_generate_translation_context_text,
        generate_series_name_text=_generate_series_name_text,
        print_info=lambda message: rprint(f"[green]{message}[/green]"),
        print_warning=lambda message: rprint(f"[yellow]{message}[/yellow]"),
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
        write_plain_txt(chinese_srt_objects, chinese_txt_path)
        rprint(f"[bold green]🎉 最终中文字幕已保存至: {chinese_srt_path} ({len(chinese_srt_objects)} 条字幕)[/bold green]")
        rprint(f"[green]• 中文TXT已写入: {chinese_txt_path}[/green]")
    except Exception as e:
        rprint(f"[red]写入中文字幕文件失败: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description=f"使用 LLM API 将英文 SRT 翻译为中文 SRT。")
    parser.add_argument("--stem", required=True, help="正在处理的文件的基本名称（stem），用于命名SRT文件。")
    parser.add_argument("--batch_size", type=int, default=getattr(config, 'DEFAULT_BATCH_SIZE', 50), help="每次请求允许的英文字幕最大行数。")

    args = parser.parse_args()

    try:
        _ensure_runtime_initialized()
        desc_text = f"使用 {SERVICE_DISPLAY_NAME} API 将英文 SRT 翻译为中文 SRT。"
        if TRANSLATION_SERVICE in {'gemini', 'vertex'} and grok_config_ok:
            desc_text += " 如果主服务失败，将使用 Grok 作为兜底。"
        parser.description = desc_text
        generate_and_translate_srt(args.stem, args.batch_size)
    except RuntimeError as exc:
        rprint(Panel(f"[bold red]错误: {exc}[/bold red]", title="配置错误", border_style="red"))
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()


