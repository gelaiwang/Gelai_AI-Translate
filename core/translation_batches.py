from __future__ import annotations

import glob
import re
import time
from pathlib import Path
from typing import Callable

import srt  # type: ignore
from rich.panel import Panel

from core.google_llm import is_google_service
from core.translation_text import (
    augment_prompt_with_hints,
    chunk_subtitles_by_line_limit,
    escape_braces,
    prepare_lines_for_batch,
    reconstruct_subtitles_from_lines,
    record_batch_error_hint,
)


def execute_batch_translation(
    *,
    english_srt_objects: list[srt.Subtitle],
    input_stem: str,
    batch_size: int,
    raw_output_dir: Path,
    video_context_text: str,
    translation_service: str,
    active_model_name: str,
    service_display_name: str,
    active_temperature: float,
    local_top_p: float,
    local_show_thinking: bool,
    local_context_length: int,
    prompt_template_content: str,
    default_context_block: str,
    client_cache: dict[tuple[str, str], object],
    provider_plan: list[tuple[str, str, str]],
    max_retry_rounds: int,
    grok_config_ok: bool,
    grok_model_name: str,
    llm_temperature_translate_cloud: float,
    resolve_client_for_provider: Callable[[dict[tuple[str, str], object], str, str], object],
    sanitize_model_for_filename: Callable[[str], str],
    short_stem_for_filename: Callable[[str, int], str],
    generate_google_text: Callable[[object, str, float], str],
    print_info: Callable[[str], None],
    print_warning: Callable[[str], None],
    print_error: Callable[[str], None],
    print_panel: Callable[[Panel], None],
    create_grok_client: Callable[[], object],
    create_openai_text: Callable[[object, str, str, float, float | None, bool | None, int | None], str],
) -> tuple[list[srt.Subtitle], list[dict], list[tuple[int, list[srt.Subtitle]]]]:
    context_block = video_context_text.strip() or default_context_block
    escaped_context_block = escape_braces(context_block)

    successful_batches: dict[int, list[srt.Subtitle]] = {}
    raw_llm_responses: list[dict] = []
    batch_error_hints: dict[int, list[str]] = {}
    batch_line_cache: dict[int, tuple[list[str], list[int]]] = {}

    print_info(f"...扫描目录 '{raw_output_dir.name}' 检查已完成的翻译批次...")
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
                print_warning(f"警告: 加载已有批次文件 '{Path(f_path).name}' 失败: {e}，将重新翻译此批次。")

    if loaded_count > 0:
        print_info(f"✅ 成功加载 {loaded_count} 个已存在的翻译批次。")
    else:
        print_info("...未找到已完成的翻译批次，将从头开始。")

    max_lines_per_batch = max(1, batch_size)
    subtitle_batches = chunk_subtitles_by_line_limit(english_srt_objects, max_lines_per_batch)
    all_batches: list[tuple[int, list[srt.Subtitle]]] = [(i, batch) for i, batch in enumerate(subtitle_batches)]

    for idx, subtitles in all_batches:
        batch_line_cache[idx] = prepare_lines_for_batch(subtitles)

    all_batches_to_process = [batch for batch in all_batches if batch[0] not in successful_batches]
    max_attempts_per_batch = 3

    if not all_batches_to_process:
        print_panel(Panel(f"所有 {len(all_batches)} 个批次均已在之前的运行中翻译完成。直接进入整合步骤。", title=f"{service_display_name} 翻译任务", border_style="green"))
    else:
        print_panel(Panel(f"总共 {len(all_batches)} 个批次（单批不超过 {max_lines_per_batch} 行英文），需要处理 {len(all_batches_to_process)} 个新批次，送往 {service_display_name} ({active_model_name}) 进行翻译...", title=f"{service_display_name} 翻译任务", border_style="blue"))

    current_round = 1
    while current_round <= max_retry_rounds and all_batches_to_process:
        if current_round > 1:
            print_panel(Panel(f"开始第 {current_round}/{max_retry_rounds} 轮重试，处理剩余 {len(all_batches_to_process)} 个失败批次...", title="重试流程", style="bold yellow"))

        round_failed_batches = all_batches_to_process

        for provider_service, provider_model_name, provider_display_name in provider_plan:
            if not round_failed_batches:
                break

            current_model_label = sanitize_model_for_filename(provider_model_name)
            current_service_label = sanitize_model_for_filename(provider_service)
            failed_in_provider: list[tuple[int, list[srt.Subtitle]]] = []

            for batch_index, batch_to_translate in round_failed_batches:
                if not batch_to_translate:
                    continue

                batch_lines, line_counts = batch_line_cache.get(batch_index, prepare_lines_for_batch(batch_to_translate))
                batch_line_cache[batch_index] = (batch_lines, line_counts)
                lines_payload = "\n".join(batch_lines)
                escaped_batch_text = escape_braces(lines_payload)
                base_prompt = prompt_template_content.format(
                    srt_content_for_llm=escaped_batch_text,
                    video_context=escaped_context_block,
                )
                expected_line_count = len(batch_lines)

                print_info(
                    f"🌐 正在翻译批次 #{batch_index + 1} (字幕 {len(batch_to_translate)} 条 / 英文行 {len(batch_lines)}) "
                    f"使用 {provider_display_name} / {provider_model_name}..."
                )

                batch_successfully_processed = False
                for attempt in range(max_attempts_per_batch):
                    try:
                        current_client = resolve_client_for_provider(client_cache, provider_service, provider_model_name)
                        current_prompt = augment_prompt_with_hints(base_prompt, batch_error_hints.get(batch_index))
                        if provider_service == "local":
                            translated_plain_text = create_openai_text(
                                current_client,
                                provider_model_name,
                                current_prompt,
                                active_temperature,
                                local_top_p,
                                local_show_thinking,
                                local_context_length,
                            )
                        elif provider_service in {"grok", "deepseek"}:
                            translated_plain_text = create_openai_text(
                                current_client,
                                provider_model_name,
                                current_prompt,
                                active_temperature,
                                None,
                                None,
                                None,
                            )
                        elif is_google_service(provider_service):
                            translated_plain_text = generate_google_text(current_client, current_prompt, active_temperature)
                        else:
                            raise RuntimeError(f"Unsupported provider service: {provider_service}")

                        short_input_stem = short_stem_for_filename(input_stem, 48)
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

                        current_batch_translated_subs = reconstruct_subtitles_from_lines(
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
                        print_info(
                            f"✅ 批次 #{batch_index + 1}, 第 {attempt+1} 次尝试"
                            f"({provider_display_name} / {provider_model_name}) 翻译成功！"
                        )
                        time.sleep(1)
                        break

                    except Exception as e:
                        resource_exhausted = "RESOURCE_EXHAUSTED" in str(e)
                        error_msg = (
                            f"批次 #{batch_index + 1}, 第 {attempt+1}/{max_attempts_per_batch} 次尝试"
                            f"({provider_display_name} / {provider_model_name}) 失败: {e}"
                        )
                        print_warning(f"⚠️ {error_msg}")
                        record_batch_error_hint(batch_error_hints, batch_index, e)
                        if resource_exhausted:
                            print_warning("⚠️ 发现 Vertex 资源耗尽，立即切换后续提供方")
                            short_input_stem = short_stem_for_filename(input_stem, 48)
                            failed_input_path = raw_output_dir / (
                                f"{short_input_stem}_b{batch_index+1:03d}_input_failed_{current_service_label}.txt"
                            )
                            with open(failed_input_path, "w", encoding="utf-8") as f:
                                f.write(lines_payload)
                            break
                        if attempt < max_attempts_per_batch - 1:
                            wait_time = 2 + attempt
                            time.sleep(wait_time)
                            print_info(f"    {wait_time}秒后重试...")
                        else:
                            final_error_msg = (
                                f"批次 #{batch_index+1} 在 {max_attempts_per_batch} 次尝试后仍失败"
                                f"（当前模型: {provider_model_name}）。将其移至下一阶段重试。"
                            )
                            print_error(f"❌ {final_error_msg}")
                            short_input_stem = short_stem_for_filename(input_stem, 48)
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

    if translation_service in {"gemini", "vertex", "deepseek"} and permanently_failed_batches:
        if not grok_config_ok:
            print_panel(Panel("[yellow]翻译后仍有失败批次，但 Grok 配置缺失，无法启动兜底程序。[/yellow]", title="Grok 兜底跳过", border_style="yellow"))
        else:
            print_panel(Panel(f"...主服务仍失败 {len(permanently_failed_batches)} 个批次，启动 Grok 作为最终兜底...", title="Fallback to Grok", style="bold red"))
            try:
                grok_client = create_grok_client()
                still_failed_after_fallback = []
                max_fallback_attempts = 2

                for batch_index, batch_to_translate in permanently_failed_batches:
                    print_info(f" G-Fallback 正在尝试翻译批次 #{batch_index + 1} 使用 Grok...")
                    batch_lines, line_counts = batch_line_cache.get(batch_index, prepare_lines_for_batch(batch_to_translate))
                    batch_line_cache[batch_index] = (batch_lines, line_counts)
                    lines_payload = "\n".join(batch_lines)
                    escaped_batch_text = escape_braces(lines_payload)
                    base_prompt = prompt_template_content.format(
                        srt_content_for_llm=escaped_batch_text,
                        video_context=escaped_context_block,
                    )

                    fallback_success = False
                    for attempt in range(max_fallback_attempts):
                        try:
                            current_prompt = augment_prompt_with_hints(base_prompt, batch_error_hints.get(batch_index))
                            translated_plain_text = create_openai_text(
                                grok_client,
                                grok_model_name,
                                current_prompt,
                                llm_temperature_translate_cloud,
                                None,
                                None,
                                None,
                            )
                            backup_path = raw_output_dir / f"{input_stem}_batch_{batch_index+1:03d}_GrokFallback_attempt_{attempt+1}.txt"
                            with open(backup_path, "w", encoding="utf-8") as f:
                                f.write(translated_plain_text)
                            raw_llm_responses.append({"batch_index": batch_index, "attempt": f"grok_fallback_{attempt+1}", "raw_text_file": str(backup_path)})

                            if not translated_plain_text or not translated_plain_text.strip():
                                raise ValueError("Grok returned empty content.")

                            current_batch_translated_subs = reconstruct_subtitles_from_lines(
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
                            print_info(f"✅ G-Fallback: 批次 #{batch_index + 1} 使用 Grok 翻译成功！")
                            time.sleep(1)
                            break
                        except Exception as e:
                            print_warning(f"⚠️ G-Fallback: 批次 #{batch_index + 1}, 第 {attempt+1}/{max_fallback_attempts} 次尝试(Grok) 失败: {e}")
                            record_batch_error_hint(batch_error_hints, batch_index, e)
                            if attempt < max_fallback_attempts - 1:
                                time.sleep(2)

                    if not fallback_success:
                        still_failed_after_fallback.append((batch_index, batch_to_translate))

                permanently_failed_batches = still_failed_after_fallback
            except Exception as e:
                print_error(f"初始化 Grok 兜底客户端失败: {e}。兜底程序中止。")

    print_info("...翻译和重试流程结束，开始整合最终结果...")
    final_translated_subs: list[srt.Subtitle] = []
    sorted_successful_indices = sorted(successful_batches.keys())

    if len(sorted_successful_indices) != len(all_batches):
        if not permanently_failed_batches:
            print_warning(f"⚠️ 警告: 最终成功批次数 ({len(sorted_successful_indices)}) 与总批次数 ({len(all_batches)}) 不符。可能有些批次文件丢失或损坏。")

    for index in sorted_successful_indices:
        final_translated_subs.extend(successful_batches[index])

    final_translated_subs = [srt.Subtitle(index=idx, start=sub.start, end=sub.end, content=sub.content) for idx, sub in enumerate(final_translated_subs, start=1)]
    return final_translated_subs, raw_llm_responses, permanently_failed_batches
