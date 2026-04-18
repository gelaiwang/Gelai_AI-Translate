from __future__ import annotations

import srt  # type: ignore
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table


def validate_llm_translation(
    original_english_subs: list[srt.Subtitle],
    translated_chinese_subs: list[srt.Subtitle] | None,
    input_stem: str,
) -> bool:
    rprint(Panel("开始验证翻译结果 (仅中文)...", title="翻译验证", border_style="magenta"))

    if translated_chinese_subs is None:
        error_msg = "主要错误: 翻译步骤未能成功返回字幕列表 (结果为 None)。验证中止。"
        rprint(f"[bold red]{error_msg}[/bold red]")
        return False

    if not isinstance(translated_chinese_subs, list):
        error_msg = f"主要错误: 翻译步骤返回的不是列表类型 (类型: {type(translated_chinese_subs)}). 验证中止。"
        rprint(f"[bold red]{error_msg}[/bold red]")
        return False

    overall_valid = True
    issues_found = 0

    if len(original_english_subs) != len(translated_chinese_subs):
        overall_valid = False
        issues_found += 1
        rprint(f"[bold yellow]⚠️ 主要警告: 原始字幕数 ({len(original_english_subs)}) 与最终翻译后字幕数 ({len(translated_chinese_subs)}) 不匹配！这可能是因为有批次永久失败。[/bold yellow]")

    validation_table = Table(title="翻译验证详情（仅显示问题项）")
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
            content_lines = translated_sub.content.strip().split("\n")
            original_content_lines_count = len(original_sub.content.strip().split("\n"))
            expected_lines = original_content_lines_count if original_content_lines_count > 1 else 1

            if len(content_lines) != expected_lines:
                issue = (
                    f"行数错误: 期望 {expected_lines} 行，得到 {len(content_lines)} 行。 "
                    f"内容: '{translated_sub.content[:80].replace(chr(10), '<NL>')}{'...' if len(translated_sub.content) > 80 else ''}'"
                )
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
                    issue_desc.split(":", 1)[1].strip() if ":" in issue_desc else issue_desc,
                )

    if issues_found > 0:
        rprint(validation_table)
        rprint(f"\n[bold yellow]⚠️ 验证发现 {issues_found} 个问题。[/bold yellow]")
    else:
        rprint("[bold green]✅ 所有成功翻译的字幕均通过基本格式和内容验证！[/bold green]")
    return overall_valid
