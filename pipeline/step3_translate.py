# 2_json_ensrt_cnsrt.py (最终工作版 - "生成时校验"模式)

import argparse

import os
import sys
import time
from pathlib import Path
import shutil
import re
from datetime import timedelta
import json
import srt

from rich.console import Console
from rich.panel import Panel

try:
    from config import (
        WORKDIR,
        CLOUD_LINE_BATCH_SIZE, LOCAL_LINE_BATCH_SIZE,
        TRANSLATION_SERVICE,
    )
except ImportError:
    print("错误: 无法从 config 包导入必要配置。请检查 `config/settings.py`。")
    sys.exit(1)

try:
    from core.llm_segment import process_audio_segment
    from core.validate_segments import validate_segments_structure
    from core.timestamp_matcher import create_english_srt
    from core.llm_translate import (
        generate_and_translate_srt,
    )
except ImportError as e:
    print(f"错误: 导入模块失败: {e}")
    sys.exit(1)

console = Console()


def write_cn_txt_from_srt(cn_srt_path: Path, cn_txt_path: Path, console: Console):
    """
    从中文 SRT 生成纯文本版，每条字幕一行，格式与英文 TXT 对齐。
    如果项目目录中存在 speakers.json，则在说话人切换时添加标签。
    """
    try:
        subs = list(srt.parse(cn_srt_path.read_text(encoding="utf-8")))
        
        # 尝试加载 speaker 数据
        speakers = []
        speakers_json = cn_srt_path.parent / "speakers.json"
        if speakers_json.exists():
            try:
                with open(speakers_json, "r", encoding="utf-8") as f:
                    speakers = json.load(f)
            except Exception:
                pass
        
        last_speaker = None
        with open(cn_txt_path, "w", encoding="utf-8", newline="\n") as f:
            for idx, sub in enumerate(subs):
                clean = re.sub(r"[\\r\\n]+", " ", sub.content).strip()
                if clean:
                    # 在说话人切换时添加标签
                    speaker = speakers[idx] if idx < len(speakers) else None
                    if speaker and speaker != last_speaker:
                        clean = f"[{speaker}] {clean}"
                        last_speaker = speaker
                    f.write(clean + "\n")
        console.print(f"[green]已生成中文 TXT：{cn_txt_path.name}[/green]")
    except Exception as err:
        console.print(f"[yellow]⚠️ 中文 TXT 生成失败，请手动检查：{err}[/yellow]")


def srt_time_to_timedelta(time_str: str) -> timedelta:
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_str)
    if match:
        return timedelta(
            hours=int(match.group(1)),
            minutes=int(match.group(2)),
            seconds=int(match.group(3)),
            milliseconds=int(match.group(4)),
        )
    return timedelta(0)


def check_srt_timestamp_gaps(
    stem: str, project_dir: Path, workdir: Path, console: Console
):
    en_srt_path = project_dir / f"[EN]-{stem}.srt"
    if not en_srt_path.exists():
        return
    console.print(f"[cyan]-- 正在检查 '{en_srt_path.name}' 的时间戳连续性... --[/cyan]")
    try:
        with open(en_srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        subtitle_blocks = re.findall(
            r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)",
            content,
        )
        if not subtitle_blocks:
            console.print(
                f"[yellow]警告: 在 '{en_srt_path.name}' 中未找到任何字幕块，无法检查。[/yellow]"
            )
            return
        last_end_time = None
        for i, block in enumerate(subtitle_blocks):
            _, start_str, end_str, _ = block
            start_time = srt_time_to_timedelta(start_str)
            if last_end_time is not None:
                if (start_time - last_end_time).total_seconds() > 20:
                    console.print(
                        f"  [bold yellow]⚠ 在项目 '{stem}' 中发现一个超过20秒的大间隙！[/bold yellow]"
                    )
            last_end_time = srt_time_to_timedelta(end_str)
        console.print("[green]✅ 时间戳检查完成。[/green]")
    except Exception as e:
        console.print(f"[bold red]错误: 检查时间戳时发生意外错误: {e}[/bold red]")


def generate_english_srt_with_retries(stem: str, project_dir: Path) -> bool:
    console.print(
        Panel(
            f"处理项目 [bold cyan]{stem}[/bold cyan]：生成英文 SRT",
            title="[1/2] 生成英文轨道",
            expand=False,
        )
    )

    asr_dir = project_dir / "asr"
    segments_dir = project_dir / "segments"

    max_validation_retries = 3
    for attempt in range(max_validation_retries):
        console.print(
            f"\n[bold]第 {attempt + 1}/{max_validation_retries} 次尝试: 启动生成与即时校验流程...[/bold]"
        )

        segments_dir.mkdir(parents=True, exist_ok=True)

        all_asr_files = sorted(asr_dir.glob("segment_*.json"))
        if not all_asr_files:
            console.print(
                f"[yellow]警告: 在 {asr_dir} 中未找到 'segment_*.json' 文件。[/yellow]"
            )
            return False

        segments_to_process = []
        for json_file in all_asr_files:
            try:
                segment_id = int(json_file.stem.split("_")[1])
                output_path = segments_dir / f"segment_{segment_id:03d}_segments.json"
                if not output_path.exists():
                    segments_to_process.append(segment_id)
            except (IndexError, ValueError):
                console.print(
                    f"[yellow]警告: 无法从文件名解析 segment ID: {json_file.name}。[/yellow]"
                )

        if not segments_to_process:
            console.print(
                "[green]-- 所有音频切片的LLM分句文件已存在，直接进入最终验证步骤。 --[/green]"
            )
        else:
            console.print(
                f"[cyan]-- 需要处理 {len(segments_to_process)} 个新的音频切片... --[/cyan]"
            )
            max_passes = 5
            current_pass = 1
            while segments_to_process and current_pass <= max_passes:
                console.print(f"\n--- 第 {current_pass}/{max_passes} 轮处理 ---")
                failed_in_this_pass = []
                for segment_id in segments_to_process:
                    try:
                        success = process_audio_segment(
                            segment_id, asr_dir, segments_dir
                        )
                        if not success:
                            raise RuntimeError(
                                f"Segment {segment_id} 在多次尝试后最终处理失败。"
                            )

                        console.print(
                            f"[green]✅ Segment {segment_id} 已成功生成并通过内部校验。[/green]"
                        )

                    except Exception as e:
                        console.print(
                            f"[yellow]警告: 处理 Segment {segment_id} 时遇到问题，将在下一轮重试。详情: {e}[/yellow]"
                        )
                        failed_in_this_pass.append(segment_id)

                segments_to_process = failed_in_this_pass
                if segments_to_process:
                    current_pass += 1

            if segments_to_process:
                console.print(
                    f"[bold red]错误: LLM分句步骤有 {len(segments_to_process)} 个切片最终处理失败。[/bold red]"
                )
                return False

        console.print(
            "\n[green]✅ 所有音频切片均已成功生成并通过了各自的即时校验。[/green]"
        )

        console.print("[cyan]-- 正在进行最终全面验证... --[/cyan]")
        try:
            validation_ok = validate_segments_structure(segments_dir, asr_dir)
            if validation_ok:
                console.print("[bold green]🎉 最终全面验证成功！[/bold green]")
                break
            else:
                console.print(
                    f"[bold yellow]警告: 最终全面验证失败。将在 10 秒后清空并重试...[/bold yellow]"
                )
                for f in segments_dir.glob("*.json"):
                    f.unlink()
                time.sleep(10)
        except Exception as e:
            console.print(f"[bold red]错误: 最终验证脚本发生意外错误: {e}[/bold red]")
            return False
    else:
        console.print(
            f"[bold red]错误: 经过 {max_validation_retries} 次重试后，分句文件仍未通过最终验证。[/bold red]"
        )
        return False

    console.print("[cyan]-- 正在匹配时间戳并生成最终英文 SRT 文件... --[/cyan]")
    try:
        create_english_srt(stem, project_dir=project_dir)
        en_srt_path = project_dir / f"[EN]-{stem}.srt"
        en_txt_path = project_dir / f"[EN]-{stem}.txt"

        # Compatibility fallback for legacy output paths that may still exist.
        legacy_en_srt_path = WORKDIR / f"[EN]-{stem}.srt"
        legacy_en_txt_path = WORKDIR / f"[EN]-{stem}.txt"
        if not en_srt_path.exists() and legacy_en_srt_path.exists():
            console.print(f"[yellow]Legacy EN SRT output detected; moving into current project_dir: {legacy_en_srt_path} -> {en_srt_path}[/yellow]")
            shutil.move(str(legacy_en_srt_path), str(en_srt_path))
        if not en_txt_path.exists() and legacy_en_txt_path.exists():
            console.print(f"[yellow]Legacy EN TXT output detected; moving into current project_dir: {legacy_en_txt_path} -> {en_txt_path}[/yellow]")
            shutil.move(str(legacy_en_txt_path), str(en_txt_path))

        console.print(f"[green]Generated EN outputs: EN SRT={en_srt_path}; EN TXT={en_txt_path}[/green]")
        check_srt_timestamp_gaps(stem, project_dir, WORKDIR, console)
    except Exception as e:
        console.print(f"[bold red]错误: 生成英文 SRT 时发生意外崩溃: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        return False
    return True


def translate_to_chinese_srt(stem: str, project_dir: Path) -> bool:
    console.print(
        Panel(
            f"处理项目 [bold cyan]{stem}[/bold cyan]：翻译为中文 SRT",
            title="[2/2] 生成中文轨道",
            expand=False,
        )
    )
    console.print(f"[cyan]Current translation project_dir: {project_dir}[/cyan]")
    try:
        # 根据服务类型选择批次大小
        if TRANSLATION_SERVICE == "local":
            batch_size = LOCAL_LINE_BATCH_SIZE
        else:
            batch_size = CLOUD_LINE_BATCH_SIZE
        generate_and_translate_srt(
            input_stem=stem,
            batch_size=batch_size,
            project_dir=project_dir,
        )
        cn_srt_path = project_dir / f"[CN]-{stem}.srt"
        cn_txt_path = project_dir / f"[CN]-{stem}.txt"
        if not cn_srt_path.exists():
            raise FileNotFoundError("翻译脚本执行完毕，但未在预期路径生成中文SRT文件。")
        console.print(f"[green]Generated CN outputs: CN SRT={cn_srt_path}; CN TXT={cn_txt_path}[/green]")
        write_cn_txt_from_srt(cn_srt_path, cn_txt_path, console)
    except Exception as e:
        console.print(f"[bold red]错误: 翻译为中文 SRT 时发生意外错误: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        return False
    return True


def main():
    global WORKDIR
    parser = argparse.ArgumentParser(description="完整字幕处理与翻译流水线")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="工作目录（默认使用 config.yaml 中的 workdir）")
    args = parser.parse_args()
    if args.workdir:
        WORKDIR = args.workdir.resolve()

    console.print(
        Panel(
            "[bold magenta]=== 完整字幕处理与翻译流水线启动 ===[/bold magenta]",
            expand=False,
        )
    )
    console.print(f"Workdir: {WORKDIR}")
    console.print(f"[cyan]Active WORKDIR: {WORKDIR}[/cyan]")
    project_dirs = [d for d in WORKDIR.iterdir() if d.is_dir() and (d / "asr").exists()]
    if not project_dirs:
        console.print(
            "[yellow]在工作目录中未找到任何有效的项目文件夹（即包含asr子目录的文件夹）。[/yellow]"
        )
        return
    console.print(f"发现 {len(project_dirs)} 个有效项目，准备处理...")
    for i, project_dir in enumerate(project_dirs):
        stem = project_dir.name
        console.print(f"\n{'=' * 80}")
        console.print(
            f"[bold]({i + 1}/{len(project_dirs)}) 开始处理项目: {stem}[/bold]"
        )
        console.print(f"[cyan]Current project_dir: {project_dir}[/cyan]")
        en_srt_path = project_dir / f"[EN]-{stem}.srt"
        cn_srt_path = project_dir / f"[CN]-{stem}.srt"
        if en_srt_path.exists() and cn_srt_path.exists():
            console.print(
                f"[green]⏩ 已跳过: 项目 '{stem}' 的英文SRT和中文SRT均已存在。[/green]"
            )
            cn_txt_path = project_dir / f"[CN]-{stem}.txt"
            write_cn_txt_from_srt(cn_srt_path, cn_txt_path, console)
            continue
        if not en_srt_path.exists():
            success_en = generate_english_srt_with_retries(stem, project_dir)
            if not success_en:
                console.print(
                    f"[red]❌ 项目 '{stem}' 未能成功生成英文SRT，跳过此项目的后续步骤。[/red]"
                )
                continue
        else:
            console.print("[green]⏩ 检测到英文SRT已存在，跳过生成步骤。[/green]")
            check_srt_timestamp_gaps(stem, project_dir, WORKDIR, console)
        if not cn_srt_path.exists():
            success_cn = translate_to_chinese_srt(stem, project_dir)
            if not success_cn:
                console.print(f"[red]❌ 项目 '{stem}' 未能成功翻译为中文SRT。[/red]")
                continue
        else:
            console.print("[green]⏩ 检测到中文SRT已存在，跳过翻译步骤。[/green]")
            cn_txt_path = project_dir / f"[CN]-{stem}.txt"
            write_cn_txt_from_srt(cn_srt_path, cn_txt_path, console)
        console.print(f"[bold green]🎉 项目 '{stem}' 全部处理完成！[/bold green]")
    console.print(f"\n{'=' * 80}")
    console.print("[bold magenta]所有任务处理完毕！[/bold magenta]")
    console.print(
        "\n[cyan]执行最终清理：删除所有项目中的临时 'output' 文件夹...[/cyan]"
    )
    cleaned_count = 0
    for project_dir in project_dirs:
        temp_output_dir = project_dir / "output"
        if temp_output_dir.is_dir():
            try:
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                cleaned_count += 1
                console.print(f"  - 已清理: {temp_output_dir}")
            except OSError as e:
                console.print(f"[red]  - 清理失败: {temp_output_dir} ({e})[/red]")
    if cleaned_count > 0:
        console.print(
            f"[green]✅ 清理完成，共删除了 {cleaned_count} 个临时文件夹。[/green]"
        )
    else:
        console.print("[green]✅ 无需清理，未找到任何临时 'output' 文件夹。[/green]")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    main()
