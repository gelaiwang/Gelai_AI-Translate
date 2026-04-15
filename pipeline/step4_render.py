# Bilingual subtitle burn-in renderer
import argparse
import platform
import subprocess
import sys
from pathlib import Path
import re
import time
import datetime

try:
    from config import (
        WORKDIR,
        FONT_PATH,
        FONT_FAMILY_EN,
        FONT_FAMILY_CN,
        SUBTITLE_FONT_SIZE_EN,
        SUBTITLE_FONT_SIZE_CN,
        RENDER_FFMPEG_BIN,
        RENDER_FFPROBE_BIN,
        RENDER_VIDEO_CODEC,
        RENDER_VIDEO_PRESET,
        RENDER_VIDEO_CRF,
        RENDER_OUTPUT_SUFFIX,
    )  # 从统一的 config 包导入配置
except ImportError:
    print("❌ Error: 无法从 config 包导入 WORKDIR。请检查 `config/settings.py` 是否存在且可导入。")
    sys.exit(1)
except Exception as e:
    print(f"❌ 导入 config 时发生错误: {e}")
    sys.exit(1)

# --- 可配置参数 ---

# FFmpeg/FFprobe路径 (如果不在系统PATH中，请指定完整路径)
FFMPEG_PATH = RENDER_FFMPEG_BIN
FFPROBE_PATH = RENDER_FFPROBE_BIN

# 输出视频参数
OUTPUT_VIDEO_PRESET = RENDER_VIDEO_PRESET
OUTPUT_VIDEO_CRF = RENDER_VIDEO_CRF
OUTPUT_VIDEO_SUFFIX = RENDER_OUTPUT_SUFFIX

# --- 字幕烧录通用参数 ---
FONT_FILE_EN = FONT_FAMILY_EN
FONT_FILE_CN = FONT_FAMILY_CN
FONT_SIZE_EN = str(SUBTITLE_FONT_SIZE_EN)
FONT_SIZE_CN = str(SUBTITLE_FONT_SIZE_CN)
FONT_COLOR_EN = "white"
FONT_COLOR_CN = "white"
OUTLINE_COLOR = "000000"
OUTLINE_WIDTH = "1.0"
EN_SUB_ALIGNMENT = "2"
CN_SUB_ALIGNMENT = "2"
SUB_MARGIN_V_EN = "10"
SUB_MARGIN_V_CN = "27"
SUB_MARGIN_H = "20"

def resolve_video_codec() -> str:
    if RENDER_VIDEO_CODEC and RENDER_VIDEO_CODEC != "auto":
        return RENDER_VIDEO_CODEC

    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError:
        return "libx264"
    except Exception:
        return "libx264"

    output = result.stdout or ""
    return "h264_nvenc" if "h264_nvenc" in output else "libx264"


def resolve_font_family(configured_family: str, *, is_cjk: bool) -> str:
    preferred = configured_family.strip()
    if preferred:
        return preferred

    system_name = platform.system().lower()
    if is_cjk:
        if system_name == "darwin":
            return "PingFang SC"
        if system_name == "windows":
            return "Microsoft YaHei"
        return "Noto Sans CJK SC"

    if system_name == "darwin":
        return "Helvetica"
    if system_name == "windows":
        return "Arial"
    return "DejaVu Sans"

def escape_ffmpeg_path(path_str: str) -> str:
    escaped_path = path_str.replace('\\', '/')
    if ':' in escaped_path:
         parts = escaped_path.split(':', 1)
         escaped_path = f"{parts[0]}\\:{parts[1]}"
    return escaped_path


def build_subtitle_filter(srt_path: Path, *, font_family: str, font_size: str, primary_color: str, outline_color: str, outline_width: str, alignment: str, margin_v: str) -> str:
    escaped_srt_path = escape_ffmpeg_path(srt_path.as_posix())
    style_parts = [
        f"Fontname={font_family}",
        f"FontSize={font_size}",
        f"PrimaryColour=&{primary_color}",
        f"OutlineColour=&{outline_color}",
        f"Outline={outline_width}",
        "BorderStyle=1",
        f"Alignment={alignment}",
        f"MarginV={margin_v}",
        f"MarginL={SUB_MARGIN_H}",
        f"MarginR={SUB_MARGIN_H}",
    ]
    if font_family == FONT_FILE_CN:
        style_parts.append("Spacing=1.2")
    style = ",".join(style_parts)
    filter_parts = [f"filename='{escaped_srt_path}'"]
    if FONT_PATH and FONT_PATH.exists():
        filter_parts.append(f"fontsdir='{escape_ffmpeg_path(FONT_PATH.parent.as_posix())}'")
    filter_parts.append(f"force_style='{style}'")
    return "subtitles=" + ":".join(filter_parts)

def run_ffmpeg_command(cmd: list[str], video_file_name: str):
    print(f"🛠️  Processing: {video_file_name}")
    start_time = time.time()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8')
        duration_regex = re.compile(r"Duration: (\d{2}):(\d{2}):(\d{2})\.\d{2}")
        time_regex = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.\d{2}")
        total_seconds = 0
        last_len = 0
        full_output = []
        for line in process.stdout: # type: ignore
            full_output.append(line)
            if total_seconds == 0:
                match = duration_regex.search(line)
                if match:
                    hours, minutes, seconds = map(int, match.groups())
                    total_seconds = hours * 3600 + minutes * 60 + seconds
            if "frame=" in line and "time=" in line:
                progress_match = time_regex.search(line)
                if progress_match and total_seconds > 0:
                    hours, minutes, seconds = map(int, progress_match.groups())
                    current_seconds = hours * 3600 + minutes * 60 + seconds
                    percent = (current_seconds / total_seconds) * 100
                    bar_length = 25
                    filled_length = int(bar_length * current_seconds // total_seconds)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    elapsed_seconds = time.time() - start_time
                    elapsed_formatted = str(datetime.timedelta(seconds=int(elapsed_seconds)))
                    speed_match = re.search(r"speed=\s*(\S+)", line)
                    speed = speed_match.group(1) if speed_match else "N/A"
                    current_time_str = progress_match.group(0).split('=')[1].strip()
                    status_line = (f"   [{bar}] {percent:.1f}% | 进度: {current_time_str} | 已用: {elapsed_formatted} | 速度: {speed.ljust(8)}")
                    pad = max(0, last_len - len(status_line))
                    sys.stdout.write(f"\r{status_line}{' ' * pad}")
                    sys.stdout.flush()
                    last_len = len(status_line)
        process.wait()
        sys.stdout.write("\n")
        if process.returncode != 0:
            print(f"❌ FFmpeg 命令执行失败，返回码: {process.returncode}，文件名: '{video_file_name}'")
            print("--- FFmpeg 详细错误日志 ---")
            for log_line in full_output:
                print(log_line, end='')
            print("--- 日志结束 ---")
            return False
    except FileNotFoundError:
        print(f"❌ 错误: 未找到 FFmpeg。请确保已安装并配置到系统路径 (PATH)，或在脚本中指定 FFMPEG_PATH。")
        return False
    except Exception as e:
        print(f"❌ 运行 FFmpeg 时发生意外错误，文件名 '{video_file_name}': {e}")
        return False
    return True

def burn_video(video_path: Path, en_srt_path: Path, cn_srt_path: Path, output_path: Path):
    filter_complex_steps = []

    filter_complex_steps.append(
        "[0:v]scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=w=1920:h=1080:x=(ow-iw)/2:y=(oh-ih)/2:color=black[bg]"
    )

    primary_color_en = "HFFFFFF"
    if FONT_COLOR_EN.lower() == "yellow":
        primary_color_en = "H00FFFF"
    primary_color_cn = "H00FFFF"
    if FONT_COLOR_CN.lower() == "white":
        primary_color_cn = "HFFFFFF"
    outline_color_val = f"H{OUTLINE_COLOR.upper()}"

    en_sub_filter = build_subtitle_filter(
        en_srt_path,
        font_family=resolve_font_family(FONT_FILE_EN, is_cjk=False),
        font_size=FONT_SIZE_EN,
        primary_color=primary_color_en,
        outline_color="H404040",
        outline_width="0.8",
        alignment=EN_SUB_ALIGNMENT,
        margin_v=SUB_MARGIN_V_EN,
    )
    cn_sub_filter = build_subtitle_filter(
        cn_srt_path,
        font_family=resolve_font_family(FONT_FILE_CN, is_cjk=True),
        font_size=FONT_SIZE_CN,
        primary_color=primary_color_cn,
        outline_color=outline_color_val,
        outline_width=OUTLINE_WIDTH,
        alignment=CN_SUB_ALIGNMENT,
        margin_v=SUB_MARGIN_V_CN,
    )

    final_filters_on_combined = f"[bg]{en_sub_filter},{cn_sub_filter}[v]"
    filter_complex_steps.append(final_filters_on_combined)

    filter_complex_string = ";".join(filter_complex_steps)

    video_codec = resolve_video_codec()
    if video_codec == "h264_nvenc":
        print("✅ Using NVIDIA NVENC for video encoding.")
    else:
        print(f"✅ Using video encoder: {video_codec}")
    ffmpeg_cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex_string,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", video_codec,
        "-preset", OUTPUT_VIDEO_PRESET,
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        str(output_path)
    ]
    if video_codec == "h264_nvenc":
        ffmpeg_cmd[10:10] = ["-rc", "vbr_hq", "-cq", OUTPUT_VIDEO_CRF]
    else:
        ffmpeg_cmd[10:10] = ["-crf", OUTPUT_VIDEO_CRF]

    return run_ffmpeg_command(ffmpeg_cmd, video_path.name)

def main():
    parser = argparse.ArgumentParser(description="Batch render bilingual burned-in videos")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="工作目录（默认使用 config.yaml 中的 workdir）")
    args = parser.parse_args()

    workdir_from_config = (args.workdir.resolve() if args.workdir
                           else Path(WORKDIR).expanduser().resolve())
    if not workdir_from_config.is_dir():
        print(f"❌ Error: WORKDIR '{workdir_from_config}' not found or is not a directory.")
        sys.exit(1)

    print(f"📂 Scanning project folders in WORKDIR: {workdir_from_config}")

    project_dirs = [d for d in workdir_from_config.iterdir() if d.is_dir()]
    if not project_dirs:
        print(f"ℹ️ No project folders found in '{workdir_from_config}'. Nothing to do.")
        sys.exit(0)

    videos_to_process = []
    missing_files_projects = []

    for project_dir in project_dirs:
        stem = project_dir.name
        output_filename = f"{OUTPUT_VIDEO_SUFFIX}{stem}.mp4"
        output_path = project_dir / output_filename
        if output_path.exists():
            print(f"⏭️  Skipping project '{stem}': Output file '{output_filename}' already exists.")
            continue

        video_path = project_dir / f"{stem}.mp4"
        en_srt_path = project_dir / f"[EN]-{stem}.srt"
        cn_srt_path = project_dir / f"[CN]-{stem}.srt"

        if video_path.exists() and en_srt_path.exists() and cn_srt_path.exists():
            videos_to_process.append({
                "video_path": video_path,
                "en_srt_path": en_srt_path,
                "cn_srt_path": cn_srt_path,
                "output_path": output_path
            })
            print(f"  ✅ Found ready project: '{stem}'")
        else:
            missing_files = []
            if not video_path.exists(): missing_files.append(video_path.name)
            if not en_srt_path.exists(): missing_files.append(en_srt_path.name)
            if not cn_srt_path.exists(): missing_files.append(cn_srt_path.name)
            if missing_files:
                missing_files_projects.append({"project_name": stem, "missing": missing_files})

    if missing_files_projects:
        print("\n" + "="*20 + " ATTENTION: Incomplete Projects " + "="*20)
        for item in missing_files_projects:
            print(f"  - Project: {item['project_name']}")
            for missing_file in item['missing']:
                print(f"     L Missing: {missing_file}")
        print("="* (40 + len(" ATTENTION: Incomplete Projects ")) + "\n")

    if not videos_to_process:
        print("ℹ️ No new projects found ready to be processed.")
        sys.exit(0)

    print(f"\n🎬 Found {len(videos_to_process)} video(s) to process. Starting burning process...")
    successful_burns, failed_burns = 0, 0

    for job in videos_to_process:
        print("-" * 60)
        if burn_video(job["video_path"], job["en_srt_path"], job["cn_srt_path"], job["output_path"]):
            print(f"✅ Successfully processed and saved: {job['output_path'].name}")
            successful_burns += 1
        else:
            print(f"❌ Failed to process project: {job['video_path'].parent.name}")
            failed_burns += 1
        print("-" * 60)

    print("\n\n" + "="*23 + " Processing Summary " + "="*23)
    print(f"  Total Projects Found: {len(project_dirs)}")
    print(f"  Successfully Processed: {successful_burns}")
    print(f"  Failed to Process: {failed_burns}")
    print(f"  Skipped (Already Done): {len(project_dirs) - successful_burns - failed_burns - len(missing_files_projects)}")
    print(f"  Skipped (Missing Files): {len(missing_files_projects)}")
    print("="* (46 + len(" Processing Summary ")) + "\n")

    if failed_burns == 0:
        print("🎉 All selected videos processed successfully!")
    else:
        print(f"⚠️  {failed_burns} video(s) failed during processing. Please check the logs above.")


if __name__ == "__main__":
    main()
