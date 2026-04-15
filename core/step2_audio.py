from __future__ import annotations

import gc
from pathlib import Path

import torch

try:
    from audio_separator.separator import Separator
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False


SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
VOCALS_SUFFIX = "_vocals.mp3"
INSTRUMENTAL_SUFFIX = "_instrumental.mp3"


def estimate_snr_wada(console, audio_path: Path) -> float:
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-10)
        wada_coefs = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        wada_g_values = np.array([0.0, 0.05, 0.13, 0.24, 0.38, 0.52, 0.65, 0.76, 0.84, 0.90, 0.94, 0.97, 0.99])
        abs_y = np.abs(y)
        mean_abs = np.mean(abs_y)
        rms = np.sqrt(np.mean(y ** 2))
        g = mean_abs / (rms + 1e-10) * np.sqrt(np.pi / 2)
        g = np.clip(g, 0.0, 1.0)
        snr_db = np.interp(g, wada_g_values, wada_coefs)
        return float(snr_db)
    except Exception as e:
        console.print(f"      [yellow]WADA-SNR 计算出错：{e}[/yellow]")
        return 0.0


def run_vocal_separation_on_project(console, logger, project_dir: Path, snr_threshold: float = 20.0) -> bool:
    if not SEPARATOR_AVAILABLE:
        console.print("    [yellow]未安装 audio-separator，跳过人声分离。[/yellow]")
        return False

    chunk_dir = project_dir / "chunks"
    wav_list = sorted(chunk_dir.glob("segment_*.wav"))
    if not wav_list:
        return True

    sample_wav = wav_list[0]
    try:
        estimated_snr = estimate_snr_wada(console, sample_wav)
        console.print(f"    [cyan]WADA-SNR: {estimated_snr:.1f} dB（阈值：{snr_threshold:.1f} dB）[/cyan]")
        if estimated_snr >= snr_threshold:
            console.print("    [green]SNR 足够高，跳过人声分离以节省时间。[/green]")
            return True
        console.print("    [yellow]SNR 较低，需要执行人声分离。[/yellow]")
    except Exception as e:
        console.print(f"    [yellow]SNR 估算失败：{e}，将继续执行分离。[/yellow]")

    files_to_separate = []
    for wav_path in wav_list:
        vocals_path = chunk_dir / f"{wav_path.stem}{VOCALS_SUFFIX}"
        if not vocals_path.exists():
            files_to_separate.append(wav_path)

    if not files_to_separate:
        console.print("    [dim green]人声文件已全部存在，跳过分离阶段。[/dim green]")
        return True

    console.print(f"    [bold cyan]开始人声分离（{len(files_to_separate)} 个文件）...[/bold cyan]")
    separator = None
    try:
        separator = Separator(output_dir=str(chunk_dir), output_format="mp3")
        console.print(f"      正在加载分离模型：{SEPARATOR_MODEL} ...")
        separator.load_model(SEPARATOR_MODEL)
        console.print("      [green]分离模型加载成功。[/green]")

        for idx, wav_path in enumerate(files_to_separate, start=1):
            console.print(f"      [cyan]>[/cyan] 分离 {idx}/{len(files_to_separate)}：{wav_path.name}")
            console.print("        [dim]正在使用 BS-Roformer 模型进行音频分离...[/dim]")
            try:
                output_files = separator.separate(str(wav_path))
                console.print("        [dim]分离完成，正在处理输出文件...[/dim]")
                for output_file in output_files:
                    output_filename = Path(output_file).name
                    output_path = chunk_dir / output_filename
                    console.print(f"        [dim]检测到输出文件：{output_filename}[/dim]")

                    if "(Vocals)" in output_filename or "vocals" in output_filename.lower():
                        new_vocals_path = chunk_dir / f"{wav_path.stem}{VOCALS_SUFFIX}"
                        if output_path.exists():
                            file_size = output_path.stat().st_size
                            if file_size < 1000:
                                console.print(f"        [yellow]人声文件过小（{file_size} bytes），可能分离失败。[/yellow]")
                                output_path.unlink()
                                continue
                            output_path.rename(new_vocals_path)
                            console.print(f"        [green]人声文件：{new_vocals_path.name}（{file_size // 1024}KB）[/green]")
                    elif "(Instrumental)" in output_filename or "instrumental" in output_filename.lower():
                        new_instrumental_path = chunk_dir / f"{wav_path.stem}{INSTRUMENTAL_SUFFIX}"
                        if output_path.exists():
                            file_size = output_path.stat().st_size
                            if file_size < 1000:
                                console.print(f"        [yellow]背景音文件过小（{file_size} bytes），已删除。[/yellow]")
                                output_path.unlink()
                                continue
                            output_path.rename(new_instrumental_path)
                            console.print(f"        [blue]背景音文件：{new_instrumental_path.name}（{file_size // 1024}KB）[/blue]")
                    elif output_path.exists():
                        console.print(f"        [yellow]未知输出文件：{output_filename}[/yellow]")
            except Exception as e:
                console.print(f"        [red]分离失败：{e}[/red]")
                logger.error(f"人声分离失败：{wav_path.name}，错误：{e}")
                continue
    except Exception as e:
        console.print(f"    [red]分离器初始化失败：{e}[/red]")
        logger.error(f"分离器初始化失败：{e}")
        return False
    finally:
        console.print("    [dim]释放分离器显存...[/dim]")
        if separator is not None:
            del separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.print("    [dim green]显存已释放。[/dim green]")

    return True
