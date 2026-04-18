# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


PROJECT_ROOT = Path(SPEC).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "gelai_translate" / "gui" / "main.py"


def optional_data(src: Path, dest: str) -> list[tuple[str, str]]:
    return [(str(src), dest)] if src.exists() else []


datas = []
datas += collect_data_files("config", includes=["prompts/*.txt"])
datas += optional_data(PROJECT_ROOT / "config.minimal.yaml", ".")
datas += optional_data(PROJECT_ROOT / "config.example.yaml", ".")
datas += optional_data(PROJECT_ROOT / "data" / "series_registry.yaml", "data")
datas += optional_data(PROJECT_ROOT / "vendor" / "ffmpeg" / "windows-x64" / "ffmpeg.exe", "vendor/ffmpeg/windows-x64")
datas += optional_data(PROJECT_ROOT / "vendor" / "ffmpeg" / "windows-x64" / "ffprobe.exe", "vendor/ffmpeg/windows-x64")

hiddenimports = []
hiddenimports += collect_submodules("whisperx")
hiddenimports += collect_submodules("pyannote.audio")
hiddenimports += [
    "google.generativeai",
    "google.genai",
    "audio_separator",
    "librosa",
    "torch",
    "torchaudio",
    "torchvision",
]

block_cipher = None

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gelai-translate-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="gelai-translate-gui",
)

