# Windows Desktop Packaging

This directory contains the first-pass packaging assets for the desktop GUI.

## Target

- `Windows 10/11`
- `one-folder` PyInstaller distribution
- entrypoint: `gelai_translate/gui/main.py`

## Expected bundled tools

Before building, place these files here:

- `vendor/ffmpeg/windows-x64/ffmpeg.exe`
- `vendor/ffmpeg/windows-x64/ffprobe.exe`

The GUI startup code prepends this directory to `PATH` in the packaged app, so
`step4` can keep using the default `ffmpeg` / `ffprobe` names.

## Build command

Run on a Windows machine:

```powershell
python -m pip install -U pip
python -m pip install -e .
python -m pip install pyinstaller
pyinstaller packaging/windows/gelai_translate_gui.spec --noconfirm
```

## Output

The distribution folder will be created under:

```text
dist/gelai-translate-gui/
```

## Notes

- This spec is intended for the desktop GUI only.
- `PyInstaller` is not a cross-compiler. Build Windows packages on Windows.
- The first shipping target should be Windows. macOS packaging can follow after
  the Windows path is validated.

