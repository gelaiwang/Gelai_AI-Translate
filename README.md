# Translate Open

`Translate_Open` is an open-source `step1-step4` video localization pipeline for YouTube long-form content.

It covers:

- `step1`: download videos, thumbnails, and metadata
- `step2`: ingest projects and run ASR with WhisperX, using VAD as the default base capability and `speaker_diarization` as an optional advanced feature
- `step3`: segment subtitles, optionally generate `translation_context.txt`, and produce EN/CN subtitle and text outputs
- `step4`: render bilingual burned-in videos

`step2` default path loads three core components:

- WhisperX transcription
- VAD
- alignment

Optional upgrades:

- vocal separation
- `speaker_diarization`

It does not include:

- channel discovery
- keyword collection
- tracker databases
- daily automation
- upload / distribution / publishing
- OpenClaw / Feishu notification logic

Downloading videos from overseas platforms requires working network access.

Recommended public default path:

- one `GEMINI_API_KEY`
- Gemini as the default translation provider
- default step2 path: WhisperX + VAD + alignment
- optional features stay off by default unless needed

## Quick Start

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_download.txt
export GEMINI_API_KEY=your_api_key_here
cp config.minimal.yaml config.yaml
python -m pipeline.step1_download --workdir ./workdir --source "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

## Layout

- `pipeline/step1_download.py`
- `pipeline/step2_ingest.py`
- `pipeline/step3_translate.py`
- `pipeline/step4_render.py`
- `core/` minimal modules used by step1-step4
- `config/` sanitized config loader and prompt templates
- `docs/STEP1_SETUP.md`
- `docs/STEP2_SETUP.md`
- `docs/STEP3_SETUP.md`
- `docs/STEP4_SETUP.md`
- `config.minimal.yaml`

If `video.pot_provider` is set to `bgutil_http`, `pipeline.step1_download` will automatically check or start the local `bgutil-ytdlp-pot-provider` service before downloading.

## Install

Download-only environment:

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_download.txt
```

Full step1-step4 environment:

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Dependency notes:

- `requirements.txt` includes both `google-generativeai` and `google-genai` intentionally:
  `google-generativeai` is used for the Gemini direct API path, while `google-genai` is used for the Vertex path.
- The pinned `torch` packages are only a baseline. If you run step2 on CUDA, you may need to reinstall
  `torch`, `torchaudio`, and `torchvision` from the official PyTorch index for your exact CUDA version after the base install.
- If you only want `step1`, stick to `requirements_download.txt`.

## Key Config

For the lowest-friction first run, start from:

```bash
cp config.minimal.yaml config.yaml
```

Default translation provider:

```yaml
services:
  translation: gemini
```

Default public model path:

```yaml
models:
  gemini:
    # Keep this name in sync with the model provider's current naming.
    translate: gemini-3-flash-preview
```

`step3` optional translation context:

```yaml
translation_context:
  enabled: true
  force_regenerate: false
  file_name: translation_context.txt
  source_max_chars: 12000
```

Notes:

- `translation_context.txt` is the only context file kept in the public step3 flow
- if the file already exists in a project directory, step3 will reuse it
- if generation fails, step3 falls back to the built-in generic context and continues

`step4` editable render parameters:

```yaml
render:
  font_path:
  font_family_en: DejaVu Sans
  font_family_cn: Noto Sans CJK SC
  subtitle_font_size_en: 13
  subtitle_font_size_cn: 22
  ffmpeg_bin: ffmpeg
  ffprobe_bin: ffprobe
  video_codec: auto
  video_preset: slow
  video_crf: 20
  output_suffix: Done_
```

Notes:

- the public step4 renderer is `pipeline/step4_render.py`
- the public renderer does not add any watermark
- `video_codec: auto` prefers `h264_nvenc` and falls back to `libx264`
- if `font_path` is empty, rendering relies on system-installed font family names
- if `font_path` is set, step4 passes its parent directory to FFmpeg as `fontsdir`

Platform font defaults:

- Linux: English `DejaVu Sans`, Chinese `Noto Sans CJK SC`
- macOS: English `Helvetica`, Chinese `PingFang SC`
- Windows: English `Arial`, Chinese `Microsoft YaHei`

## Docs

- `docs/STEP1_SETUP.md`: step1 download, bgutil, cookies
- `docs/STEP2_SETUP.md`: WhisperX models, VRAM, VAD baseline, and optional speaker diarization
- `docs/STEP3_SETUP.md`: translation context generation
- `docs/STEP4_SETUP.md`: render behavior and FFmpeg/font setup

## Release Notes

- `LICENSE`: MIT
- this repo is intended as a core pipeline, not a full automation or publishing platform
- clean-machine verification is still recommended before publishing
