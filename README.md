# Translate Open

`Translate_Open` is an open-source `step1-step4` video localization pipeline for long-form YouTube content.

This public repository focuses on the core production path only:

- `step1`: download videos, thumbnails, and metadata
- `step2`: ingest projects and run ASR with WhisperX
- `step3`: generate EN/CN subtitles and text outputs
- `step4`: render bilingual burned-in videos

It is intentionally limited to the core pipeline. It does **not** include:

- channel discovery
- keyword collection
- tracker databases
- daily automation
- upload / distribution / publishing
- OpenClaw / Feishu notification logic

Downloading videos from overseas platforms requires working network access.

## What This Repo Is For

`Translate_Open` is meant for users who want a practical, scriptable localization pipeline for long-form video.

Typical use cases:

- localizing English YouTube interviews, lectures, or podcasts into Chinese
- generating EN/CN subtitle outputs for later editing
- creating bilingual burned-in videos for internal review or public repost workflows
- using WhisperX + LLM translation as a reproducible step-by-step pipeline instead of a GUI tool

This repository is **not** trying to be:

- a one-click desktop product
- a full publishing platform
- a channel monitoring system
- a cloud service with managed dependencies

## Public Default Path

The default public path is intentionally conservative:

- one `GEMINI_API_KEY`
- Gemini as the default translation provider
- `step2` default path: WhisperX + VAD + alignment
- optional advanced features remain off unless needed

In practice, that means:

- `speaker_diarization` is optional
- vocal separation is optional
- `translation_context.txt` is optional
- `step4` uses automatic font and codec defaults when possible

## End-to-End Flow

The normal workflow is:

1. `step1` downloads source media into a work directory
2. `step2` creates per-video project folders and runs ASR
3. `step3` generates English/Chinese subtitle and text outputs
4. `step4` renders bilingual burned-in output videos

A typical project folder after the full pipeline may contain:

- original video file
- thumbnail / metadata sidecar files
- `asr/`
- `segments/`
- `[EN]-<stem>.srt`
- `[EN]-<stem>.txt`
- `[CN]-<stem>.srt`
- `[CN]-<stem>.txt`
- optional `translation_context.txt`
- `Done_<stem>.mp4`

## Repository Layout

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
- `config.example.yaml`

## Quick Start

If you want the shortest first-run path, start with `step1` only.

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

That command only runs `step1`. To run the full pipeline, install full dependencies and then execute the later steps explicitly.

## Installation

### Download-Only Environment

Use this when you only want `step1`.

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_download.txt
```

### Full Step1-Step4 Environment

Use this when you want the full pipeline.

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Dependency Notes

- `requirements.txt` includes both `google-generativeai` and `google-genai` intentionally:
  `google-generativeai` is used for the Gemini direct API path, while `google-genai` is used for the Vertex path.
- The pinned `torch` packages are only a baseline. If you run `step2` on CUDA, you may need to reinstall
  `torch`, `torchaudio`, and `torchvision` from the official PyTorch index for your exact CUDA version after the base install.
- If you only want `step1`, stick to `requirements_download.txt`.

## Configuration

For the lowest-friction first run, start from:

```bash
cp config.minimal.yaml config.yaml
```

Use `config.example.yaml` if you want the fuller set of knobs.

### Default Translation Provider

```yaml
services:
  translation: gemini
```

### Default Public Model Path

```yaml
models:
  gemini:
    # Keep this name in sync with the model provider's current naming.
    translate: gemini-3-flash-preview
```

### Step2 Defaults

Recommended first-run defaults:

```yaml
asr:
  model_name: medium
  device: auto
  compute_type: auto
  batch_size: auto
  use_vocal_separation: false
  speaker_diarization: false
```

The default `step2` path loads three core components:

- WhisperX transcription
- VAD
- alignment

Optional upgrades:

- vocal separation
- `speaker_diarization`

### Step3 Optional Translation Context

```yaml
translation_context:
  enabled: true
  force_regenerate: false
  file_name: translation_context.txt
  source_max_chars: 12000
```

Notes:

- `translation_context.txt` is the only context file kept in the public `step3` flow
- if the file already exists in a project directory, `step3` will reuse it
- if generation fails, `step3` falls back to the built-in generic context and continues

### Step4 Render Parameters

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

- the public renderer is `pipeline/step4_render.py`
- the public renderer does not add any watermark
- `video_codec: auto` prefers `h264_nvenc` and falls back to `libx264`
- if `font_path` is empty, rendering relies on system-installed font family names
- if `font_path` is set, `step4` passes its parent directory to FFmpeg as `fontsdir`

Platform font defaults:

- Linux: English `DejaVu Sans`, Chinese `Noto Sans CJK SC`
- macOS: English `Helvetica`, Chinese `PingFang SC`
- Windows: English `Arial`, Chinese `Microsoft YaHei`

## How To Run Each Step

### Step1: Download

```bash
python -m pipeline.step1_download \
  --workdir ./workdir \
  --source "https://www.youtube.com/watch?v=VIDEO_ID"
```

or

```bash
python -m pipeline.step1_download \
  --workdir ./workdir \
  --source "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

If `video.pot_provider` is set to `bgutil_http`, `pipeline.step1_download` will automatically check or start the local `bgutil-ytdlp-pot-provider` service before downloading.

Expected outputs from `step1`:

- downloaded media
- thumbnail / cover image
- metadata sidecar files such as `info.json`
- `downloaded_ids.json`

### Step2: Ingest and ASR

```bash
python -m pipeline.step2_ingest --workdir ./workdir
```

Expected outputs from `step2`:

- per-video project folders
- extracted audio
- `asr/`
- chunk artifacts
- WhisperX transcription outputs
- optional diarization outputs

### Step3: Subtitle Translation

```bash
python -m pipeline.step3_translate --workdir ./workdir
```

Expected outputs from `step3`:

- `[EN]-<stem>.srt`
- `[EN]-<stem>.txt`
- `[CN]-<stem>.srt`
- `[CN]-<stem>.txt`
- optional `translation_context.txt`

### Step4: Render

```bash
python -m pipeline.step4_render --workdir ./workdir
```

Expected outputs from `step4`:

- `Done_<stem>.mp4`

## Operational Notes

### Step1 Notes

- downloading from overseas platforms requires working network access
- some YouTube downloads may require cookies or `bgutil_http`
- the public repository supports cookies / browser auth, but those are not part of the minimum first-run path

### Step2 Notes

- `step2` is the heaviest part of the pipeline
- CPU-only runs are possible, but much slower
- GPU users may need to align their local PyTorch install with their CUDA runtime
- Hugging Face authentication may be required for the default VAD path and for diarization-related models

### Step3 Notes

- the public default path uses Gemini
- subtitle translation is designed to preserve structure first, then improve readability
- `translation_context.txt` is optional and exists only to improve consistency across a project

### Step4 Notes

- rendering depends on `ffmpeg`
- font availability differs across Linux, macOS, and Windows
- if your subtitles render with missing glyphs, adjust `font_path`, `font_family_cn`, and `font_family_en`

## What Is Still Intentionally Missing

This public repository excludes the private automation and publishing layers from the original internal system.

Still intentionally missing:

- upload queue management
- channel discovery
- keyword collection
- Bilibili or other publishing automation
- daily reports
- tracker / database orchestration
- private notification and operations tooling

## Documentation

- `docs/STEP1_SETUP.md`: step1 download, bgutil, cookies
- `docs/STEP2_SETUP.md`: WhisperX models, VRAM, VAD baseline, and optional speaker diarization
- `docs/STEP3_SETUP.md`: translation context generation
- `docs/STEP4_SETUP.md`: render behavior and FFmpeg/font setup

## License

- `LICENSE`: MIT

## Practical Expectation Setting

This repository is the public core pipeline, not the full private automation stack.

For common cases, you can get started with one translation API key and the documented local dependencies.
For harder cases, especially around YouTube access, GPU setup, and Hugging Face-gated models, you should expect some local environment work.
