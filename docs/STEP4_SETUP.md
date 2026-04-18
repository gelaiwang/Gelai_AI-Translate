# Step4 Setup

`step4` renders the final burned-in bilingual video from:

- the source video `*.mp4`
- the English subtitle file `[EN]-*.srt`
- the Chinese subtitle file `[CN]-*.srt`

The public renderer entry is:

- `pipeline/step4_render.py`

## Config

The `render` section in `config.yaml` controls the public step4 behavior:

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

## Important Defaults

- `font_path` is optional in the open-source version.
- if `font_path` is set, step4 passes its parent directory to FFmpeg as `fontsdir`
- if `font_path` is empty, step4 relies on system-installed font family names
- `video_codec: auto` prefers `h264_nvenc` when FFmpeg reports it as available, otherwise it falls back to `libx264`.

## Recommended Usage

If you want the simplest portable setup:

- install `ffmpeg` so it is available on `PATH`
- leave `video_codec: auto`
- set `font_family_en` and `font_family_cn` to fonts that exist on your system

If you need a specific font file, set:

```yaml
render:
  font_path: /path/to/your/font.ttf
```

## Runtime Behavior

For each project directory under `workdir`, step4 looks for:

- `{stem}.mp4`
- `[EN]-{stem}.srt`
- `[CN]-{stem}.srt`

If all three files exist, it renders:

- `{output_suffix}{stem}.mp4`

By default, that means:

- `Done_{stem}.mp4`

## Open-Source Scope

The public step4 flow keeps only the core burn-in renderer.

It does not include:

- uploader-specific render variants
- publishing pipeline integration
- post-render distribution logic
