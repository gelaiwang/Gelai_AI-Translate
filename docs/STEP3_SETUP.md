# Step3 Setup

`step3` is responsible for:

- segmenting the ASR output into subtitle-ready English lines
- generating `[EN]-*.srt` and `[EN]-*.txt`
- translating subtitles into Chinese
- generating `[CN]-*.srt` and `[CN]-*.txt`
- optionally generating `translation_context.txt` for translation consistency

## What `translation_context.txt` Does

`translation_context.txt` is a reusable helper file for subtitle translation.

It is meant to improve:

- terminology consistency
- person / company / product name handling
- tone consistency across batches
- series-level translation stability

It is not meant to generate:

- platform metadata
- publishing copy
- titles, descriptions, highlights, or tags

## Config

Add this section to `config.yaml`:

```yaml
translation_context:
  enabled: true
  force_regenerate: false
  file_name: translation_context.txt
  source_max_chars: 12000
```

Meaning:

- `enabled`: when `true`, step3 will generate `translation_context.txt` if the file is missing
- `force_regenerate`: when `true`, step3 will regenerate the file even if one already exists
- `file_name`: file name written into each project directory
- `source_max_chars`: maximum subtitle excerpt length sent into the context-generation prompt

## Runtime Behavior

Step3 resolves translation context in this order:

1. If the project directory already contains `translation_context.txt` and `force_regenerate` is `false`, reuse it.
2. Otherwise, if `enabled` is `true`, generate a new one from project metadata and an English subtitle excerpt.
3. If generation fails or the generated file does not contain the required structure, fall back to the built-in default context and continue translation.

This means `translation_context.txt` is optional. Step3 can still complete without it.

## Required Output Structure

The generated file must contain these two sections:

- `【翻译风格基准】`
- `【微型术语表】`

The bundled prompt template for this file is `config/prompts/translation_context.txt`.

## Recommended Use

- Enable it for channels or series with repeated speakers, repeated terminology, or dense technical vocabulary.
- Leave it disabled if you want the simplest possible first run.
- Use `force_regenerate: true` only when you intentionally want to refresh the context after prompt or metadata changes.

## Public Repo Scope

The open-source step3 flow keeps only `translation_context.txt` as the optional context artifact.

It does not generate:

- `context.md`
- `platform_info.md`
- publishing-oriented metadata files
