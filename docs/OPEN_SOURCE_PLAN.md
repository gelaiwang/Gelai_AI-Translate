# Open-Source Extraction Plan

This directory is the first extraction target for a public `step1-4` release.

## Included

- `pipeline/step1_download.py`
- `pipeline/step2_ingest.py`
- `pipeline/step3_translate.py`
- `pipeline/step4_render.py`
- direct core dependencies used by those modules
- sanitized config loader
- prompt templates required by step3

## Intentionally Excluded

- any `.env`, `config.yaml`, cookies, or cloud auth files
- tracker databases and tracker code
- channel discovery / keyword collection
- daily automation and OpenClaw notification logic
- step5 distribution / upload / publishing
- NAS-specific operational scripts

## Files That Should Stay Private In The Original Project

- `config.yaml`
- `.gemini.env`
- `.grok.env`
- `.vertex.env`
- `youtube_cookies.txt`
- `cookies.json`
- `data/*.db`
- `state/*`
- `pipeline/step5_upload*.py`
- `tools/channel_discovery.py`
- `tools/keyword_collection.py`
- `tools/daily_pipeline.py`
- `tools/file_distribute.py`
- `tools/notify_openclaw.py`

## Next Cleanup Tasks

1. Remove remaining references to private operational assumptions from the copied modules.
2. Verify which prompt files are truly required and delete the rest.
3. Add a clean CLI entrypoint layer that does not depend on local wrappers.
4. Test on a fresh machine with only `config.example.yaml` and `.env`.
5. Add sample commands and expected output structure to the README.
