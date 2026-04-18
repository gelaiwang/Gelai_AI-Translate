# Desktop GUI

`Translate_Open` now includes an early desktop GUI scaffold built with `PySide6`.

## Current scope

The desktop GUI is a local workbench for the existing `step1-step4` pipeline:

- Project page
- Step1 download page
- Step2 ASR page
- Step3 translate page
- Step4 render page
- shared logs page

It calls the existing pipeline entrypoints directly:

- `pipeline.step1_download.run(...)`
- `pipeline.step2_ingest.run(...)`
- `pipeline.step3_translate.run(...)`
- `pipeline.step4_render.run(...)`

## Start in development mode

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
gelai-translate-gui
```

## Current behavior

- single-task execution only
- shared log panel for all steps
- Project page drives `config.yaml` and `workdir`
- step pages call the real pipeline `run(...)` functions in a worker thread

## Packaging

The first packaging target is Windows `one-folder` distribution using PyInstaller.

See:

- `packaging/windows/gelai_translate_gui.spec`
- `packaging/windows/README.md`

