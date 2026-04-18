# Desktop GUI

`Translate_Open` 现在已经带有一版早期桌面 GUI 骨架，基于 `PySide6`。

## 当前范围

这版桌面 GUI 是对现有 `step1-step4` 流水线的本地工作台封装：

- Project 页
- Step1 下载页
- Step2 ASR 页
- Step3 翻译页
- Step4 渲染页
- 全局 Logs 页

它直接调用现有 pipeline 入口：

- `pipeline.step1_download.run(...)`
- `pipeline.step2_ingest.run(...)`
- `pipeline.step3_translate.run(...)`
- `pipeline.step4_render.run(...)`

## 开发环境启动

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
gelai-translate-gui
```

## 当前行为

- 只支持单任务执行
- 所有 step 共用一个日志面板
- `Project` 页负责管理 `config.yaml` 和 `workdir`
- 各 step 页面在后台线程中调用真实的 pipeline `run(...)`

## 打包

第一阶段打包目标是 Windows `one-folder` 发行包，基于 PyInstaller。

参见：

- `packaging/windows/gelai_translate_gui.spec`
- `packaging/windows/README.md`

