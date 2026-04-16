# Translate Open

[English](./README.md) | 简体中文

`Translate_Open` 是一个面向 YouTube 长视频的开源 `step1-step4` 本地化流水线。

这个公开仓库只保留核心生产路径：

- `step1`：下载视频、封面和元数据
- `step2`：导入项目并使用 WhisperX 运行 ASR
- `step3`：生成中英字幕和文本输出
- `step4`：烧录双语字幕视频

它是一个刻意收口后的公开版本，**不**包含以下私有自动化层：

- 频道巡检
- 关键词采集
- tracker 数据库
- 每日自动化任务
- 上传 / 分发 / 发布
- OpenClaw / Feishu 通知逻辑

下载海外平台视频需要可用网络环境。

## 这个仓库适合做什么

`Translate_Open` 适合需要一个可脚本化、可复现的长视频本地化流水线的用户。

典型使用场景：

- 把英文 YouTube 访谈、课程、播客翻译成中文
- 生成中英文字幕，后续再手工编辑
- 生成双语烧录视频，供内部审核或二次分发
- 用 WhisperX + LLM 构建一个明确分步的流水线，而不是 GUI 工具

这个仓库**不是**：

- 一键式桌面产品
- 完整发布平台
- 频道监控系统
- 托管式云服务

## 默认公开路径

公开版默认路径尽量保持低门槛：

- 一个 `GEMINI_API_KEY`
- Gemini 作为默认翻译 provider
- `step2` 默认路径为 WhisperX + VAD + alignment
- 高级功能默认关闭，按需再开

也就是说：

- `speaker_diarization` 是可选项
- vocal separation 是可选项
- `translation_context.txt` 是可选项
- `step4` 默认尽量自动选择字体和编码器

## 整体流程

典型流程如下：

1. `step1` 下载源视频到工作目录
2. `step2` 创建项目目录并跑 ASR
3. `step3` 生成 EN/CN 字幕和文本
4. `step4` 输出双语烧录视频

完整跑完后，一个项目目录通常会包含：

- 原始视频
- 封面 / 元数据 sidecar 文件
- `asr/`
- `segments/`
- `[EN]-<stem>.srt`
- `[EN]-<stem>.txt`
- `[CN]-<stem>.srt`
- `[CN]-<stem>.txt`
- 可选 `translation_context.txt`
- `Done_<stem>.mp4`

## 仓库结构

- `pipeline/step1_download.py`
- `pipeline/step2_ingest.py`
- `pipeline/step3_translate.py`
- `pipeline/step4_render.py`
- `core/`：step1-step4 依赖的核心模块
- `config/`：清洗后的配置加载和 prompt 模板
- `docs/STEP1_SETUP.md`
- `docs/STEP2_SETUP.md`
- `docs/STEP3_SETUP.md`
- `docs/STEP4_SETUP.md`
- `config.minimal.yaml`
- `config.example.yaml`

## 快速开始

如果你只想先跑通最短路径，建议先从 `step1` 开始。

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

上面这条命令只执行 `step1`。如果你要跑完整流程，需要安装完整依赖，然后再依次执行后续步骤。

## 安装

### 仅下载环境

如果你只想跑 `step1`，使用这个：

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_download.txt
```

### 完整 step1-step4 环境

如果你想跑完整流程，使用这个：

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 依赖说明

- `requirements.txt` 同时包含 `google-generativeai` 和 `google-genai`，这是有意保留的：
  `google-generativeai` 用于 Gemini 直连 API 路径，`google-genai` 用于 Vertex 路径。
- 当前固定的 `torch` 版本只是一个基线。如果你在 CUDA 环境中运行 `step2`，可能需要根据自己的 CUDA 版本，从 PyTorch 官方源重新安装
  `torch`、`torchaudio` 和 `torchvision`。
- 如果你只需要 `step1`，请坚持使用 `requirements_download.txt`。

## 配置

如果你想以最低门槛启动，先从这里开始：

```bash
cp config.minimal.yaml config.yaml
```

如果你想要更多配置项，再使用 `config.example.yaml`。

### 默认翻译 Provider

```yaml
services:
  translation: gemini
```

### 默认公开模型路径

```yaml
models:
  gemini:
    # Keep this name in sync with the model provider's current naming.
    translate: gemini-3-flash-preview
```

### Step2 默认配置

推荐首次运行使用：

```yaml
asr:
  model_name: medium
  device: auto
  compute_type: auto
  batch_size: auto
  use_vocal_separation: false
  speaker_diarization: false
```

默认 `step2` 路径会加载三个核心组件：

- WhisperX transcription
- VAD
- alignment

可选增强项：

- vocal separation
- `speaker_diarization`

### Step3 可选翻译上下文

```yaml
translation_context:
  enabled: true
  force_regenerate: false
  file_name: translation_context.txt
  source_max_chars: 12000
```

说明：

- `translation_context.txt` 是公开版 `step3` 唯一保留的上下文文件
- 如果项目目录里已经存在，`step3` 会直接复用
- 如果生成失败，`step3` 会回退到内置的通用上下文并继续执行

### Step4 渲染参数

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

说明：

- 公开渲染器是 `pipeline/step4_render.py`
- 公开版默认不加任何水印
- `video_codec: auto` 会优先尝试 `h264_nvenc`，没有时回退到 `libx264`
- 如果 `font_path` 为空，渲染依赖系统已安装字体
- 如果设置了 `font_path`，`step4` 会把其父目录作为 `fontsdir` 传给 FFmpeg

平台默认字体：

- Linux：英文 `DejaVu Sans`，中文 `Noto Sans CJK SC`
- macOS：英文 `Helvetica`，中文 `PingFang SC`
- Windows：英文 `Arial`，中文 `Microsoft YaHei`

## 每一步怎么运行

### Step1：下载

```bash
python -m pipeline.step1_download \
  --workdir ./workdir \
  --source "https://www.youtube.com/watch?v=VIDEO_ID"
```

或者：

```bash
python -m pipeline.step1_download \
  --workdir ./workdir \
  --source "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

如果 `video.pot_provider` 设成 `bgutil_http`，`pipeline.step1_download` 会在下载前自动检查或启动本地 `bgutil-ytdlp-pot-provider`。

`step1` 的典型输出：

- 下载的视频文件
- 缩略图 / 封面
- 元数据 sidecar 文件
- `downloaded_ids.json`

### Step2：导入与 ASR

```bash
python -m pipeline.step2_ingest --workdir ./workdir
```

`step2` 的典型输出：

- 每个视频对应的项目目录
- 提取后的音频
- `asr/`
- chunk 中间产物
- WhisperX 识别输出
- 可选 diarization 输出

### Step3：字幕翻译

```bash
python -m pipeline.step3_translate --workdir ./workdir
```

`step3` 的典型输出：

- `[EN]-<stem>.srt`
- `[EN]-<stem>.txt`
- `[CN]-<stem>.srt`
- `[CN]-<stem>.txt`
- 可选 `translation_context.txt`

### Step4：渲染

```bash
python -m pipeline.step4_render --workdir ./workdir
```

`step4` 的典型输出：

- `Done_<stem>.mp4`

## 运行边界说明

### Step1

- 下载海外平台视频需要可用网络环境
- 某些 YouTube 视频可能需要 cookies 或 `bgutil_http`
- 公开版保留了 cookies / browser auth 支持，但它们不是最小路径的默认要求

### Step2

- `step2` 是整套流程里最重的一步
- CPU 可以跑，但会慢很多
- GPU 用户可能需要按自己的 CUDA 环境调整 PyTorch 安装
- 默认 VAD 路径以及 diarization 相关模型，可能需要 Hugging Face 认证

### Step3

- 公开版默认翻译 provider 是 Gemini
- 字幕翻译优先保证结构稳定，再追求可读性
- `translation_context.txt` 只是为了提高术语和风格一致性，不是发布产物

### Step4

- 渲染依赖 `ffmpeg`
- Linux / macOS / Windows 的字体可用性不同
- 如果字幕缺字或显示异常，需要手动调整 `font_path`、`font_family_cn`、`font_family_en`

## 这次公开版刻意去掉了什么

这个仓库不包含原始内部系统里的私有自动化和发布层。

刻意去掉的内容包括：

- 上传队列管理
- 频道巡检
- 关键词采集
- Bilibili 或其它平台发布自动化
- 每日报告
- tracker / 数据库编排
- 私有通知与运维工具

## 文档索引

- `docs/STEP1_SETUP.md`：step1 下载、bgutil、cookies
- `docs/STEP2_SETUP.md`：WhisperX 模型、显存、VAD 默认路径、可选说话人识别
- `docs/STEP3_SETUP.md`：translation context 生成逻辑
- `docs/STEP4_SETUP.md`：渲染行为与 FFmpeg / 字体配置

## License

- `LICENSE`: MIT

## 预期管理

这个仓库是公开核心流水线，不是完整私有自动化栈。

对多数常见路径来说，你可以通过一个翻译 API key 加上文档里的本地依赖把流程跑起来。
但对于更复杂的场景，尤其是 YouTube 访问、GPU 环境和 Hugging Face gated 模型，你仍然需要做本地环境适配。
