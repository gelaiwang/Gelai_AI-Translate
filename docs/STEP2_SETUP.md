# Step2 Setup

This document covers the public `step2` path:

- three core models/components: WhisperX transcription, VAD, and alignment
- audio extraction
- chunking
- WhisperX transcription
- VAD as a default base capability
- alignment
- optional vocal separation
- optional speaker diarization

## First-run defaults

For a public first run, use:

```yaml
asr:
  model_name: medium
  device: auto
  compute_type: auto
  batch_size: auto
  use_vocal_separation: false
  speaker_diarization: false
```

This keeps the first run focused on getting ASR working first.

## Dependency notes

- `requirements.txt` keeps both Google SDKs on purpose:
  `google-generativeai` is used for the Gemini direct API path, and `google-genai`
  is used for the Vertex path.
- The pinned `torch`, `torchaudio`, and `torchvision` versions are a baseline, not a universal CUDA guarantee.
  If you use NVIDIA GPUs, reinstall the PyTorch stack from the official PyTorch wheel index for your exact CUDA runtime.
- CPU-only users can usually keep the pinned defaults as-is.

Operational model:

- the default step2 path loads three core components: transcription, VAD, and alignment
- `VAD` is treated as a default base capability in the public step2 pipeline.
- `speaker_diarization` is treated as an optional advanced capability for multi-speaker audio.

## WhisperX model size and VRAM guidance

- `small`
  Use for CPU-only machines or GPUs with up to about 8 GB VRAM.
- `medium`
  Recommended default for 8-12 GB VRAM.
- `large-v3`
  Use for 16 GB+ VRAM when you want the best quality.

Repository defaults:

- `model_name: medium`
- `device: auto`
- `compute_type: auto`
- `batch_size: auto`

## Automatic device and batch-size selection

`step2_ingest.py` now exposes:

- `asr.device`
  `auto`, `cuda`, or `cpu`
- `asr.compute_type`
  `auto`, `float16`, `float32`, or another WhisperX-supported value
- `asr.batch_size`
  `auto` or an integer

When `auto` is used:

- CUDA available -> `device=cuda`
- no CUDA -> `device=cpu`
- CUDA -> `compute_type=float16`
- CPU -> `compute_type=float32`

Batch size is selected from the detected environment:

- CPU -> `4`
- GPU + `small` -> `4` to `8`
- GPU + `medium` -> `8` to `16`
- GPU + `large-v3` -> `8` to `16`

If you hit CUDA OOM, lower one of these first:

1. `asr.batch_size`
2. `asr.model_name`
3. `asr.use_vocal_separation`
4. `asr.speaker_diarization`

## Hugging Face authentication

`speaker_diarization` is the optional advanced feature that most clearly depends on Hugging Face gated `pyannote` models.

`VAD` is part of the default step2 pipeline and should be understood as a base capability, not an optional enhancement toggle in the public docs.

You must do both:

1. accept the model terms on Hugging Face
2. provide a usable token locally

Recommended setup:

```bash
huggingface-cli login
export HF_TOKEN=your_token_here
```

The repository reads the token from the variable configured here:

```yaml
asr:
  hf_token_env: HF_TOKEN
```

## Required model access

For the default VAD-based step2 path, the repository may need:

- `pyannote/voice-activity-detection`

If you enable speaker diarization, it also needs:

- `pyannote/speaker-diarization-3.1`

## Recommended public rollout

Keep these off by default in the public first release:

- `use_vocal_separation`
- `speaker_diarization`

Then document them as optional upgrades after users confirm:

- WhisperX model loading works
- the default VAD path works
- alignment works
- Hugging Face auth works
- the machine has enough VRAM
