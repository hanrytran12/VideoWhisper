# Environment Setup & Windows ML-Dep Gotchas

Single-machine, GPU-tight setup. The hard-won fixes below are baked into `requirements.txt` and `config.yaml` defaults — read this before changing pinned versions or transcription settings.

## Target machine
- **GPU: RTX 3050 Laptop, ~4 GB VRAM.** The original project spec assumed ~12 GB; everything here is tuned down to fit 4 GB. Run one model at a time, unload between stages.
- **Python 3.11** in a venv at `.venv` (system Python is 3.13, which many ML wheels don't support yet).
- **OS:** Windows 11. FFmpeg + ffprobe on PATH (installed via `winget install Gyan.FFmpeg`).
- **Ollama** running locally for the VLM stage.

## Install

```powershell
# 1. venv on Python 3.11 (not 3.13)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. torch + torchaudio from the CUDA 12.1 index FIRST (not from requirements.txt)
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3. the rest
pip install -r requirements.txt

# 4. FFmpeg (if not already on PATH)
winget install Gyan.FFmpeg

# 5. Ollama + the VLM model
ollama serve            # leave running
ollama pull minicpm-v
```

Run the pipeline with the venv interpreter directly so you don't depend on activation:
```powershell
.\.venv\Scripts\python.exe run.py 7913973282827.mp4
```

## Known gotchas (already fixed — don't regress them)

**WhisperX dropped.** `whisperx==3.1.5` pulls `av` (PyAV), which tries to build from source on this Windows toolchain and fails (missing `avformat.lib`). Fix: use `faster-whisper==1.0.3` directly — it has native word-level timestamps (`word_timestamps=True`), so the separate wav2vec2 align step is unnecessary. Don't reintroduce whisperx.

**`numpy` must stay `<2`.** torch 2.2.2 cu121 wheels are built against numpy 1.x; numpy 2.x crashes them. This cascades: `opencv-python` is pinned to `4.9.0.80` because 4.13 requires numpy≥2. Keep `numpy<2.0` and don't bump opencv past the numpy-1.x-compatible line.

**Set `transcribe.language` explicitly.** faster-whisper auto-detect (`language: null`) can crash with `max() arg is an empty sequence` on near-silent or very short audio. Pin `vi`/`ja`/`en` in `config.yaml`.

**Keep `vad_filter: false`.** With `vad_filter: true` the VAD trimmed the *entire* track here, yielding 0 segments. If you get an empty transcript, first check the audio isn't silent (`ffmpeg ... volumedetect`) and confirm VAD is off.

**`protobuf==3.20.*`** is pinned for MediaPipe compatibility — newer protobuf breaks the FaceMesh import.

## VRAM strategy (the 4 GB constraint)
- `transcribe.compute_type: int8` and `model: medium` fit comfortably; `float16`/`large` may OOM.
- MiniCPM-V (`minicpm-v`, ~5.5 GB) won't fit fully in 4 GB — Ollama runs it with **partial CPU offload**. Tune `vlm.num_gpu` down if you hit OOM; raise `vlm.timeout` since CPU offload is slower.
- Stages run **sequentially** (`run.py` never parallelizes). `s2_transcribe.py` calls `gpu.free()` (`utils/gpu.py`) after unloading Whisper so VRAM is clear before the VLM stage.
- When adding a dependency, prefer prebuilt wheels (`pip install --only-binary :all: <pkg>`) to avoid Windows compile failures like the PyAV one above.

## Sanity checks
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # expect True
ffmpeg -version                                                                   # on PATH?
curl http://127.0.0.1:11434/api/tags                                              # Ollama up + model listed?
```
The README references `setup.bat`, `check_setup.py`, and `SETUP.md`, but none exist in the repo — use the steps above instead.
