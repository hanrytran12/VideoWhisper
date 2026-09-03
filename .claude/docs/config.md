# Configuration Reference (`config.yaml`)

Loaded once by `PipelineContext.__init__` (`yaml.safe_load`). Every stage reads its own section with `cfg.get("key", default)`, so missing keys fall back to the inlined code defaults shown below — the file is a convenience, not strictly required. Default config path is `config.yaml`; override with `--config`.

## `output_root`
Base dir for all artifacts. Per-video outputs go to `<output_root>/<video_stem>/`. Default `output`.

## `transcribe` (S2)
| key | meaning | values / notes |
|-----|---------|----------------|
| `model` | faster-whisper model size | `small` \| `medium` \| `large-v3-turbo`. Bigger = more VRAM/slower. Default `medium`. |
| `device` | inference device | `cuda` \| `cpu`. Auto-falls back to CPU if CUDA absent. Default `cuda`. |
| `compute_type` | quantization | `int8` (lowest VRAM) \| `float16` (if room). Default `int8`. |
| `language` | spoken language | `vi` \| `ja` \| `en` \| `null` (auto). **Set explicitly** — auto-detect can crash on near-silent audio (see setup.md). |
| `vad_filter` | voice-activity trim | Keep `false`. `true` has zeroed out entire audio tracks here. |
| `align` | word-level timestamps | `true` → `word_timestamps=True`. |
| `batch_size` | — | **Present in file but unused** by `s2_transcribe.py`. No effect. |
| `beam_size` | decode beams | Not in the file; code default `5`. |

## `separate` (S1, disabled by default)
| key | meaning | notes |
|-----|---------|-------|
| `model` | Demucs model | `htdemucs`. CPU-only, 2-stem (vocals / no_vocals). |

Enable via `stages.separate: true` only if transcription quality suffers from background noise — it's slow on CPU.

## `asd` (S3)
| key | meaning | notes |
|-----|---------|-------|
| `sample_fps` | frame sampling rate | Higher = finer lip tracking, slower. Default `5`. |
| `max_faces` | MediaPipe `max_num_faces` | Default `4`. |
| `iou_thresh` | face-track association IoU | Lower = looser tracking. Default `0.3`. |
| `active_std_thresh` | lip-openness std → speaking | Higher = stricter "is talking". Default `0.012`. Main knob for active-speaker sensitivity. |
| `window` | ± samples for rolling std | Default `5`. |

## `av_match` (S4)
| key | meaning | notes |
|-----|---------|-------|
| `min_score` | min active-fraction to assign a face | `score = active/total` samples in segment span. Below this → no speaker assigned → segment becomes `offscreen`/`voiceover`. Default `0.3`. |

## `frames` (S6)
| key | meaning | notes |
|-----|---------|-------|
| `jpeg_quality` | output JPEG quality | 1–100. Default `90`. |

## `vlm` (S7)
| key | meaning | notes |
|-----|---------|-------|
| `ollama_host` | Ollama base URL | Default `http://127.0.0.1:11434`. Must be running. |
| `model` | VLM model tag | `minicpm-v` (pull via `ollama pull`). |
| `prompt` | description prompt | Currently Vietnamese; describes characters, setting, mood. Edit to change output language/detail. |
| `timeout` | per-request seconds | Default `120`. Raise for large models on CPU offload. |
| `num_gpu` | layers offloaded to GPU | Passed as Ollama `options.num_gpu`. Lower on tight VRAM (partial CPU offload). Default `20`. |

## `stages` (on/off flags)
Boolean per stage. When `--stages` is omitted, the run includes each stage whose flag is truthy (missing key defaults to `True`).

```yaml
stages:
  separate: false      # off by default — slow CPU step
  transcribe: true
  asd: true
  av_match: true
  offscreen: true
  frames: true
  vlm_context: true
  assemble: true
```

`--stages a,b,c` on the CLI overrides these flags entirely (runs exactly the listed subset, still in `STAGE_ORDER`).

## VRAM-tight profile (≈4 GB)
`transcribe.model: medium`, `compute_type: int8`, `vlm.num_gpu` low (partial CPU offload), `separate: false`. Stages run sequentially with `gpu.free()` between — see `.claude/docs/setup.md`.
