# VideoWhisper

Local, single-machine video analysis pipeline. Runs a video through sequential AI stages — transcription, active-speaker detection, audio-visual matching, off-screen-speech classification, frame extraction, and VLM visual context — and emits one structured `final_output.json`: transcript segments enriched with speaker identity, speech type, and a visual description per segment.

Everything runs locally (faster-whisper for ASR, MediaPipe for faces, Ollama for the VLM). No cloud APIs.

## Run it

```powershell
# use the project venv interpreter (Python 3.11)
.\.venv\Scripts\python.exe run.py 7913973282827.mp4
```

| command | effect |
|---------|--------|
| `python run.py <video>` | run all stages enabled in `config.yaml` |
| `python run.py <video> --stages transcribe,frames,vlm_context,assemble` | run a subset (this = the MVP path) |
| `python run.py <video> --force` | ignore cache, re-run |
| `python run.py <video> --config <path>` | use a different config |

**Prerequisites:** Python 3.11 venv at `.venv`; FFmpeg + ffprobe on PATH; Ollama running (`ollama serve`) with the VLM model pulled (`ollama pull minicpm-v`); CUDA 12.1 optional (falls back to CPU). Full setup → `.claude/docs/setup.md`.

Output lands in `output/<video_stem>/` — intermediate JSON per stage plus the final `final_output.json`.

## Architecture at a glance

Eight stages run **strictly sequentially** (one model in VRAM at a time — the target is a 4 GB laptop GPU). Each stage writes its own intermediate JSON and is **cached by file existence**: if its output already exists it's skipped unless `--force`.

```
separate → transcribe → asd → av_match → offscreen → frames → vlm_context → assemble
  (s1)        (s2)       (s3)    (s4)        (s5)       (s6)      (s7)         (s8)
```

- `separate` is **off by default** (slow CPU Demucs step).
- Stages 3–5 (face/speaker analysis) are optional — the MVP path `transcribe,frames,vlm_context,assemble` still produces a valid output (segments default to `onscreen`, empty `speakers`).

## Where things live

| path | what |
|------|------|
| `run.py` | CLI entry point, `STAGE_REGISTRY` + `STAGE_ORDER` |
| `pipeline/context.py` | `PipelineContext` — config load, `out_dir`, all intermediate-file paths |
| `pipeline/stage_base.py` | `Stage` ABC — `outputs`/`is_cached`/`execute`/`run` caching contract |
| `pipeline/s1_separate.py` … `s8_assemble.py` | the eight stages (one file each) |
| `utils/schema.py` | `Segment` / `Speaker` / `VideoAnalysis` dataclasses + `validate()` |
| `utils/ollama_client.py` | `OllamaVLM` HTTP client + on-disk image-hash cache |
| `utils/gpu.py` | `free()` / `has_cuda()` VRAM helpers |
| `config.yaml` | all tunables + per-stage on/off flags |
| `output/<stem>/` | per-video artifacts (gitignored) |

## Where to read next (progressive disclosure)

| question | doc |
|----------|-----|
| How does orchestration / context / caching / the VLM two-layer cache work? | `.claude/docs/pipeline.md` |
| What does each stage read & write? What's the JSON/output schema? | `.claude/docs/stage_contracts.md` |
| What does each `config.yaml` field do? How do I tune for VRAM? | `.claude/docs/config.md` |
| How do I install it? Why these pinned versions? Build errors? | `.claude/docs/setup.md` |

## Critical gotchas (detail in `setup.md`)

- **numpy must stay `<2`** (torch cu121 built against numpy 1.x); opencv is pinned to `4.9.0.80` for the same reason.
- **No WhisperX** — its PyAV build fails on Windows; we use `faster-whisper` directly.
- **Set `transcribe.language` explicitly** — auto-detect can crash on near-silent audio.
- **Keep `vad_filter: false`** — `true` has zeroed entire audio tracks (0 segments).
- Stages run one at a time and call `gpu.free()` between GPU stages — don't parallelize.

## Conventions for changes

- All intermediate paths come from `PipelineContext` properties — never hardcode a path.
- Read/write JSON via `ctx.read_json` / `ctx.write_json` (UTF-8, `ensure_ascii=False`).
- Adding a stage: new `pipeline/sN_*.py` subclassing `Stage`, register in `STAGE_REGISTRY` **and** `STAGE_ORDER`, add a `stages.<name>` flag + config section, add any new path as a `PipelineContext` property.
- `segment` index is positional (transcript array order) and is the join key across every stage.
