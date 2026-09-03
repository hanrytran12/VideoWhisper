# Pipeline Internals

How the orchestrator, context, stage lifecycle, and caching fit together. For per-stage I/O shapes see `.claude/docs/stage_contracts.md`; for tunables see `.claude/docs/config.md`.

## Entry point — `run.py`

`main(argv)` parses args (`video`, `--config`, `--stages`, `--force`), builds a `PipelineContext`, computes the selected stage list, and runs each enabled stage in fixed order.

- `STAGE_REGISTRY` — `dict[name -> Stage class]`.
- `STAGE_ORDER` — the canonical execution order: `separate, transcribe, asd, av_match, offscreen, frames, vlm_context, assemble`.
- Stage selection: `--stages a,b,c` runs exactly that subset; otherwise every stage whose `stages.<name>` flag in config is truthy (default `True` if the key is missing).
- Unknown stage names exit with code `2`. Any exception inside a stage prints a traceback and exits `1`. The loop always walks `STAGE_ORDER`, so a subset still runs in dependency order regardless of the order you list it in `--stages`.

To add a stage: write `pipeline/sN_<name>.py` subclassing `Stage`, register it in both `STAGE_REGISTRY` and `STAGE_ORDER`, add a `stages.<name>` flag and a config section, and add any new intermediate path as a property on `PipelineContext`.

## `PipelineContext` — `pipeline/context.py`

Constructed once per run. Responsibilities:
- Resolves `video_path` (raises `FileNotFoundError` if missing).
- Loads `config` via `yaml.safe_load`.
- Computes the per-video output dir: `out_dir = <output_root>/<video_stem>/`, created on init along with `frames/`.
- Exposes every intermediate file as a `@property` (single source of truth for paths — never hardcode these elsewhere).
- JSON helpers `read_json` / `write_json` (UTF-8, `ensure_ascii=False`, indent 2).
- `probe_duration()` — video length in seconds via `ffprobe`.

### Intermediate file map (all under `out_dir`)
| property | path |
|----------|------|
| `audio_path` | `audio.wav` |
| `vocals_path` | `separated/vocals.wav` |
| `transcript_path` | `transcript.json` |
| `asd_path` | `asd_results.json` |
| `av_match_path` | `av_match.json` |
| `offscreen_path` | `offscreen.json` |
| `frames_dir` | `frames/` |
| `frames_index_path` | `frames_index.json` |
| `context_cache_path` | `context_cache.json` |
| `final_path` | `final_output.json` |

## Stage lifecycle — `pipeline/stage_base.py`

```python
class Stage(ABC):
    name = "stage"
    def outputs(self, ctx) -> list[Path]: return []      # files this stage writes
    def is_cached(self, ctx) -> bool:                      # all outputs exist?
        outs = self.outputs(ctx); return bool(outs) and all(p.exists() for p in outs)
    @abstractmethod
    def execute(self, ctx) -> None: ...                    # the actual work
    def run(self, ctx, force=False):
        if not force and self.is_cached(ctx): print(f"[{name}] cached — skip"); return
        self.execute(ctx)
```

Caching is purely file-existence based — there is no content hashing or timestamp check. If an output file exists, the stage is skipped (unless `--force`). To re-run a single stage, delete its output file or pass `--force`.

## Sequential execution & VRAM

Stages run strictly one at a time, never in parallel. This is deliberate: the target machine has only ~4 GB VRAM, so only one model can be resident at once. `s2_transcribe.py` calls `gpu.free()` (`utils/gpu.py`: `gc.collect()` + `torch.cuda.empty_cache()`/`ipc_collect()`) after unloading Whisper, before later GPU work (Ollama VLM). See `.claude/docs/setup.md` for the full VRAM strategy.

## Two-layer VLM cache — `context_cache.json`

`context_cache.json` holds **two kinds of keys in one file**:

1. **Image-hash entries** — written by `OllamaVLM` (`utils/ollama_client.py`). Key = `sha256(image_bytes)[:16]`, value = description text. On every `describe()` call, a cache hit returns instantly; a miss calls Ollama `/api/generate` and persists the result. This dedupes identical frames and survives across runs.
2. **Per-segment entries** — written by `VLMContextStage` after processing the frames index. Key = `seg:<segment_id>`, value = that segment's description.

`AssembleStage` reads **only** `seg:`-prefixed keys. Consequence: re-running `frames` (new midpoints/JPEGs) invalidates per-segment context, but the image-hash layer stays hot, so unchanged frames are not re-described. Note `VLMContextStage` reads `vlm._cache` directly to merge the two layers — they intentionally coexist in the same file.

## Typical invocations

```powershell
.\.venv\Scripts\python.exe run.py 7913973282827.mp4              # all enabled stages
.\.venv\Scripts\python.exe run.py video.mp4 --stages transcribe,frames,vlm_context,assemble  # MVP path
.\.venv\Scripts\python.exe run.py video.mp4 --force              # ignore cache, re-run
```

The MVP path (`transcribe,frames,vlm_context,assemble`) skips face/speaker analysis entirely and still produces a valid `final_output.json` — segments default to `onscreen` with empty `speakers`.
