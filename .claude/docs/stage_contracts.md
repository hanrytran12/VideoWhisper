# Stage Contracts & Output Schema

Per-stage I/O reference. Every stage subclasses `Stage` (`pipeline/stage_base.py`): it declares `outputs(ctx)` (the files it writes) and `execute(ctx)` (the work). A stage is skipped when all its outputs already exist, unless `--force`. All intermediate files live in `output/<video_stem>/` and are UTF-8 JSON (`ensure_ascii=False`).

## Stage table

| # | name | file | class | reads | writes |
|---|------|------|-------|-------|--------|
| 1 | `separate` | `pipeline/s1_separate.py` | `SeparateStage` | video → `audio.wav` | `separated/vocals.wav` |
| 2 | `transcribe` | `pipeline/s2_transcribe.py` | `TranscribeStage` | `vocals.wav` or `audio.wav` | `transcript.json` |
| 3 | `asd` | `pipeline/s3_asd.py` | `ASDStage` | video | `asd_results.json` |
| 4 | `av_match` | `pipeline/s4_av_match.py` | `AVMatchStage` | `transcript.json`, `asd_results.json` | `av_match.json` |
| 5 | `offscreen` | `pipeline/s5_offscreen.py` | `OffscreenStage` | `av_match.json` | `offscreen.json` |
| 6 | `frames` | `pipeline/s6_frames.py` | `FramesStage` | video, `transcript.json` | `frames_index.json` + `frames/*.jpg` |
| 7 | `vlm_context` | `pipeline/s7_vlm_context.py` | `VLMContextStage` | `frames_index.json` + frames | `context_cache.json` |
| 8 | `assemble` | `pipeline/s8_assemble.py` | `AssembleStage` | all of the above | `final_output.json` |

Stage 8 treats stages 3–5 (and 6/7) as optional: if an intermediate file is absent (e.g. an MVP-only `transcribe,frames,vlm_context,assemble` run), those segments fall back to `speech_type="onscreen"` with no face data.

## Per-stage detail

### S1 `separate` → `separated/vocals.wav`
Extracts mono 16 kHz `audio.wav` via `ffmpeg`, then runs `python -m demucs --two-stems vocals -d cpu -n <model>` into a temp tree, copies the produced `vocals.wav` into place, and deletes the temp tree. CPU-only. Disabled by default in `config.yaml` (`stages.separate: false`).

### S2 `transcribe` → `transcript.json`
Re-extracts `audio.wav` if missing, prefers `vocals.wav` when S1 ran. Loads `faster_whisper.WhisperModel(model, device, compute_type)` and transcribes with `word_timestamps=cfg.align`. Falls back to CPU if `device: cuda` but CUDA is absent. Calls `gpu.free()` when done.
```json
{ "language": "ja",
  "segments": [ { "start": 0.0, "end": 3.2, "text": "...",
    "words": [ { "start": 0.0, "end": 0.4, "word": "...", "score": 0.98 } ] } ] }
```
Segment index is positional (array order) and is the join key (`segment: i`) used by every later stage.

### S3 `asd` → `asd_results.json`
Samples frames at `sample_fps`, runs MediaPipe `FaceMesh`, derives a bbox + normalized lip-opening (landmarks 13/14/78/308) per face, tracks faces across frames by bbox IoU (`_iou`, threshold `iou_thresh`), then flags `is_active` when the rolling std of lip openness over `±window` samples ≥ `active_std_thresh`. Audio-free heuristic, not SyncNet.
```json
{ "sample_fps": 5, "video_fps": 25.0, "total_frames": 1234,
  "detections": [ { "t": 1.2, "face_id": "face_00", "bbox": [x1,y1,x2,y2],
    "lip": 0.031, "lip_std": 0.014, "is_active": true } ] }
```

### S4 `av_match` → `av_match.json`
Per segment, gathers ASD detections within `[start,end]`, computes each face's `score = active_samples / total_samples`, picks the top face, and assigns it only if `score ≥ av_match.min_score`.
```json
{ "matches": [ { "segment": 0, "start": 0.0, "end": 3.2,
    "assigned_face": "face_00", "score": 0.62,
    "faces_present": ["face_00","face_01"], "face_box": [x1,y1,x2,y2] } ] }
```
`assigned_face` / `face_box` are `null` when no face clears the threshold.

### S5 `offscreen` → `offscreen.json`
Pure classification over `av_match.json`: assigned face → `onscreen`; faces present but none active → `offscreen`; no faces present → `voiceover`.
```json
{ "segments": [ { "segment": 0, "speech_type": "onscreen" } ] }
```
Only emits `onscreen | offscreen | voiceover`. `flashback` / `internal_monologue` are valid in the schema but reserved for future VLM-driven refinement — nothing emits them today.

### S6 `frames` → `frames_index.json` (+ `frames/frame_<ms>.jpg`)
Seeks to each segment's midpoint (`CAP_PROP_POS_MSEC`) and writes one JPEG at `frames.jpeg_quality`. `frame_path` is relative to `out_dir` (portable). `frame_path` is `null` if the seek/read failed.
```json
[ { "segment": 0, "timestamp": 1.6, "frame_path": "frames/frame_00001600.jpg" } ]
```

### S7 `vlm_context` → `context_cache.json`
Builds `OllamaVLM(host, model, prompt, timeout, num_gpu, cache_path)`, pings `/api/tags` (raises if Ollama unreachable), then calls `vlm.describe(frame)` for each index entry. A failed describe stores `""` for that segment rather than aborting. See `.claude/docs/pipeline.md` for the two-layer cache (image-hash keys + `seg:<id>` keys) written into this one file.

### S8 `assemble` → `final_output.json`
Joins transcript + frames index + `seg:`-prefixed context + av_match + offscreen into `Segment` objects, builds the `Speaker` list from ASD tracks (`_build_speakers`, spans merged with `_merge_spans(gap=1.0)`), sets `duration` from `ctx.probe_duration()` (ffprobe; falls back to last segment end), runs `validate()` (warnings only, never aborts), and writes the file.

## Output schema (`utils/schema.py`)

```python
SPEECH_TYPES = {"onscreen", "offscreen", "flashback", "voiceover", "internal_monologue"}

@dataclass
class Segment:
    start: float; end: float; text: str
    speaker_id: Optional[str] = None      # av_match assigned_face, e.g. "face_00"
    speaker_name: Optional[str] = None     # never populated yet
    speech_type: str = "onscreen"          # must be in SPEECH_TYPES
    face_boxes: list = []                  # [matched bbox] or []
    is_active_speaker: bool = False        # True iff a face was assigned
    context_description: str = ""          # VLM text
    frame_paths: list = []                 # one relative JPEG path, or []

@dataclass
class Speaker:
    id: str                                # face_id, e.g. "face_00"
    appearance_timestamps: list = []       # [[start, end], ...] merged spans
    face_embeddings: list = []             # always [] (reserved)
    best_frames: list = []                 # always [] (reserved)

@dataclass
class VideoAnalysis:
    video_path: str; duration: float
    segments: list = []                    # list[Segment dicts]
    speakers: list = []                    # list[Speaker dicts]
```

`validate(data)` returns a list of problem strings (empty == valid). It checks: the four top-level keys exist; each segment has `start`/`end`/`text`; `speech_type` ∈ `SPEECH_TYPES`; `start ≤ end`. Problems are printed but never block the write — `final_output.json` is always produced.
