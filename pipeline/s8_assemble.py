"""S8 — Assemble transcript + frames + VLM context + ASD/AV-match into final_output.json.

Stages 3-5 are optional: if their output files are absent (e.g. MVP-only run),
segments fall back to onscreen defaults with no face data.
"""
from __future__ import annotations

from collections import defaultdict

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage
from utils.schema import Segment, Speaker, VideoAnalysis, validate


class AssembleStage(Stage):
    name = "assemble"

    def outputs(self, ctx):
        return [ctx.final_path]

    def execute(self, ctx: PipelineContext) -> None:
        transcript = ctx.read_json(ctx.transcript_path)
        segments_in = transcript.get("segments", [])

        # frames index: segment -> frame_path
        frame_by_seg: dict[int, str] = {}
        if ctx.frames_index_path.exists():
            for entry in ctx.read_json(ctx.frames_index_path):
                frame_by_seg[int(entry["segment"])] = entry.get("frame_path")

        # context descriptions keyed "seg:<id>"
        ctx_by_seg: dict[str, str] = {}
        if ctx.context_cache_path.exists():
            cache = ctx.read_json(ctx.context_cache_path)
            ctx_by_seg = {
                k.split(":", 1)[1]: v
                for k, v in cache.items()
                if k.startswith("seg:")
            }

        # AV match: segment -> {assigned_face, score, face_box}
        av_by_seg: dict[int, dict] = {}
        if ctx.av_match_path.exists():
            for m in ctx.read_json(ctx.av_match_path).get("matches", []):
                av_by_seg[int(m["segment"])] = m

        # off-screen: segment -> speech_type
        speech_by_seg: dict[int, str] = {}
        if ctx.offscreen_path.exists():
            for r in ctx.read_json(ctx.offscreen_path).get("segments", []):
                speech_by_seg[int(r["segment"])] = r["speech_type"]

        # diarization: segment -> voice speaker_id (primary identity)
        voice_by_seg: dict[int, str] = {}
        if ctx.diarize_path.exists():
            for r in ctx.read_json(ctx.diarize_path).get("segments", []):
                if r.get("speaker_id"):
                    voice_by_seg[int(r["segment"])] = r["speaker_id"]

        segments_out = []
        for i, seg in enumerate(segments_in):
            rel = frame_by_seg.get(i)
            frame_paths = [rel] if rel else []
            av = av_by_seg.get(i, {})
            assigned = av.get("assigned_face")
            face_box = av.get("face_box")
            # voice is primary; fall back to face id when no voice cluster
            voice_id = voice_by_seg.get(i)
            speaker_id = voice_id or assigned
            segments_out.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=(seg.get("text") or "").strip(),
                    speaker_id=speaker_id,
                    speech_type=speech_by_seg.get(i, "onscreen"),
                    face_boxes=[face_box] if face_box else [],
                    is_active_speaker=bool(assigned),
                    context_description=ctx_by_seg.get(str(i), ""),
                    frame_paths=frame_paths,
                )
            )

        speakers = self._build_speakers(ctx, segments_out)

        try:
            duration = ctx.probe_duration()
        except Exception:
            duration = segments_out[-1].end if segments_out else 0.0

        analysis = VideoAnalysis(
            video_path=str(ctx.video_path),
            duration=duration,
            segments=[s.__dict__ for s in segments_out],
            speakers=[s.__dict__ for s in speakers],
        )
        data = analysis.to_dict()

        problems = validate(data)
        if problems:
            print("[assemble] schema warnings:")
            for p in problems:
                print(f"  - {p}")

        ctx.write_json(ctx.final_path, data)
        print(
            f"[assemble] wrote {ctx.final_path} "
            f"({len(segments_out)} segments, {len(speakers)} speakers)"
        )

    def _build_speakers(self, ctx, segments_out) -> list[Speaker]:
        """Build the speaker list. Voice clusters (diarize) are primary; ASD
        face tracks are added for any face ids not already covered by a voice."""
        speakers: list[Speaker] = []
        seen: set[str] = set()

        # voice speakers from diarization: spans from segment start/end
        if ctx.diarize_path.exists():
            spans_by_voice: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for r in ctx.read_json(ctx.diarize_path).get("segments", []):
                sid = r.get("speaker_id")
                if sid:
                    spans_by_voice[sid].append((float(r["start"]), float(r["end"])))
            for sid in sorted(spans_by_voice):
                merged = _merge_intervals(sorted(spans_by_voice[sid]), gap=1.0)
                speakers.append(
                    Speaker(
                        id=sid,
                        appearance_timestamps=[[round(a, 2), round(b, 2)] for a, b in merged],
                        face_embeddings=[],
                        best_frames=[],
                    )
                )
                seen.add(sid)

        # face speakers from ASD tracks (only ids not already a voice)
        if ctx.asd_path.exists():
            dets = ctx.read_json(ctx.asd_path).get("detections", [])
            times_by_face: dict[str, list[float]] = defaultdict(list)
            for d in dets:
                times_by_face[d["face_id"]].append(d["t"])
            for fid in sorted(times_by_face):
                if fid in seen:
                    continue
                spans = _merge_spans(sorted(times_by_face[fid]), gap=1.0)
                speakers.append(
                    Speaker(
                        id=fid,
                        appearance_timestamps=[[round(a, 2), round(b, 2)] for a, b in spans],
                        face_embeddings=[],
                        best_frames=[],
                    )
                )
        return speakers


def _merge_intervals(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Merge sorted [start, end] intervals, joining those within `gap` seconds."""
    if not intervals:
        return []
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _merge_spans(times: list[float], gap: float) -> list[tuple[float, float]]:
    """Merge sorted sample timestamps into [start, end] spans, joining gaps <= `gap`."""
    if not times:
        return []
    spans = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev <= gap:
            prev = t
        else:
            spans.append((start, prev))
            start = prev = t
    spans.append((start, prev))
    return spans
