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

        segments_out = []
        for i, seg in enumerate(segments_in):
            rel = frame_by_seg.get(i)
            frame_paths = [rel] if rel else []
            av = av_by_seg.get(i, {})
            assigned = av.get("assigned_face")
            face_box = av.get("face_box")
            segments_out.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=(seg.get("text") or "").strip(),
                    speaker_id=assigned,
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
        """Group ASD tracks into speakers with appearance spans + a best frame."""
        if not ctx.asd_path.exists():
            return []
        dets = ctx.read_json(ctx.asd_path).get("detections", [])
        if not dets:
            return []

        # collect timestamps + pick the frame with widest lip opening as "best"
        times_by_face: dict[str, list[float]] = defaultdict(list)
        best_by_face: dict[str, dict] = {}
        for d in dets:
            fid = d["face_id"]
            times_by_face[fid].append(d["t"])
            if fid not in best_by_face or d["lip"] > best_by_face[fid]["lip"]:
                best_by_face[fid] = d

        speakers = []
        for fid in sorted(times_by_face):
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
