"""S4 — AV matching (heuristic).

For each transcript segment, find the face that is most often active during the
segment's time span (using ASD detections). Assign that face_id with a score =
fraction of in-span samples where the face was active. No active face => no match.
"""
from __future__ import annotations

from collections import defaultdict

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage


class AVMatchStage(Stage):
    name = "av_match"

    def outputs(self, ctx):
        return [ctx.av_match_path]

    def execute(self, ctx: PipelineContext) -> None:
        transcript = ctx.read_json(ctx.transcript_path)
        segments = transcript.get("segments", [])
        asd = ctx.read_json(ctx.asd_path)
        dets = asd.get("detections", [])

        cfg = ctx.config.get("av_match", {})
        min_score = float(cfg.get("min_score", 0.3))

        matches = []
        for i, seg in enumerate(segments):
            start, end = float(seg["start"]), float(seg["end"])
            # samples within the segment span
            in_span = [d for d in dets if start <= d["t"] <= end]
            # per-face: count total samples and active samples in span
            total_by_face: dict[str, int] = defaultdict(int)
            active_by_face: dict[str, int] = defaultdict(int)
            box_by_face: dict[str, list] = {}
            for d in in_span:
                fid = d["face_id"]
                total_by_face[fid] += 1
                if d["is_active"]:
                    active_by_face[fid] += 1
                box_by_face.setdefault(fid, d["bbox"])

            best_face, best_score = None, 0.0
            for fid, total in total_by_face.items():
                score = active_by_face[fid] / total if total else 0.0
                if score > best_score:
                    best_face, best_score = fid, score

            faces_present = sorted(total_by_face.keys())
            assigned = best_face if best_score >= min_score else None
            matches.append({
                "segment": i,
                "start": start,
                "end": end,
                "assigned_face": assigned,
                "score": round(best_score, 3),
                "faces_present": faces_present,
                "face_box": box_by_face.get(assigned) if assigned else None,
            })

        ctx.write_json(ctx.av_match_path, {"matches": matches})
        n_assigned = sum(1 for m in matches if m["assigned_face"])
        print(f"[av_match] {n_assigned}/{len(matches)} segments matched to a face")
