"""S5 — Off-screen speech detection.

Classifies each segment's speech_type using AV-match results:
  - has an assigned active face            -> onscreen
  - faces present but none active          -> offscreen (someone on screen, but not speaking the line)
  - no faces present at all during segment -> voiceover (narration over a faceless shot)

This is a coarse heuristic. flashback / internal_monologue need narrative cues
beyond what we detect here, so we leave those for a future VLM-driven refinement.
"""
from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage


class OffscreenStage(Stage):
    name = "offscreen"

    def outputs(self, ctx):
        return [ctx.offscreen_path]

    def execute(self, ctx: PipelineContext) -> None:
        av = ctx.read_json(ctx.av_match_path)
        matches = av.get("matches", [])

        results = []
        for m in matches:
            if m.get("assigned_face"):
                speech_type = "onscreen"
            elif m.get("faces_present"):
                speech_type = "offscreen"
            else:
                speech_type = "voiceover"
            results.append({
                "segment": m["segment"],
                "speech_type": speech_type,
            })

        ctx.write_json(ctx.offscreen_path, {"segments": results})
        counts: dict[str, int] = {}
        for r in results:
            counts[r["speech_type"]] = counts.get(r["speech_type"], 0) + 1
        print(f"[offscreen] speech_type counts: {counts}")
