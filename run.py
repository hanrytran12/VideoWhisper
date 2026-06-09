"""CLI entrypoint for the video analysis pipeline.

Usage:
    python run.py path/to/video.mp4
    python run.py video.mp4 --stages transcribe,frames
    python run.py video.mp4 --force
    python run.py video.mp4 --config config.yaml
"""
from __future__ import annotations

import argparse
import sys

from pipeline.context import PipelineContext
from pipeline.s1_separate import SeparateStage
from pipeline.s2_transcribe import TranscribeStage
from pipeline.s3_asd import ASDStage
from pipeline.s4_av_match import AVMatchStage
from pipeline.s5_offscreen import OffscreenStage
from pipeline.s6_frames import FramesStage
from pipeline.s7_vlm_context import VLMContextStage
from pipeline.s8_assemble import AssembleStage

# stage name -> class, in execution order
STAGE_REGISTRY = {
    "separate": SeparateStage,
    "transcribe": TranscribeStage,
    "asd": ASDStage,
    "av_match": AVMatchStage,
    "offscreen": OffscreenStage,
    "frames": FramesStage,
    "vlm_context": VLMContextStage,
    "assemble": AssembleStage,
}
STAGE_ORDER = [
    "separate", "transcribe", "asd", "av_match",
    "offscreen", "frames", "vlm_context", "assemble",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Video analysis pipeline")
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--stages",
        default=None,
        help="comma-separated subset to run (default: all enabled in config)",
    )
    ap.add_argument(
        "--force", action="store_true", help="ignore cache, re-run stages"
    )
    args = ap.parse_args(argv)

    ctx = PipelineContext(args.video, args.config)
    enabled = ctx.config.get("stages", {})

    if args.stages:
        selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    else:
        selected = [s for s in STAGE_ORDER if enabled.get(s, True)]

    unknown = [s for s in selected if s not in STAGE_REGISTRY]
    if unknown:
        print(f"unknown stages: {unknown}", file=sys.stderr)
        return 2

    print(f"video: {ctx.video_path}")
    print(f"output: {ctx.out_dir}")
    print(f"stages: {selected}")
    
    import traceback
    for name in STAGE_ORDER:
        if name not in selected:
            continue
        try:
            print(f"\n>>> Starting stage: {name}")
            STAGE_REGISTRY[name]().run(ctx, force=args.force)
            print(f">>> Finished stage: {name}")
        except Exception as e:
            print(f"\n!!! ERROR in stage {name}: {e}")
            traceback.print_exc()
            return 1

    print(f"\nDone. Final output: {ctx.final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
