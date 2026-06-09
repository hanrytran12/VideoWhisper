"""S6 — Frame extraction: one representative frame per segment (segment midpoint)."""
from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage


class FramesStage(Stage):
    name = "frames"

    def outputs(self, ctx):
        return [ctx.frames_index_path]

    def execute(self, ctx: PipelineContext) -> None:
        import cv2

        transcript = ctx.read_json(ctx.transcript_path)
        segments = transcript.get("segments", [])

        cap = cv2.VideoCapture(str(ctx.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {ctx.video_path}")
        quality = int(ctx.config.get("frames", {}).get("jpeg_quality", 90))

        index: list[dict] = []
        for i, seg in enumerate(segments):
            mid = (float(seg["start"]) + float(seg["end"])) / 2.0
            ms = int(mid * 1000)
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                index.append({"segment": i, "timestamp": mid, "frame_path": None})
                continue
            fname = f"frame_{ms:08d}.jpg"
            fpath = ctx.frames_dir / fname
            cv2.imwrite(
                str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            # store path relative to output dir for portability
            rel = fpath.relative_to(ctx.out_dir).as_posix()
            index.append({"segment": i, "timestamp": mid, "frame_path": rel})

        cap.release()
        ctx.write_json(ctx.frames_index_path, index)
