"""S1 — Audio separation with Demucs (CPU). Produces vocals.wav for cleaner transcription."""
from __future__ import annotations

import shutil
import subprocess
import sys

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage


class SeparateStage(Stage):
    name = "separate"

    def outputs(self, ctx):
        return [ctx.vocals_path]

    def _extract_audio(self, ctx: PipelineContext) -> None:
        if ctx.audio_path.exists():
            return
        cmd = [
            "ffmpeg", "-y", "-i", str(ctx.video_path),
            "-ac", "1", "-ar", "16000", "-vn",
            str(ctx.audio_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def execute(self, ctx: PipelineContext) -> None:
        self._extract_audio(ctx)
        cfg = ctx.config.get("separate", {})
        model = cfg.get("model", "htdemucs")
        sep_root = ctx.out_dir / "demucs_tmp"

        # demucs --two-stems vocals -d cpu -o <out> <audio>
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-d", "cpu",
            "-n", model,
            "-o", str(sep_root),
            str(ctx.audio_path),
        ]
        subprocess.run(cmd, check=True)

        # demucs writes <sep_root>/<model>/<audio_stem>/vocals.wav
        produced = sep_root / model / ctx.audio_path.stem / "vocals.wav"
        if not produced.exists():
            raise RuntimeError(f"demucs did not produce {produced}")
        ctx.vocals_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, ctx.vocals_path)
        shutil.rmtree(sep_root, ignore_errors=True)
