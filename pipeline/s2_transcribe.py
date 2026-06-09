"""S2 — Transcription with faster-whisper (native word-level timestamps).

WhisperX was the original plan, but its `av` build fails on this Windows
toolchain. faster-whisper produces word-level timestamps directly via
`word_timestamps=True`, so we use it and skip the separate wav2vec2 align step.
"""
from __future__ import annotations

import subprocess

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage
from utils import gpu


class TranscribeStage(Stage):
    name = "transcribe"

    def outputs(self, ctx):
        return [ctx.transcript_path]

    def _extract_audio(self, ctx: PipelineContext) -> None:
        if ctx.audio_path.exists():
            return
        # 16kHz mono PCM — what whisper expects
        cmd = [
            "ffmpeg", "-y", "-i", str(ctx.video_path),
            "-ac", "1", "-ar", "16000", "-vn",
            str(ctx.audio_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def execute(self, ctx: PipelineContext) -> None:
        from faster_whisper import WhisperModel

        cfg = ctx.config["transcribe"]
        device = cfg.get("device", "cuda")
        if device == "cuda" and not gpu.has_cuda():
            print("[transcribe] CUDA unavailable — falling back to CPU")
            device = "cpu"
        compute_type = cfg.get("compute_type", "int8")
        language = cfg.get("language")  # None => auto-detect

        self._extract_audio(ctx)

        # prefer separated vocals if S1 produced them
        audio_in = ctx.vocals_path if ctx.vocals_path.exists() else ctx.audio_path

        model = WhisperModel(
            cfg.get("model", "medium"),
            device=device,
            compute_type=compute_type,
        )
        seg_iter, info = model.transcribe(
            str(audio_in),
            language=language,
            beam_size=cfg.get("beam_size", 5),
            vad_filter=cfg.get("vad_filter", False),
            word_timestamps=cfg.get("align", True),
        )

        segments = []
        for s in seg_iter:
            words = []
            if s.words:
                words = [
                    {"start": w.start, "end": w.end, "word": w.word, "score": w.probability}
                    for w in s.words
                ]
            segments.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "words": words,
            })

        out = {"language": info.language, "segments": segments}
        ctx.write_json(ctx.transcript_path, out)
        del model
        gpu.free()
