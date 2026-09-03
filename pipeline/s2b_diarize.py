"""S2b — Voice diarization: assign a voice-based speaker_id per transcript segment.

Embeds each segment's audio span with Resemblyzer (d-vector), then clusters the
embeddings to group segments by voice. Speaker count is auto-detected via
agglomerative clustering with a cosine-distance threshold — no need to know how
many speakers up front.

CPU-only (Resemblyzer is light). Runs after transcribe, before face stages.
Output is the *primary* speaker identity; face stages (ASD/AV-match) refine it.
"""
from __future__ import annotations

import numpy as np

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage


class DiarizeStage(Stage):
    name = "diarize"

    def outputs(self, ctx):
        return [ctx.diarize_path]

    def execute(self, ctx: PipelineContext) -> None:
        import librosa
        from resemblyzer import VoiceEncoder

        transcript = ctx.read_json(ctx.transcript_path)
        segments = transcript.get("segments", [])

        cfg = ctx.config.get("diarize", {})
        dist_thresh = float(cfg.get("distance_threshold", 0.30))
        min_dur = float(cfg.get("min_segment_sec", 0.4))

        # prefer separated vocals if S1 produced them
        audio_in = ctx.vocals_path if ctx.vocals_path.exists() else ctx.audio_path
        if not audio_in.exists():
            raise FileNotFoundError(
                f"no audio at {audio_in} — run transcribe (extracts audio.wav) first"
            )

        wav, sr = librosa.load(str(audio_in), sr=16000, mono=True)
        encoder = VoiceEncoder("cpu")

        embeds: list[np.ndarray] = []
        embed_idx: list[int] = []  # segment index for each embedding
        for i, seg in enumerate(segments):
            start, end = float(seg["start"]), float(seg["end"])
            if end - start < min_dur:
                continue
            a = int(start * sr)
            b = int(end * sr)
            clip = wav[a:b]
            if clip.size < int(min_dur * sr):
                continue
            try:
                emb = encoder.embed_utterance(clip)
            except Exception:
                continue
            embeds.append(emb)
            embed_idx.append(i)

        labels = self._cluster(embeds, dist_thresh)

        # map each clustered segment -> "spk_NN"; ensure stable label ordering
        seg_speaker: dict[int, str] = {}
        for seg_i, lab in zip(embed_idx, labels):
            seg_speaker[seg_i] = f"spk_{lab:02d}"

        out_segments = []
        for i, seg in enumerate(segments):
            out_segments.append({
                "segment": i,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker_id": seg_speaker.get(i),  # None for too-short/skipped
            })

        n_speakers = len(set(seg_speaker.values()))
        ctx.write_json(ctx.diarize_path, {
            "n_speakers": n_speakers,
            "segments": out_segments,
        })
        n_assigned = sum(1 for s in out_segments if s["speaker_id"])
        print(f"[diarize] {n_speakers} voices over {n_assigned}/{len(out_segments)} segments")

    @staticmethod
    def _cluster(embeds: list[np.ndarray], dist_thresh: float) -> list[int]:
        """Agglomerative clustering on cosine distance. Auto speaker count.

        Returns a label per embedding. Labels are relabeled by first appearance
        so spk_00 is the first speaker heard.
        """
        if not embeds:
            return []
        if len(embeds) == 1:
            return [0]

        from sklearn.cluster import AgglomerativeClustering

        X = np.vstack(embeds)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=dist_thresh,
        )
        raw = clustering.fit_predict(X)

        # relabel by first appearance order for stable, readable IDs
        remap: dict[int, int] = {}
        labels = []
        for r in raw:
            if r not in remap:
                remap[r] = len(remap)
            labels.append(remap[r])
        return labels
