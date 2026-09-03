"""Pipeline context: shared paths, config, and intermediate-file management."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


class PipelineContext:
    def __init__(self, video_path: str, config_path: str = "config.yaml"):
        self.video_path = Path(video_path).resolve()
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)

        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        root = Path(self.config.get("output_root", "output"))
        self.out_dir = (root / self.video_path.stem).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.frames_dir).mkdir(parents=True, exist_ok=True)

    # --- standard intermediate paths ---
    @property
    def audio_path(self) -> Path:
        return self.out_dir / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.out_dir / "transcript.json"

    @property
    def frames_dir(self) -> Path:
        return self.out_dir / "frames"

    @property
    def frames_index_path(self) -> Path:
        return self.out_dir / "frames_index.json"

    @property
    def context_cache_path(self) -> Path:
        return self.out_dir / "context_cache.json"

    @property
    def vocals_path(self) -> Path:
        return self.out_dir / "separated" / "vocals.wav"

    @property
    def diarize_path(self) -> Path:
        return self.out_dir / "diarize.json"

    @property
    def asd_path(self) -> Path:
        return self.out_dir / "asd_results.json"

    @property
    def av_match_path(self) -> Path:
        return self.out_dir / "av_match.json"

    @property
    def offscreen_path(self) -> Path:
        return self.out_dir / "offscreen.json"

    @property
    def final_path(self) -> Path:
        return self.out_dir / "final_output.json"

    # --- helpers ---
    @staticmethod
    def read_json(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, data) -> None:
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def probe_duration(self) -> float:
        """Video duration in seconds via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
