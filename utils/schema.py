"""Output schema: dataclasses + validation for final_output.json."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

SPEECH_TYPES = {
    "onscreen",
    "offscreen",
    "flashback",
    "voiceover",
    "internal_monologue",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    speech_type: str = "onscreen"
    face_boxes: list = field(default_factory=list)
    is_active_speaker: bool = False
    context_description: str = ""
    frame_paths: list = field(default_factory=list)


@dataclass
class Speaker:
    id: str
    appearance_timestamps: list = field(default_factory=list)
    face_embeddings: list = field(default_factory=list)
    best_frames: list = field(default_factory=list)


@dataclass
class VideoAnalysis:
    video_path: str
    duration: float
    segments: list = field(default_factory=list)
    speakers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate(data: dict) -> list[str]:
    """Return a list of validation problems (empty == valid)."""
    problems: list[str] = []
    for key in ("video_path", "duration", "segments", "speakers"):
        if key not in data:
            problems.append(f"missing top-level key: {key}")

    for i, seg in enumerate(data.get("segments", [])):
        for key in ("start", "end", "text"):
            if key not in seg:
                problems.append(f"segment[{i}] missing {key}")
        st = seg.get("speech_type")
        if st is not None and st not in SPEECH_TYPES:
            problems.append(f"segment[{i}] invalid speech_type: {st}")
        if seg.get("start", 0) > seg.get("end", 0):
            problems.append(f"segment[{i}] start > end")
    return problems
