"""S3 — Active Speaker Detection via mediapipe FaceMesh.

Samples frames at a fixed FPS, detects faces, tracks them across frames by
bbox IoU, and measures lip-opening variance over a short window. Faces whose
mouth moves (variance above threshold) are flagged as the active speaker.

This is a pragmatic heuristic, not SyncNet — it does not use audio. AV matching
(S4) combines this with the audio timeline.
"""
from __future__ import annotations

import numpy as np

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage

# FaceMesh landmark indices for inner lips (upper/lower) and mouth corners.
_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_CORNER = 78
_RIGHT_CORNER = 308


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class ASDStage(Stage):
    name = "asd"

    def outputs(self, ctx):
        return [ctx.asd_path]

    def execute(self, ctx: PipelineContext) -> None:
        import cv2
        import mediapipe as mp

        cfg = ctx.config.get("asd", {})
        sample_fps = float(cfg.get("sample_fps", 5))
        max_faces = int(cfg.get("max_faces", 4))
        iou_thresh = float(cfg.get("iou_thresh", 0.3))
        # lip-opening std (normalized by face height) above this => speaking
        active_std_thresh = float(cfg.get("active_std_thresh", 0.012))
        window = int(cfg.get("window", 5))  # frames per side for variance

        cap = cv2.VideoCapture(str(ctx.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {ctx.video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(video_fps / sample_fps)))

        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # tracks: list of dicts {id, last_bbox, samples:[{t, bbox, lip}]}
        tracks: list[dict] = []
        next_id = 0
        frame_idx = 0

        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_idx += 1
                continue

            t = frame_idx / video_fps
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            dets = []
            if res.multi_face_landmarks:
                for lm in res.multi_face_landmarks:
                    pts = lm.landmark
                    xs = [p.x * w for p in pts]
                    ys = [p.y * h for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    face_h = max(1.0, y2 - y1)
                    lip_gap = abs(pts[_LOWER_LIP].y - pts[_UPPER_LIP].y) * h
                    lip_norm = lip_gap / face_h
                    dets.append({
                        "bbox": [x1, y1, x2, y2],
                        "lip": lip_norm,
                    })

            # match detections to existing tracks by IoU
            used = set()
            for det in dets:
                best, best_iou = None, iou_thresh
                for tr in tracks:
                    if tr["id"] in used:
                        continue
                    i = _iou(det["bbox"], tr["last_bbox"])
                    if i >= best_iou:
                        best, best_iou = tr, i
                if best is None:
                    best = {"id": next_id, "last_bbox": det["bbox"], "samples": []}
                    next_id += 1
                    tracks.append(best)
                best["last_bbox"] = det["bbox"]
                best["samples"].append({"t": t, "bbox": det["bbox"], "lip": det["lip"]})
                used.add(best["id"])

            frame_idx += 1

        cap.release()
        mesh.close()

        # decide active per sample using rolling std of lip openness within track
        results = []
        for tr in tracks:
            samples = tr["samples"]
            lips = [s["lip"] for s in samples]
            for i, s in enumerate(samples):
                lo = max(0, i - window)
                hi = min(len(lips), i + window + 1)
                local = lips[lo:hi]
                std = float(np.std(local)) if len(local) > 1 else 0.0
                results.append({
                    "t": round(s["t"], 3),
                    "face_id": f"face_{tr['id']:02d}",
                    "bbox": [round(v, 1) for v in s["bbox"]],
                    "lip": round(s["lip"], 4),
                    "lip_std": round(std, 4),
                    "is_active": std >= active_std_thresh,
                })

        results.sort(key=lambda r: (r["t"], r["face_id"]))
        ctx.write_json(ctx.asd_path, {
            "sample_fps": sample_fps,
            "video_fps": video_fps,
            "total_frames": total,
            "detections": results,
        })
        print(f"[asd] {len(tracks)} tracks, {len(results)} detections")
