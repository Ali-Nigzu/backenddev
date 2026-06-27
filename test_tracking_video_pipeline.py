from pathlib import Path
from collections import defaultdict
import sys

import cv2
import pytest

from detection.detection_engine import detect
from embed.embed_engine import embed
from build_observation import build_observation

from track import TrackV2, TrackV2Config


# -------------------------
# INIT VIDEO
# -------------------------
video_path = "videoplayback.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    if "pytest" in sys.modules:
        pytest.skip(f"Cannot open video: {video_path}", allow_module_level=True)
    raise ValueError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120:
    fps = 10

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_path = "output/tracking_output.mp4"

Path(out_path).parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_w, frame_h)
)


# -------------------------
# MODELS
# -------------------------
embedder = embed()

tracker = TrackV2(TrackV2Config())

active_tracks = []
short_track_ids = {}
next_short_id = 1

frame_idx = 0

print("\n========================")
print("RUNNING TRACKING PIPELINE")
print("========================\n")


# =========================================================
# FRAME LOOP
# =========================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx * (1.0 / fps)

    frame_packet = {
        "frame_id": f"frame-{frame_idx}",
        "timestamp": timestamp,
        "image": frame
    }

    # -------------------------
    # DETECT
    # -------------------------
    detections = detect(frame_packet)

    # -------------------------
    # BUILD OBSERVATIONS
    # -------------------------
    observations_by_ts = defaultdict(list)

    for det in detections:

        emb_result = embedder.embed({
            "detection_id": det["detection_id"],
            "image": det["image"]
        })

        obs = build_observation(
            detection=det,
            embedding=emb_result
        )

        observations_by_ts[timestamp].append(obs)

    # -------------------------
    # TRACKING
    # -------------------------
    all_tracks, assignment_map = tracker.update(observations_by_ts)
    active_tracks = [t for t in all_tracks if t.state != "CLOSED"]

    # -------------------------
    # BUILD FRAME-LOCAL MAP (FIXED)
    # -------------------------
    detection_to_track = {}

    for t in active_tracks:
        for det_id in t.detection_history[-len(detections):]:
            detection_to_track[det_id] = t.runtime_track_id

    # -------------------------
    # SHORT IDS
    # -------------------------
    for t in active_tracks:
        if t.runtime_track_id not in short_track_ids:
            short_track_ids[t.runtime_track_id] = next_short_id
            next_short_id += 1

    # -------------------------
    # DRAW
    # -------------------------
    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        det_id = det["detection_id"]

        runtime_track_id = detection_to_track.get(det_id, None)

        if runtime_track_id is None:
            label = "T?"
        else:
            label = f"T{short_track_ids[runtime_track_id]}"

        text_x = int(x1)
        text_y = int(y1)

        if text_y < 20:
            text_y = int(y1 + 20)

        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(frame,
                      (text_x, text_y - th - 4),
                      (text_x + tw + 4, text_y + 4),
                      (0, 255, 0),
                      -1)

        cv2.putText(frame,
                    label,
                    (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2)

    writer.write(frame)
    frame_idx += 1


# -------------------------
# CLEANUP
# -------------------------
cap.release()
writer.release()

print("\n========================")
print("DONE")
print("========================")
print(f"Saved to:\n{out_path}")