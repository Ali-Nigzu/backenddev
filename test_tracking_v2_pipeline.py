"""Minimal real integration runner for Detect -> Track -> Event -> Demographic -> Assemble."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any

import cv2

from assemble import Assemble
from demographics import Demographic
from detect import Detect
from events import Event
from load.load import load
from track import Track

DEFAULT_REPLAY_PATH = "output/tracking_replay.mp4"
DEFAULT_OUTPUT_BATCH_PATH = "output/output_batch.json"
DEFAULT_LOCAL_VIDEO_PATH = Path(__file__).resolve().parent / "test03fps.mp4"
SOURCE_URI = "gs://camostesting/" "Orgs/Sites/Devices/TestCamera/"
TIMEFRAME = {
    "start": "2026-08-04T11:38:55.000Z",
    "end": "2026-08-04T11:39:05.000Z",
}
SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "TestAdminSA.json"
LINE_CONFIG = {
    "point_a": {"x": 100.0, "y": 300.0},
    "point_b": {"x": 700.0, "y": 300.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Detect -> Track -> Event -> Demographic -> Assemble pipeline."
    )
    parser.add_argument(
        "--output", default=DEFAULT_REPLAY_PATH, help="Annotated replay path"
    )
    parser.add_argument(
        "--output-batch",
        default=DEFAULT_OUTPUT_BATCH_PATH,
        help="OutputBatch JSON path",
    )
    parser.add_argument(
        "--local-video",
        type=Path,
        default=DEFAULT_LOCAL_VIDEO_PATH,
        help="Local MP4 used automatically if GCS loading fails",
    )
    return parser.parse_args()


def load_local_video(video_path: Path) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Local fallback video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open local fallback video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 3.0

    frames = []
    frame_index = 0
    try:
        while True:
            decoded, bgr_image = capture.read()
            if not decoded:
                break
            frames.append(
                {
                    "frame_id": f"local:{video_path.name}:{frame_index:08d}",
                    "timestamp": frame_index / fps,
                    "image": cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB),
                }
            )
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames decoded from local fallback video: {video_path}")

    print("Frame source: LOCAL VIDEO")
    print(f"Local video: {video_path.resolve()}")
    print(f"Frames loaded: {len(frames)}")
    print(f"FPS: {fps:.3f}")
    return {"frames": frames}


def get_frame_batch(local_video_path: Path) -> dict[str, Any]:
    try:
        print("Frame source: trying GCS")
        with SERVICE_ACCOUNT_PATH.open(encoding="utf-8") as file:
            service_account_info = json.load(file)

        frame_batch = load(
            SOURCE_URI,
            TIMEFRAME,
            service_account_info,
        )
        if not frame_batch["frames"]:
            raise ValueError(
                "No timestamp-named JPG frames found in the configured timeframe"
            )

        print("Frame source: GCS")
        print(f"Frames loaded: {len(frame_batch['frames'])}")
        return frame_batch
    except Exception as exc:
        print("GCS frame load failed:")
        print(f"{type(exc).__name__}: {exc}")
        return load_local_video(local_video_path)


def get_frame_size(frame_batch: dict[str, Any]) -> tuple[int, int]:
    first_image = frame_batch["frames"][0]["image"]
    return (first_image.shape[1], first_image.shape[0])


def get_replay_fps(frame_batch: dict[str, Any]) -> float:
    timestamps = [float(frame["timestamp"]) for frame in frame_batch["frames"]]
    positive_intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if later - earlier > 0.0
    ]
    if not positive_intervals:
        raise ValueError(
            "Cannot derive replay FPS from fewer than two distinct frame timestamps"
        )
    return 1.0 / median(positive_intervals)


def create_video_writer(
    output_path: Path, fps: float, frame_size: tuple[int, int]
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Cannot open replay output: {output_path}")
    return writer


def draw_replay(
    frame_batch: dict[str, Any],
    detection_batch: dict[str, Any],
    track_batch: dict[str, Any],
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> None:
    track_ids_by_timestamp_centre = {
        (
            float(point["timestamp"]),
            round(float(point["centre"]["x"]), 6),
            round(float(point["centre"]["y"]), 6),
        ): str(track["track_id"])
        for track in track_batch["tracks"]
        for point in track["path"]
    }
    writer = create_video_writer(output_path, fps, frame_size)
    try:
        for frame, frame_detections in zip(
            frame_batch["frames"], detection_batch["detections"], strict=True
        ):
            bgr_output = cv2.cvtColor(frame["image"], cv2.COLOR_RGB2BGR)
            timestamp = float(frame_detections["timestamp"])
            for detection in frame_detections["detections"]:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = (
                    int(round(float(bbox[key]))) for key in ("x1", "y1", "x2", "y2")
                )
                centre = detection["centre"]
                track_id = track_ids_by_timestamp_centre.get(
                    (
                        timestamp,
                        round(float(centre["x"]), 6),
                        round(float(centre["y"]), 6),
                    ),
                    "?",
                )
                cv2.rectangle(bgr_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    bgr_output,
                    f"Track {track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.line(
                bgr_output,
                (int(LINE_CONFIG["point_a"]["x"]), int(LINE_CONFIG["point_a"]["y"])),
                (int(LINE_CONFIG["point_b"]["x"]), int(LINE_CONFIG["point_b"]["y"])),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(bgr_output)
    finally:
        writer.release()


def write_output_batch(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    replay_path = Path(args.output)
    output_batch_path = Path(args.output_batch)

    frame_batch = get_frame_batch(args.local_video)

    frame_size = get_frame_size(frame_batch)
    fps = get_replay_fps(frame_batch)

    detection_batch = Detect()(frame_batch)
    track_started = time.perf_counter()
    track_batch = Track()(detection_batch)
    track_seconds = time.perf_counter() - track_started

    event_batch = Event()(
        track_batch,
        LINE_CONFIG,
    )

    track_lengths = [len(track["path"]) for track in track_batch["tracks"]]
    average_length = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
    median_length = median(track_lengths) if track_lengths else 0.0
    assigned_observations = sum(track_lengths)
    detection_count = sum(
        len(frame_detections["detections"])
        for frame_detections in detection_batch["detections"]
    )
    print(f"Track runtime: {track_seconds:.3f}s")
    print(f"Tracks returned: {len(track_lengths)}")
    print(f"Average observations per Track: {average_length:.2f}")
    print(f"Median observations per Track: {median_length:.2f}")
    print(
        f"Short Tracks (<=3 observations): {sum(length <= 3 for length in track_lengths)}"
    )
    print(f"Shortest Track: {min(track_lengths) if track_lengths else 0}")
    print(f"Longest Track: {max(track_lengths) if track_lengths else 0}")
    print(f"Unassigned/? detections: {max(detection_count - assigned_observations, 0)}")
    print(f"Events produced: {len(event_batch['events'])}")

    demographics_batch = Demographic()(event_batch, frame_batch)
    output_batch = Assemble()(
        event_batch,
        demographics_batch,
        TIMEFRAME["start"],
    )
    del event_batch, demographics_batch
    draw_replay(frame_batch, detection_batch, track_batch, replay_path, fps, frame_size)
    del frame_batch, detection_batch, track_batch
    write_output_batch(output_batch, output_batch_path)
    print(json.dumps(output_batch, sort_keys=True))


if __name__ == "__main__":
    main()
