"""Minimal real integration runner for Detect -> Track -> Event -> Demographic."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demographics import Demographic
from detect import Detect
from events import Event
from track import Track

DEFAULT_VIDEO_PATH = "videoplayback.mp4"
DEFAULT_REPLAY_PATH = "output/tracking_replay.mp4"
DEFAULT_EVENTS_OUTPUT = "output/events_with_demographics.json"
LINE_CONFIG = {
    "point_a": {"x": 100.0, "y": 300.0},
    "point_b": {"x": 700.0, "y": 300.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Detect -> Track -> Event -> Demographic pipeline."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--output", default=DEFAULT_REPLAY_PATH, help="Annotated replay path")
    parser.add_argument(
        "--events-output",
        default=DEFAULT_EVENTS_OUTPUT,
        help="Events-with-demographics JSON path",
    )
    return parser.parse_args()


def build_frame_batch_from_video(video_path: Path) -> tuple[dict[str, list[dict[str, Any]]], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or fps > 240.0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[dict[str, Any]] = []
        frame_index = 0
        while True:
            ok, bgr_image = cap.read()
            if not ok:
                break
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_image = np.ascontiguousarray(rgb_image)
            frames.append(
                {
                    "frame_id": f"frame-{frame_index}",
                    "timestamp": float(frame_index) / fps,
                    "image": rgb_image,
                }
            )
            frame_index += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return {"frames": frames}, fps, (width, height)


def create_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
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


def validate_detection_batch(detection_batch: dict[str, Any], frame_batch: dict[str, Any]) -> None:
    frame_detections = detection_batch.get("detections")
    frames = frame_batch["frames"]
    if not isinstance(frame_detections, list):
        raise ValueError("DetectionBatch.detections must be a list")
    if len(frame_detections) != len(frames):
        raise ValueError("DetectionBatch.detections length must match FrameBatch.frames")
    for index, (frame, detections) in enumerate(zip(frames, frame_detections, strict=True)):
        if detections["frame_id"] != frame["frame_id"]:
            raise ValueError(f"DetectionBatch.detections[{index}].frame_id does not match FrameBatch")
        if float(detections["timestamp"]) != float(frame["timestamp"]):
            raise ValueError(f"DetectionBatch.detections[{index}].timestamp does not match FrameBatch")


def draw_replay(
    frame_batch: dict[str, Any],
    detection_batch: dict[str, Any],
    tracking_state: dict[str, Any],
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
        for track in tracking_state["tracks"]
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
                    (timestamp, round(float(centre["x"]), 6), round(float(centre["y"]), 6)), "?"
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


def validate_event_best_crops(event_batch: dict[str, Any], frame_batch: dict[str, Any]) -> None:
    frame_ids = {frame["frame_id"] for frame in frame_batch["frames"]}
    for event in event_batch["events"]:
        frame_id = event["best_crop"]["frame_id"]
        if frame_id not in frame_ids:
            raise ValueError(f"Event track_id={event['track_id']} references missing frame_id={frame_id}")


def join_events_and_demographics(event_batch: dict[str, Any], demographics_batch: dict[str, Any]) -> dict[str, Any]:
    demographics_by_track = {result["track_id"]: result for result in demographics_batch["results"]}
    return {
        "events": [
            {
                "track_id": event["track_id"],
                "timestamp": float(event["timestamp"]),
                "event_type": int(event["event_type"]),
                "age": int(demographics_by_track[event["track_id"]]["age"]),
                "sex": int(demographics_by_track[event["track_id"]]["sex"]),
            }
            for event in event_batch["events"]
        ]
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    replay_path = Path(args.output)
    events_output_path = Path(args.events_output)
    started_at = time.perf_counter()

    frame_batch, fps, frame_size = build_frame_batch_from_video(video_path)
    frame_batch_memory = sum(frame["image"].nbytes for frame in frame_batch["frames"])

    detection_batch = Detect()(frame_batch)
    validate_detection_batch(detection_batch, frame_batch)

    tracking_state = {"tracks": []}
    tracking_state = Track(tracking_state, detection_batch)

    event_batch = Event(tracking_state, LINE_CONFIG)
    validate_event_best_crops(event_batch, frame_batch)

    demographics_batch = Demographic()(event_batch, frame_batch)
    final_output = join_events_and_demographics(event_batch, demographics_batch)
    draw_replay(frame_batch, detection_batch, tracking_state, replay_path, fps, frame_size)
    write_json(final_output, events_output_path)

    runtime_seconds = time.perf_counter() - started_at
    peak_memory_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Pipeline complete")
    print(f"Frames processed: {len(frame_batch['frames'])}")
    print(f"DetectionBatch Detections count: {len(detection_batch['detections'])}")
    print(f"Tracks produced: {len(tracking_state['tracks'])}")
    print(f"Events produced: {len(event_batch['events'])}")
    print(f"Demographics produced: {len(demographics_batch['results'])}")
    print(f"Replay path: {replay_path}")
    print(f"JSON path: {events_output_path}")
    print(f"Pipeline runtime seconds: {runtime_seconds:.3f}")
    print(f"FrameBatch memory bytes: {frame_batch_memory}")
    print(f"Peak memory KiB: {peak_memory_kib}")
    print("Source video open count: 1")
    print("Second decode: no")


if __name__ == "__main__":
    main()
