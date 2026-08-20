"""Local inspection pipeline for a manually chosen device timeframe."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import median
import cv2

from assemble import Assemble
from demographics import Demographic
from detect import Detect
from events import Event
from initialise import initialise
from load.load import load
from track import Track


def _load_service_account_info() -> dict:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must be configured")
    with Path(credentials_path).open(encoding="utf-8") as file:
        return json.load(file)


def _run_directory(device_id: int, timeframe: dict) -> Path:
    def window_part(value: str) -> str:
        return value.replace("-", "").replace(":", "").replace(".", "")

    root = Path("output") / "visualize"
    stem = (
        f"device-{device_id}_"
        f"{window_part(timeframe['start'])}_{window_part(timeframe['end'])}"
    )
    candidate = root / stem
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stem}-{suffix}"
        suffix += 1
    (candidate / "tracks").mkdir(parents=True)
    (candidate / "events").mkdir()
    (candidate / "data").mkdir()
    return candidate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _get_replay_fps(frame_batch: dict) -> float:
    timestamps = [float(frame["timestamp"]) for frame in frame_batch["frames"]]
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if later - earlier > 0.0
    ]
    if not intervals:
        raise ValueError(
            "Cannot derive replay FPS from fewer than two distinct frame timestamps"
        )
    return 1.0 / median(intervals)


def _write_crop(frames_by_id: dict, crop: dict, path: Path) -> None:
    image = frames_by_id[crop["frame_id"]]
    height, width = image.shape[:2]
    bbox = crop["bbox"]
    left = max(0, min(width, int(float(bbox["x1"]))))
    top = max(0, min(height, int(float(bbox["y1"]))))
    right = max(0, min(width, int(float(bbox["x2"]))))
    bottom = max(0, min(height, int(float(bbox["y2"]))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop for {path.name}")
    if not cv2.imwrite(
        str(path), cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"Unable to write thumbnail: {path}")


def _write_thumbnails(
    track_batch: dict,
    event_batch: dict,
    frames_by_id: dict,
    run_directory: Path,
) -> None:
    for track in track_batch["tracks"]:
        _write_crop(
            frames_by_id,
            track["best_crop"],
            run_directory / "tracks" / f"track-{track['track_id']}.jpg",
        )
    for event_index, event in enumerate(event_batch["events"], start=1):
        _write_crop(
            frames_by_id,
            event["best_crop"],
            run_directory
            / "events"
            / f"event-{event_index:03d}-track-{event['track_id']}.jpg",
        )


def _write_replay(
    frame_batch: dict,
    detection_batch: dict,
    track_batch: dict,
    event_track_ids: set,
    analysis_config: dict,
    path: Path,
) -> None:
    track_ids_by_timestamp_centre = {
        (
            float(point["timestamp"]),
            round(float(point["centre"]["x"]), 6),
            round(float(point["centre"]["y"]), 6),
        ): track["track_id"]
        for track in track_batch["tracks"]
        for point in track["path"]
    }
    first_image = frame_batch["frames"][0]["image"]
    frame_size = (first_image.shape[1], first_image.shape[0])
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _get_replay_fps(frame_batch),
        frame_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Cannot open replay output: {path}")

    try:
        for frame, frame_detections in zip(
            frame_batch["frames"], detection_batch["detections"], strict=True
        ):
            output = cv2.cvtColor(frame["image"], cv2.COLOR_RGB2BGR)
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
                    )
                )
                if track_id is None:
                    continue
                colour = (0, 255, 0) if track_id in event_track_ids else (0, 0, 255)
                cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    output,
                    f"Track {track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
            cv2.line(
                output,
                (
                    int(analysis_config["point_a"]["x"]),
                    int(analysis_config["point_a"]["y"]),
                ),
                (
                    int(analysis_config["point_b"]["x"]),
                    int(analysis_config["point_b"]["y"]),
                ),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(output)
    finally:
        writer.release()


def _summary(
    device_id: int,
    timeframe: dict,
    frame_batch: dict,
    detection_batch: dict,
    track_batch: dict,
    event_batch: dict,
    track_seconds: float,
    run_directory: Path,
) -> dict:
    track_lengths = [len(track["path"]) for track in track_batch["tracks"]]
    detection_count = sum(
        len(frame_detections["detections"])
        for frame_detections in detection_batch["detections"]
    )
    return {
        "device_id": device_id,
        "timeframe": timeframe,
        "frames_loaded": len(frame_batch["frames"]),
        "track_runtime_seconds": track_seconds,
        "tracks_returned": len(track_lengths),
        "average_observations_per_track": (
            sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
        ),
        "median_observations_per_track": median(track_lengths) if track_lengths else 0.0,
        "short_tracks": sum(length <= 3 for length in track_lengths),
        "shortest_track": min(track_lengths) if track_lengths else 0,
        "longest_track": max(track_lengths) if track_lengths else 0,
        "unassigned_detections": max(detection_count - sum(track_lengths), 0),
        "events_produced": len(event_batch["events"]),
        "event_track_ids": sorted(
            {event["track_id"] for event in event_batch["events"]}
        ),
        "output_folder": str(run_directory),
        "replay_path": str(run_directory / "replay.mp4"),
    }


def Visualize(device_id: int, timeframe: dict) -> None:
    """Run the real pipeline for a supplied timeframe and save local artifacts."""
    context = initialise(device_id)
    context["timeframe"] = timeframe
    run_directory = _run_directory(context["device_id"], timeframe)

    frame_batch = load(
        context["gcs_source_uri"],
        context["timeframe"],
        _load_service_account_info(),
    )
    if not frame_batch["frames"]:
        raise ValueError("No timestamp-named JPG frames found in the supplied timeframe")

    detection_batch = Detect()(frame_batch)
    track_started = time.perf_counter()
    track_batch = Track()(detection_batch)
    track_seconds = time.perf_counter() - track_started
    event_batch = Event()(track_batch, context["analysis_config"])
    frames_by_id = {
        frame["frame_id"]: frame["image"] for frame in frame_batch["frames"]
    }
    event_track_ids = {event["track_id"] for event in event_batch["events"]}

    _write_thumbnails(track_batch, event_batch, frames_by_id, run_directory)
    _write_replay(
        frame_batch,
        detection_batch,
        track_batch,
        event_track_ids,
        context["analysis_config"],
        run_directory / "replay.mp4",
    )
    _write_json(run_directory / "data" / "tracks.json", track_batch)
    _write_json(run_directory / "data" / "events.json", event_batch)
    summary = _summary(
        context["device_id"],
        timeframe,
        frame_batch,
        detection_batch,
        track_batch,
        event_batch,
        track_seconds,
        run_directory,
    )
    _write_json(run_directory / "data" / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))

    demographics_batch = Demographic()(event_batch, frame_batch)
    output_batch = Assemble()(
        event_batch,
        demographics_batch,
        context["timeframe"]["start"],
        context["device_id"],
    )
    _write_json(run_directory / "data" / "demographics.json", demographics_batch)
    _write_json(run_directory / "data" / "output.json", output_batch)
