import csv
import shutil
from pathlib import Path
from statistics import median
import cv2

from ..assemble import Assemble
from ..demographics import Demographic
from ..detect import Detect
from ..events import Event
from ..initialise import initialise
from ..load import load
from ..track import Track

def _run_directory(device_id: int, timeframe: dict) -> Path:
    def window_part(value: str) -> str:
        return value.replace("-", "").replace(":", "").replace(".", "")

    root = Path("output") / "visualise"
    stem = (
        f"device-{device_id}_"
        f"{window_part(timeframe['start'])}_{window_part(timeframe['end'])}"
    )
    candidate = root / stem
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)
    return candidate

def _write_events_csv(output_batch: dict, path: Path) -> None:
    fieldnames = (
        "device_id",
        "event_id",
        "event",
        "timestamp",
        "sex",
        "age_bucket",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_batch["rows"])

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

def Visualise(device_id: int, timeframe: dict) -> None:

    context = initialise(device_id)
    context["timeframe"] = timeframe
    run_directory = _run_directory(context["device_id"], timeframe)

    frame_batch = load(
        context["gcs_source_uri"],
        context["timeframe"],
    )
    if not frame_batch["frames"]:
        raise ValueError("No timestamp-named JPG frames found in the supplied timeframe")

    detection_batch = Detect()(frame_batch)
    track_batch = Track()(detection_batch)
    event_batch = Event()(track_batch, context["analysis_config"])
    event_track_ids = {event["track_id"] for event in event_batch["events"]}

    _write_replay(
        frame_batch,
        detection_batch,
        track_batch,
        event_track_ids,
        context["analysis_config"],
        run_directory / "replay.mp4",
    )

    demographics_batch = Demographic()(event_batch, frame_batch)
    output_batch = Assemble()(
        event_batch,
        demographics_batch,
        context["timeframe"]["start"],
        context["device_id"],
    )
    _write_events_csv(output_batch, run_directory / "events.csv")
