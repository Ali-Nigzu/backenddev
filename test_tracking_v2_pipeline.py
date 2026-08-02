"""Replay Detect -> Track over a video, then run Event once."""

import argparse
import json
from pathlib import Path


DEFAULT_VIDEO_PATH = "videoplayback.mp4"
DEFAULT_OUTPUT_NAME = "tracking_replay.mp4"
DEFAULT_EVENTS_OUTPUT = "output/events_with_demographics.json"
LINE_CONFIG = {
    "point_a": {"x": 100.0, "y": 300.0},
    "point_b": {"x": 700.0, "y": 300.0},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Detect -> Track replay over a video."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_VIDEO_PATH,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Annotated replay output path",
    )
    parser.add_argument(
        "--events-output",
        default=DEFAULT_EVENTS_OUTPUT,
        help="Enriched events JSON output path",
    )
    return parser.parse_args()


def track_color(track_id: str) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track ID."""

    value = 0
    for character in str(track_id):
        value = (value * 131 + ord(character)) % 0xFFFFFF

    # Keep colors bright enough to stand out against video frames.
    return (
        80 + (value & 0x7F),
        80 + ((value >> 7) & 0x7F),
        80 + ((value >> 14) & 0x7F),
    )


def point_key(point: dict) -> tuple[float, float]:
    centre = point["centre"] if "centre" in point else point
    return (round(float(centre["x"]), 6), round(float(centre["y"]), 6))


def active_track_ids(tracking_state, detection_batch) -> set[str]:
    """Return active track IDs before this frame update."""

    from track.tracker import _classify_track

    timestamp = float(detection_batch["timestamp"])
    return {
        str(track["track_id"])
        for track in tracking_state["tracks"]
        if _classify_track(track, timestamp) == "active"
    }


def current_frame_assignments(
    tracking_state, detection_batch, _previous_active_track_ids: set[str]
) -> list[tuple[dict, dict]]:
    """Pair tracks updated on this frame with their detections."""

    timestamp = float(detection_batch["timestamp"])
    detections_by_centre = {}
    for detection in detection_batch["detections"]:
        detections_by_centre.setdefault(point_key(detection["centre"]), []).append(
            detection
        )

    assignments = []
    for track in tracking_state["tracks"]:
        latest_point = track["path"][-1]
        if float(latest_point["timestamp"]) != timestamp:
            continue

        matching_detections = detections_by_centre.get(point_key(latest_point), [])
        if not matching_detections:
            continue

        assignments.append((track, matching_detections.pop(0)))

    return assignments


LINE_COLOR_BGR = (255, 255, 255)
POINT_A_COLOR_BGR = (0, 0, 255)
POINT_B_COLOR_BGR = (255, 0, 0)
LABEL_COLOR_BGR = (255, 255, 255)
LINE_THICKNESS = 2
POINT_RADIUS = 6


def line_point_xy(point: dict, name: str) -> tuple[float, float]:
    if not isinstance(point, dict):
        raise ValueError(f"{name} must be an object")
    for field in ("x", "y"):
        if field not in point:
            raise ValueError(f"Missing required {name} field: {field}")
    x = float(point["x"])
    y = float(point["y"])
    return x, y


def draw_line_point(frame, point: tuple[float, float], color: tuple[int, int, int]) -> None:
    import cv2

    cv2.circle(
        frame,
        (int(round(point[0])), int(round(point[1]))),
        POINT_RADIUS,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def clipped_line_points(
    point_a: tuple[float, float], point_b: tuple[float, float], width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ax, ay = point_a
    bx, by = point_b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        raise ValueError("point_a and point_b must define a non-zero line")

    line_points: list[tuple[float, float]] = []
    max_x = float(width - 1)
    max_y = float(height - 1)

    if dx != 0:
        for x in (0.0, max_x):
            scale = (x - ax) / dx
            y = ay + scale * dy
            if 0.0 <= y <= max_y:
                line_points.append((x, y))

    if dy != 0:
        for y in (0.0, max_y):
            scale = (y - ay) / dy
            x = ax + scale * dx
            if 0.0 <= x <= max_x:
                line_points.append((x, y))

    unique_points: list[tuple[float, float]] = []
    for line_point in line_points:
        rounded = (round(line_point[0], 6), round(line_point[1], 6))
        if all(
            rounded != (round(point[0], 6), round(point[1], 6))
            for point in unique_points
        ):
            unique_points.append(line_point)

    if len(unique_points) < 2:
        return None

    return (
        (int(round(unique_points[0][0])), int(round(unique_points[0][1]))),
        (int(round(unique_points[1][0])), int(round(unique_points[1][1]))),
    )


def draw_line_overlay(frame, line_config: dict) -> None:
    import cv2

    point_a = line_point_xy(line_config["point_a"], "point_a")
    point_b = line_point_xy(line_config["point_b"], "point_b")
    height, width = frame.shape[:2]
    clipped_points = clipped_line_points(point_a, point_b, width, height)
    if clipped_points is not None:
        cv2.line(
            frame,
            clipped_points[0],
            clipped_points[1],
            LINE_COLOR_BGR,
            thickness=LINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )
    draw_line_point(frame, point_a, POINT_A_COLOR_BGR)
    draw_line_point(frame, point_b, POINT_B_COLOR_BGR)
    cv2.putText(
        frame,
        "A",
        (int(round(point_a[0])) + 8, int(round(point_a[1])) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        LABEL_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "B",
        (int(round(point_b[0])) + 8, int(round(point_b[1])) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        LABEL_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )


def draw_tracking_state(
    frame,
    tracking_state,
    detection_batch,
    active_ids: set[str],
    previous_track_ids: set[str] | None = None,
) -> dict:
    import cv2

    previous_track_ids = previous_track_ids or set()
    timestamp = float(detection_batch["timestamp"])
    assignments = current_frame_assignments(tracking_state, detection_batch, active_ids)
    matched_track_ids = {str(track["track_id"]) for track, _detection in assignments}
    birth_track_ids = {
        str(track["track_id"])
        for track in tracking_state["tracks"]
        if str(track["track_id"]) not in previous_track_ids
        and float(track["path"][-1]["timestamp"]) == timestamp
    }
    unmatched_active_ids = set(active_ids) - matched_track_ids

    height, width = frame.shape[:2]
    for track, detection in assignments:
        bbox = detection["bbox"]
        x1 = max(0, min(width - 1, int(round(float(bbox["x1"])))))
        y1 = max(0, min(height - 1, int(round(float(bbox["y1"])))))
        x2 = max(0, min(width - 1, int(round(float(bbox["x2"])))))
        y2 = max(0, min(height - 1, int(round(float(bbox["y2"])))))
        track_id = str(track["track_id"])
        color = track_color(track_id)
        label = f"Track {track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        label_y1 = max(0, y1 - text_height - baseline - 4)
        label_y2 = label_y1 + text_height + baseline + 4
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 6, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    debug = {
        "active": len(active_ids),
        "detections": len(detection_batch["detections"]),
        "matched": len(matched_track_ids),
        "births": len(birth_track_ids),
        "unmatched_active": len(unmatched_active_ids),
    }
    overlay = f"tracks={len(tracking_state['tracks'])} det={debug['detections']} matched={debug['matched']}"
    cv2.rectangle(
        frame, (8, 8), (min(width - 1, 8 + 12 * len(overlay)), 36), (0, 0, 0), -1
    )
    cv2.putText(
        frame,
        overlay,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return debug


def update_track_summary(track_summary, tracking_state) -> None:
    for track in tracking_state["tracks"]:
        track_id = track["track_id"]
        first_seen = float(track["path"][0]["timestamp"])
        last_seen = float(track["path"][-1]["timestamp"])
        summary = track_summary.setdefault(
            track_id, {"first_seen": first_seen, "last_seen": last_seen}
        )
        summary["first_seen"] = min(summary["first_seen"], first_seen)
        summary["last_seen"] = max(summary["last_seen"], last_seen)


def event_label(event_type: int) -> str:
    return "ENTRY" if event_type == 1 else "EXIT"


def format_elapsed_time(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def print_event_summary(event_batch: dict, fps: float) -> None:
    events = event_batch["events"]
    entry_count = sum(1 for event in events if event["event_type"] == 1)
    exit_count = sum(1 for event in events if event["event_type"] == 0)
    tracks_with_events = {event["track_id"] for event in events}

    print("\nEVENTS")
    print("======")
    if not events:
        print("- none")
    for event in events:
        bbox = event["best_crop"]["bbox"]
        event_seconds = float(event["timestamp"])
        print(
            f"track_id={event['track_id']} "
            f"event_type={event['event_type']} "
            f"label={event_label(event['event_type'])} "
            f"timestamp_seconds={float(event['timestamp']):.3f} "
            f"elapsed={event_seconds:.3f}s "
            f"time={format_elapsed_time(event_seconds)} "
            f"best_crop_frame={event['best_crop']['frame_id']} "
            f"bbox=({float(bbox['x1']):.3f}, {float(bbox['y1']):.3f}, "
            f"{float(bbox['x2']):.3f}, {float(bbox['y2']):.3f})"
        )

    print(f"Total events: {len(events)}")
    print(f"Total entries: {entry_count}")
    print(f"Total exits: {exit_count}")
    print(f"Unique track IDs with events: {len(tracks_with_events)}")


def print_track_summary(track_summary, frame_count: int) -> None:
    print("\nSUMMARY")
    print("=======")
    print(f"frames: {frame_count}")
    print(f"tracks: {len(track_summary)}")

    if not track_summary:
        print("- none")
        return

    for track_id in sorted(
        track_summary, key=lambda value: int(value) if value.isdecimal() else value
    ):
        summary = track_summary[track_id]
        first_seen = summary["first_seen"]
        last_seen = summary["last_seen"]
        duration = last_seen - first_seen
        print(
            f"Track ID: {track_id} | "
            f"first_seen_frame={first_seen:.0f} | "
            f"last_seen_frame={last_seen:.0f} | "
            f"duration_frames={duration:.0f}"
        )


def make_frame_record(frame_index: int, fps: float, bgr_frame) -> dict:
    import cv2
    import numpy as np

    timestamp = float(frame_index) / float(fps)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    if rgb_frame.dtype != np.uint8:
        raise ValueError(f"Frame frame-{frame_index} image must have dtype uint8")
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError(f"Frame frame-{frame_index} image must have shape H x W x 3")
    if rgb_frame.shape[0] <= 0 or rgb_frame.shape[1] <= 0:
        raise ValueError(f"Frame frame-{frame_index} image must have positive dimensions")
    if not rgb_frame.flags.c_contiguous:
        rgb_frame = np.ascontiguousarray(rgb_frame)
    return {
        "frame_id": f"frame-{frame_index}",
        "timestamp": float(timestamp),
        "image": rgb_frame,
    }


def validate_detection_batch_matches_frame_batch(detection_batch: dict, frame_batch: dict) -> None:
    if not isinstance(detection_batch, dict):
        raise ValueError("DetectionBatch must be an object")
    if "detections" not in detection_batch or not isinstance(detection_batch["detections"], list):
        raise ValueError("DetectionBatch.detections must be a list")
    frames = frame_batch["frames"]
    detections = detection_batch["detections"]
    if len(detections) != len(frames):
        raise ValueError("DetectionBatch.detections length does not match FrameBatch.frames")
    for index, (frame, frame_detections) in enumerate(zip(frames, detections, strict=True)):
        if frame_detections["frame_id"] != frame["frame_id"]:
            raise ValueError(
                f"Detection result {index} frame_id does not match FrameBatch frame_id: "
                f"{frame_detections['frame_id']} != {frame['frame_id']}"
            )
        if float(frame_detections["timestamp"]) != float(frame["timestamp"]):
            raise ValueError(
                f"Detection result {index} timestamp does not match FrameBatch timestamp: "
                f"{frame_detections['timestamp']} != {frame['timestamp']}"
            )


def validate_event_best_crop_frames(event_batch: dict, frame_batch: dict) -> None:
    frame_ids = {frame["frame_id"] for frame in frame_batch["frames"]}
    for event in event_batch["events"]:
        frame_id = event["best_crop"]["frame_id"]
        if frame_id not in frame_ids:
            raise ValueError(
                f"Event track_id={event['track_id']} references missing frame_id {frame_id} "
                f"bbox={event['best_crop']['bbox']}"
            )


def demographics_for_events(event_batch: dict, frame_batch: dict) -> dict:
    if not event_batch["events"]:
        return {"results": []}
    from demographics import Demographic

    validate_event_best_crop_frames(event_batch, frame_batch)
    demographic = Demographic()
    return demographic(event_batch, frame_batch)


def build_enriched_events(event_batch: dict, demographics_batch: dict) -> dict:
    event_track_ids = {event["track_id"] for event in event_batch["events"]}
    demographics_by_track = {}
    for result in demographics_batch["results"]:
        track_id = result["track_id"]
        if track_id in demographics_by_track:
            raise ValueError(f"Duplicate demographic result for track_id={track_id}")
        demographics_by_track[track_id] = result
    extra_results = set(demographics_by_track) - event_track_ids
    if extra_results:
        raise ValueError(f"Demographic results without matching events: {sorted(extra_results)}")
    enriched_events = []
    for event in event_batch["events"]:
        track_id = event["track_id"]
        if track_id not in demographics_by_track:
            raise ValueError(f"Missing demographic result for track_id={track_id}")
        result = demographics_by_track[track_id]
        enriched_events.append(
            {
                "track_id": track_id,
                "timestamp": float(event["timestamp"]),
                "event_type": int(event["event_type"]),
                "age": int(result["age"]),
                "sex": int(result["sex"]),
            }
        )
    enriched_events.sort(key=lambda event: (float(event["timestamp"]), str(event["track_id"]), int(event["event_type"])))
    return {"events": enriched_events}


def write_enriched_events(enriched_event_batch: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched_event_batch, indent=2, sort_keys=True) + "\n")


def format_sex(sex: int) -> str:
    labels = {0: "female", 1: "male"}
    if sex not in labels:
        raise ValueError(f"Invalid sex value: {sex}")
    return f"{sex}({labels[sex]})"


def print_enriched_event_summary(enriched_event_batch: dict) -> None:
    print("\nEVENTS WITH DEMOGRAPHICS")
    print("========================")
    if not enriched_event_batch["events"]:
        print("- none")
        return
    for event in enriched_event_batch["events"]:
        print(
            f"track_id={event['track_id']} "
            f"event_type={event['event_type']} "
            f"label={event_label(event['event_type'])} "
            f"timestamp_seconds={float(event['timestamp']):.3f} "
            f"age={event['age']} "
            f"sex={format_sex(event['sex'])}"
        )

def main():
    args = parse_args()
    video_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).with_name(DEFAULT_OUTPUT_NAME)
    )

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_index = 0

    print("DETECT -> TRACK REPLAY")
    print("======================")
    print(f"input: {video_path}")
    print(f"line_config: {LINE_CONFIG}")

    try:
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            frames.append(make_frame_record(frame_index, fps, bgr_frame))
            frame_index += 1
            print(f"\rdecoded frames: {frame_index}", end="", flush=True)
    finally:
        cap.release()

    frame_batch = {"frames": frames}
    frame_batch_memory = sum(frame["image"].nbytes for frame in frames)

    from detect import Detect

    detect = Detect()
    detection_batch = detect(frame_batch)
    validate_detection_batch_matches_frame_batch(detection_batch, frame_batch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Cannot open replay output: {output_path}")

    from track import Track

    tracking_state = {"tracks": []}
    track_summary = {}

    try:
        for index, (frame, frame_detections) in enumerate(
            zip(frame_batch["frames"], detection_batch["detections"], strict=True)
        ):
            bgr_frame = cv2.cvtColor(frame["image"], cv2.COLOR_RGB2BGR)
            active_ids = active_track_ids(tracking_state, frame_detections)
            previous_track_ids = {
                str(track["track_id"]) for track in tracking_state["tracks"]
            }
            tracking_state = Track(tracking_state, frame_detections)
            update_track_summary(track_summary, tracking_state)
            debug = draw_tracking_state(
                bgr_frame,
                tracking_state,
                frame_detections,
                active_ids,
                previous_track_ids,
            )
            draw_line_overlay(bgr_frame, LINE_CONFIG)
            if debug["births"] or debug["unmatched_active"]:
                print(
                    f"\nframe {index}: active={debug['active']} det={debug['detections']} "
                    f"matched={debug['matched']} births={debug['births']} "
                    f"unmatched_active={debug['unmatched_active']}"
                )
            writer.write(bgr_frame)
            print(f"\rprocessed frames: {index + 1}", end="", flush=True)
    finally:
        writer.release()

    from events import Event

    event_batch = Event(tracking_state, LINE_CONFIG)
    validate_event_best_crop_frames(event_batch, frame_batch)
    demographics_batch = demographics_for_events(event_batch, frame_batch)
    enriched_event_batch = build_enriched_events(event_batch, demographics_batch)
    write_enriched_events(enriched_event_batch, Path(args.events_output))
    print_track_summary(track_summary, frame_index)
    print_event_summary(event_batch, fps)
    print_enriched_event_summary(enriched_event_batch)
    print("\nReplay complete")
    print(f"\nFrames processed: {frame_index}")
    print(f"DetectionBatch per-frame objects: {len(detection_batch['detections'])}")
    print(f"Tracks created: {len(track_summary)}")
    print(f"Events produced: {len(event_batch['events'])}")
    print(f"Demographic results: {len(demographics_batch['results'])}")
    print(f"FrameBatch image memory bytes: {frame_batch_memory}")
    print(f"Output video path: {output_path}")
    print(f"Events JSON path: {Path(args.events_output)}")


if __name__ == "__main__":
    main()
