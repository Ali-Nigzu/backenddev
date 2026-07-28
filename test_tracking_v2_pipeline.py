"""Replay Detect -> Track over a video, then run Event once."""

import argparse
from pathlib import Path


DEFAULT_VIDEO_PATH = "videoplayback.mp4"
DEFAULT_OUTPUT_NAME = "tracking_replay.mp4"
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

        candidates = detections_by_centre.get(point_key(latest_point), [])
        if not candidates:
            continue

        assignments.append((track, candidates.pop(0)))

    return assignments


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


def _format_video_time(total_seconds: float) -> str:
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
        event_seconds = float(event["timestamp"]) / float(fps)
        print(
            f"Track {event['track_id']} | "
            f"{event_seconds:.3f}s | "
            f"{_format_video_time(event_seconds)} | "
            f"event_type={event['event_type']} | "
            f"{event_label(event['event_type'])} | "
            f"best_crop_frame={event['best_crop']['frame_id']} | "
            f"bbox=({float(bbox['x1']):.3f}, {float(bbox['y1']):.3f}, "
            f"{float(bbox['x2']):.3f}, {float(bbox['y2']):.3f})"
        )

    print(f"Total events: {len(events)}")
    print(f"Entries: {entry_count}")
    print(f"Exits: {exit_count}")
    print(f"Tracks with events: {len(tracks_with_events)}")


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


def main():
    args = parse_args()
    video_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).with_name(DEFAULT_OUTPUT_NAME)
    )

    import cv2

    from detect import Detect
    from events import Event
    from events.line_overlay import draw_line_overlay
    from track import Track

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open replay output: {output_path}")

    detect = Detect()
    tracking_state = {"tracks": []}
    track_summary = {}
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

            timestamp = float(frame_index)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            if not rgb_frame.flags.c_contiguous:
                rgb_frame = rgb_frame.copy()

            frame = {
                "frame_id": f"frame-{frame_index}",
                "timestamp": float(timestamp),
                "image": rgb_frame,
            }

            detection_batch = detect(frame)
            active_ids = active_track_ids(tracking_state, detection_batch)
            previous_track_ids = {
                str(track["track_id"]) for track in tracking_state["tracks"]
            }
            tracking_state = Track(tracking_state, detection_batch)
            update_track_summary(track_summary, tracking_state)
            debug = draw_tracking_state(
                bgr_frame,
                tracking_state,
                detection_batch,
                active_ids,
                previous_track_ids,
            )
            draw_line_overlay(bgr_frame, LINE_CONFIG)
            if debug["births"] or debug["unmatched_active"]:
                print(
                    f"\nframe {frame_index}: active={debug['active']} det={debug['detections']} "
                    f"matched={debug['matched']} births={debug['births']} "
                    f"unmatched_active={debug['unmatched_active']}"
                )
            writer.write(bgr_frame)

            frame_index += 1
            print(f"\rprocessed frames: {frame_index}", end="", flush=True)
    finally:
        cap.release()
        writer.release()

    event_batch = Event(tracking_state, LINE_CONFIG)
    print_track_summary(track_summary, frame_index)
    print_event_summary(event_batch, fps)
    print("\nReplay complete")
    print(f"\nFrames processed: {frame_index}")
    print(f"Tracks created: {len(track_summary)}")
    print(f"Annotated replay: {output_path}")


if __name__ == "__main__":
    main()
