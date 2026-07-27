"""Replay Detect -> Embed -> Observe -> Track over an input video.

This script intentionally stops at Track V2. It does not run Event,
Demographic, Assemble, or Output stages.
"""

import argparse
from pathlib import Path


DEFAULT_VIDEO_PATH = "videoplayback.mp4"
DEFAULT_OUTPUT_NAME = "tracking_replay.mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the V2 Detect -> Embed -> Observe -> Track replay over a video."
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
    center = point["center"] if "center" in point else point
    return (round(float(center["x"]), 6), round(float(center["y"]), 6))


def active_track_ids(tracking_state, observation_batch, config) -> set[str]:
    """Return active track IDs before this frame update."""

    from track.lifecycle import ACTIVE, classify_track

    timestamp = float(observation_batch["timestamp"])
    return {
        str(track["track_id"])
        for track in tracking_state["tracks"]
        if classify_track(track, timestamp, config).state == ACTIVE
    }


def current_frame_assignments(
    tracking_state, observation_batch, active_track_ids: set[str]
) -> list[tuple[dict, dict]]:
    """Pair confirmed active tracks updated on this frame with their observations."""

    timestamp = float(observation_batch["timestamp"])
    observations_by_center = {}
    for observation in observation_batch["observations"]:
        observations_by_center.setdefault(point_key(observation["center"]), []).append(observation)

    assignments = []
    for track in tracking_state["tracks"]:
        if str(track["track_id"]) not in active_track_ids:
            continue

        latest_point = track["path"][-1]
        if float(latest_point["timestamp"]) != timestamp:
            continue

        candidates = observations_by_center.get(point_key(latest_point), [])
        if not candidates:
            continue

        assignments.append((track, candidates.pop(0)))

    return assignments


def draw_tracking_state(
    frame,
    tracking_state,
    observation_batch,
    active_ids: set[str],
    previous_track_ids: set[str] | None = None,
) -> dict:
    import cv2

    previous_track_ids = previous_track_ids or set()
    timestamp = float(observation_batch["timestamp"])
    assignments = current_frame_assignments(tracking_state, observation_batch, active_ids)
    matched_track_ids = {str(track["track_id"]) for track, _observation in assignments}
    birth_track_ids = {
        str(track["track_id"])
        for track in tracking_state["tracks"]
        if str(track["track_id"]) not in previous_track_ids
        and float(track["path"][-1]["timestamp"]) == timestamp
    }
    unmatched_active_ids = set(active_ids) - matched_track_ids

    height, width = frame.shape[:2]
    for track, observation in assignments:
        bbox = observation["bbox"]
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
        "observations": len(observation_batch["observations"]),
        "matched": len(matched_track_ids),
        "births": len(birth_track_ids),
        "unmatched_active": len(unmatched_active_ids),
    }
    overlay = f"tracks={len(tracking_state['tracks'])} obs={debug['observations']} matched={debug['matched']}"
    cv2.rectangle(frame, (8, 8), (min(width - 1, 8 + 12 * len(overlay)), 36), (0, 0, 0), -1)
    cv2.putText(frame, overlay, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
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
            f"first_seen={first_seen:.6f}s | "
            f"last_seen={last_seen:.6f}s | "
            f"duration={duration:.6f}s"
        )


def main():
    args = parse_args()
    video_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(__file__).with_name(DEFAULT_OUTPUT_NAME)

    import cv2

    from detection import Detect
    from embed import Embed
    from observe import Observe
    from track import Track, TrackV2Config

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
    embed = Embed()
    observe = Observe()
    config = TrackV2Config()
    tracking_state = {"tracks": []}
    track_summary = {}
    frame_index = 0

    print("V2 TRACK REPLAY")
    print("===============")
    print(f"input: {video_path}")

    try:
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / fps
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            if not rgb_frame.flags.c_contiguous:
                rgb_frame = rgb_frame.copy()

            frame = {
                "frame_id": f"frame-{frame_index}",
                "timestamp": float(timestamp),
                "image": rgb_frame,
            }

            detection_batch = detect(frame)
            embedding_batch = embed(detection_batch)
            observation_batch = observe(detection_batch, embedding_batch)
            active_ids = active_track_ids(tracking_state, observation_batch, config)
            previous_track_ids = {str(track["track_id"]) for track in tracking_state["tracks"]}
            tracking_state = Track(tracking_state, observation_batch, config)
            update_track_summary(track_summary, tracking_state)
            debug = draw_tracking_state(
                bgr_frame,
                tracking_state,
                observation_batch,
                active_ids,
                previous_track_ids,
            )
            if debug["births"] or debug["unmatched_active"]:
                print(
                    f"\nframe {frame_index}: active={debug['active']} obs={debug['observations']} "
                    f"matched={debug['matched']} births={debug['births']} "
                    f"unmatched_active={debug['unmatched_active']}"
                )
            writer.write(bgr_frame)

            frame_index += 1
            print(f"\rprocessed frames: {frame_index}", end="", flush=True)
    finally:
        cap.release()
        writer.release()

    print_track_summary(track_summary, frame_index)
    print("\nReplay complete")
    print(f"\nFrames processed: {frame_index}")
    print(f"Tracks created: {len(track_summary)}")
    print(f"Annotated replay: {output_path}")


if __name__ == "__main__":
    main()

