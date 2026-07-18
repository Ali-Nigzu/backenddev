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


def confirmed_active_track_ids(tracking_state, observation_batch, config) -> set[str]:
    """Return track IDs classified as confirmed active before this frame update."""

    from track.tracker import _partition_track_indices

    timestamp = float(observation_batch["timestamp"])
    active_indices, _tentative_indices = _partition_track_indices(
        tracking_state["tracks"], timestamp, config
    )
    return {
        str(tracking_state["tracks"][index]["track_id"]) for index in active_indices
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
    frame, tracking_state, observation_batch, active_track_ids: set[str]
) -> None:
    import cv2

    height, width = frame.shape[:2]
    for track, observation in current_frame_assignments(
        tracking_state, observation_batch, active_track_ids
    ):
        bbox = observation["bbox"]
        x1 = max(0, min(width - 1, int(round(float(bbox["x1"])))))
        y1 = max(0, min(height - 1, int(round(float(bbox["y1"])))))
        x2 = max(0, min(width - 1, int(round(float(bbox["x2"])))))
        y2 = max(0, min(height - 1, int(round(float(bbox["y2"])))))
        color = track_color(str(track["track_id"]))
        label = f"Track {track['track_id']} {float(observation['confidence']):.2f}"

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
            active_track_ids = confirmed_active_track_ids(
                tracking_state, observation_batch, config
            )
            tracking_state = Track(tracking_state, observation_batch, config)
            update_track_summary(track_summary, tracking_state)
            draw_tracking_state(
                bgr_frame, tracking_state, observation_batch, active_track_ids
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


def _test_observation(detection_id, timestamp, x, y, confidence=0.9, embedding=None):
    embedding = embedding if embedding is not None else {"values": [1.0, 0.0]}
    return {
        "detection_id": str(detection_id),
        "bbox": {"x1": x - 1.0, "y1": y - 1.0, "x2": x + 1.0, "y2": y + 1.0},
        "center": {"x": float(x), "y": float(y)},
        "embedding": embedding,
        "confidence": float(confidence),
    }


def _test_batch(frame_id, timestamp, observations):
    return {
        "frame_id": str(frame_id),
        "timestamp": float(timestamp),
        "observations": observations,
    }


def test_track_v2_deterministic_birth_and_continuation():
    from track import Track, TrackV2Config

    config = TrackV2Config()
    first = _test_batch("f1", 0.0, [_test_observation("b", 0.0, 10.0, 10.0)])
    second = _test_batch("f2", 1.0, [_test_observation("a", 1.0, 12.0, 10.0)])

    state_a = {"tracks": []}
    state_b = {"tracks": []}
    Track(state_a, first, config)
    Track(state_a, second, config)
    Track(state_b, first, config)
    Track(state_b, second, config)

    assert state_a == state_b
    assert len(state_a["tracks"]) == 1
    assert state_a["tracks"][0]["track_id"] == "1"
    assert len(state_a["tracks"][0]["path"]) == 2


def test_track_v2_impossible_motion_creates_new_track():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        motion_tolerance_px=5.0,
        localization_jitter_px=0.0,
        motion_tolerance_growth_px_per_sec=0.0,
        max_physical_speed_px_per_sec=50.0,
    )
    state = {"tracks": []}

    Track(state, _test_batch("f1", 0.0, [_test_observation("a", 0.0, 0.0, 0.0)]), config)
    Track(state, _test_batch("f2", 1.0, [_test_observation("b", 1.0, 100.0, 0.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [1, 1]


def test_track_v2_appearance_does_not_rescue_impossible_motion():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        motion_tolerance_px=5.0,
        localization_jitter_px=0.0,
        motion_tolerance_growth_px_per_sec=0.0,
        max_physical_speed_px_per_sec=50.0,
        appearance_tiebreak_enabled=True,
    )
    state = {"tracks": []}
    embedding = {"values": [0.1, 0.9]}

    Track(state, _test_batch("f1", 0.0, [_test_observation("a", 0.0, 0.0, 0.0, embedding=embedding)]), config)
    Track(state, _test_batch("f2", 1.0, [_test_observation("b", 1.0, 100.0, 0.0, embedding=embedding)]), config)

    assert len(state["tracks"]) == 2


def test_track_v2_rejects_future_track_state_loudly():
    from track import Track, TrackV2Config

    state = {"tracks": []}
    Track(state, _test_batch("f1", 10.0, [_test_observation("a", 10.0, 0.0, 0.0)]), TrackV2Config())

    try:
        Track(state, _test_batch("f0", 9.0, [_test_observation("b", 9.0, 0.0, 0.0)]), TrackV2Config())
    except ValueError as exc:
        assert "newer than the observation batch" in str(exc)
    else:
        raise AssertionError("future track state should fail loudly")
