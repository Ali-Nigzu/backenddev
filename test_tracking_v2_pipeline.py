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
    frame,
    tracking_state,
    observation_batch,
    active_track_ids: set[str],
    previous_track_ids: set[str] | None = None,
) -> dict:
    import cv2

    previous_track_ids = previous_track_ids or set()
    timestamp = float(observation_batch["timestamp"])
    assignments = current_frame_assignments(tracking_state, observation_batch, active_track_ids)
    matched_track_ids = {str(track["track_id"]) for track, _observation in assignments}
    birth_track_ids = {
        str(track["track_id"])
        for track in tracking_state["tracks"]
        if str(track["track_id"]) not in previous_track_ids
        and float(track["path"][-1]["timestamp"]) == timestamp
    }
    unmatched_active_ids = set(active_track_ids) - matched_track_ids

    height, width = frame.shape[:2]
    for track, observation in assignments:
        bbox = observation["bbox"]
        x1 = max(0, min(width - 1, int(round(float(bbox["x1"])))))
        y1 = max(0, min(height - 1, int(round(float(bbox["y1"])))))
        x2 = max(0, min(width - 1, int(round(float(bbox["x2"])))))
        y2 = max(0, min(height - 1, int(round(float(bbox["y2"])))))
        track_id = str(track["track_id"])
        color = (0, 0, 255) if track_id in birth_track_ids else track_color(track_id)
        label_prefix = "BIRTH" if track_id in birth_track_ids else "Track"
        label = f"{label_prefix} {track['track_id']} {float(observation['confidence']):.2f}"

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

    for track in tracking_state["tracks"]:
        track_id = str(track["track_id"])
        if track_id not in unmatched_active_ids:
            continue
        center = track["path"][-1]["center"]
        x = max(0, min(width - 1, int(round(float(center["x"])))))
        y = max(0, min(height - 1, int(round(float(center["y"])))))
        cv2.circle(frame, (x, y), 8, (160, 160, 160), 2)
        cv2.putText(frame, f"miss {track_id}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    debug = {
        "active": len(active_track_ids),
        "observations": len(observation_batch["observations"]),
        "matched": len(matched_track_ids),
        "births": len(birth_track_ids),
        "unmatched_active": len(unmatched_active_ids),
    }
    overlay = (
        f"active={debug['active']} obs={debug['observations']} "
        f"matched={debug['matched']} births={debug['births']} "
        f"unmatched_active={debug['unmatched_active']}"
    )
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
            active_track_ids = confirmed_active_track_ids(
                tracking_state, observation_batch, config
            )
            previous_track_ids = {str(track["track_id"]) for track in tracking_state["tracks"]}
            tracking_state = Track(tracking_state, observation_batch, config)
            update_track_summary(track_summary, tracking_state)
            debug = draw_tracking_state(
                bgr_frame, tracking_state, observation_batch, active_track_ids, previous_track_ids
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
        birth_suppression_strength=0.0,
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
        birth_suppression_strength=0.0,
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


def test_track_v2_standing_detector_jitter_keeps_one_identity():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        motion_tolerance_px=12.0,
        localization_jitter_px=8.0,
        motion_tolerance_growth_px_per_sec=4.0,
        max_physical_speed_px_per_sec=120.0,
    )
    state = {"tracks": []}
    positions = [
        (100.0, 100.0),
        (102.5, 98.0),
        (99.5, 101.5),
        (101.0, 99.0),
        (100.5, 100.5),
    ]

    for index, (x, y) in enumerate(positions):
        Track(state, _test_batch(f"f{index}", index / 10.0, [_test_observation(index, index / 10.0, x, y)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == len(positions)


def test_track_v2_short_missed_detection_keeps_confirmed_identity():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        confirmation_hits=2,
        detector_miss_tolerance_sec=1.0,
        motion_tolerance_px=20.0,
        localization_jitter_px=5.0,
        motion_tolerance_growth_px_per_sec=20.0,
        max_physical_speed_px_per_sec=150.0,
    )
    state = {"tracks": []}

    Track(state, _test_batch("f0", 0.0, [_test_observation("a", 0.0, 50.0, 50.0)]), config)
    Track(state, _test_batch("f1", 0.1, [_test_observation("b", 0.1, 52.0, 50.0)]), config)
    Track(state, _test_batch("f2", 0.2, []), config)
    Track(state, _test_batch("f3", 0.5, [_test_observation("c", 0.5, 58.0, 51.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3


def test_track_v2_slow_walker_survives_localization_wobble():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        motion_tolerance_px=15.0,
        localization_jitter_px=6.0,
        motion_tolerance_growth_px_per_sec=8.0,
        max_physical_speed_px_per_sec=140.0,
    )
    state_a = {"tracks": []}
    state_b = {"tracks": []}
    positions = [(20.0, 20.0), (23.0, 21.0), (25.0, 19.5), (29.0, 21.0), (32.0, 20.0)]
    batches = [
        _test_batch(f"f{index}", index * 0.2, [_test_observation(chr(97 + index), index * 0.2, x, y)])
        for index, (x, y) in enumerate(positions)
    ]

    for batch in batches:
        Track(state_a, batch, config)
    for batch in batches:
        Track(state_b, batch, config)

    assert state_a == state_b
    assert [track["track_id"] for track in state_a["tracks"]] == ["1"]
    assert len(state_a["tracks"][0]["path"]) == len(positions)


def test_track_v2_same_timestamp_position_change_uses_fallback_when_births_suppressed():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        motion_tolerance_px=20.0,
        localization_jitter_px=2.0,
        max_physical_speed_px_per_sec=500.0,
    )
    state = {"tracks": []}

    Track(state, _test_batch("f0", 1.0, [_test_observation("a", 1.0, 10.0, 10.0)]), config)
    Track(state, _test_batch("f0b", 1.0, [_test_observation("b", 1.0, 20.0, 10.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 2


def test_track_v2_continuity_beats_microscopic_motion_advantage():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        confirmation_hits=1,
        motion_tolerance_px=30.0,
        localization_jitter_px=10.0,
        continuity_strength=0.2,
        takeover_margin=0.5,
        max_physical_speed_px_per_sec=200.0,
    )
    state = {"tracks": []}
    Track(state, _test_batch("f0", 0.0, [_test_observation("a", 0.0, 0.0, 0.0)]), config)
    Track(state, _test_batch("f1", 0.1, [_test_observation("b", 0.1, 2.0, 0.0)]), config)
    # Create a newer but less continuous challenger close to the next observation.
    Track(state, _test_batch("f2", 0.2, [_test_observation("c", 0.2, 100.0, 0.0)]), config)
    Track(state, _test_batch("f3", 0.3, [_test_observation("d", 0.3, 4.0, 0.0)]), config)

    track_one = next(track for track in state["tracks"] if track["track_id"] == "1")
    assert len(track_one["path"]) == 4


def _multi_observations(prefix, timestamp, count, offset_x=0.0, offset_y=0.0):
    return [
        _test_observation(
            f"{prefix}-{index:02d}",
            timestamp,
            50.0 + index * 50.0 + offset_x,
            100.0 + offset_y,
        )
        for index in range(count)
    ]


def _seed_confirmed_tracks(count=8, config=None):
    from track import Track, TrackV2Config

    config = config or TrackV2Config(confirmation_hits=2, detector_miss_tolerance_sec=2.0)
    state = {"tracks": []}
    Track(state, _test_batch("seed-0", 0.0, _multi_observations("seed0", 0.0, count)), config)
    Track(state, _test_batch("seed-1", 0.1, _multi_observations("seed1", 0.1, count, offset_x=1.0)), config)
    assert len(state["tracks"]) == count
    return state, config


def test_track_v2_birth_suppression_forbids_births_when_active_equals_observations():
    from track import Track, TrackV2Config

    state, config = _seed_confirmed_tracks(
        8,
        TrackV2Config(
            confirmation_hits=2,
            detector_miss_tolerance_sec=2.0,
            motion_tolerance_px=8.0,
            localization_jitter_px=2.0,
            max_physical_speed_px_per_sec=30.0,
        ),
    )

    Track(state, _test_batch("jitter", 0.2, _multi_observations("jitter", 0.2, 8, offset_x=20.0)), config)

    assert len(state["tracks"]) == 8
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 3) == 8


def test_track_v2_birth_suppression_forbids_births_when_active_exceeds_observations():
    from track import Track

    state, config = _seed_confirmed_tracks(8)

    Track(state, _test_batch("partial", 0.2, _multi_observations("partial", 0.2, 7, offset_x=15.0)), config)

    assert len(state["tracks"]) == 8
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 3) == 7
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 2) == 1


def test_track_v2_birth_suppression_allows_only_overflow_births():
    from track import Track

    state, config = _seed_confirmed_tracks(8)

    observations = _multi_observations("overflow", 0.2, 8, offset_x=2.0)
    observations.extend(
        [
            _test_observation("overflow-new-08", 0.2, 600.0, 300.0),
            _test_observation("overflow-new-09", 0.2, 650.0, 300.0),
        ]
    )
    Track(state, _test_batch("overflow", 0.2, observations), config)

    assert len(state["tracks"]) == 10
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 1) == 2


def test_track_v2_birth_suppression_startup_still_births_when_no_active_tracks():
    from track import Track, TrackV2Config

    state = {"tracks": []}
    Track(state, _test_batch("startup", 0.0, _multi_observations("startup", 0.0, 5)), TrackV2Config())

    assert len(state["tracks"]) == 5
    assert all(len(track["path"]) == 1 for track in state["tracks"])


def test_track_v2_birth_suppression_restores_after_temporary_occlusion():
    from track import Track, TrackV2Config

    state, config = _seed_confirmed_tracks(
        8,
        TrackV2Config(confirmation_hits=2, detector_miss_tolerance_sec=2.0, motion_tolerance_px=10.0),
    )

    Track(state, _test_batch("occluded", 0.2, _multi_observations("occ", 0.2, 6, offset_x=5.0)), config)
    Track(state, _test_batch("restored", 0.3, _multi_observations("restore", 0.3, 8, offset_x=8.0)), config)

    assert len(state["tracks"]) == 8
    assert sum(1 for track in state["tracks"] if len(track["path"]) >= 3) == 8


def test_track_v2_rejects_invalid_birth_suppression_strength():
    from track import Track, TrackV2Config

    state = {"tracks": []}
    try:
        Track(
            state,
            _test_batch("bad", 0.0, [_test_observation("a", 0.0, 0.0, 0.0)]),
            TrackV2Config(birth_suppression_strength=1.1),
        )
    except ValueError as exc:
        assert "birth_suppression_strength" in str(exc)
    else:
        raise AssertionError("invalid birth_suppression_strength should fail loudly")


def test_track_v2_hard_suppression_absorbs_when_normal_motion_rejects_all():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        confirmation_hits=2,
        detector_miss_tolerance_sec=2.0,
        motion_tolerance_px=1.0,
        localization_jitter_px=0.0,
        motion_tolerance_growth_px_per_sec=0.0,
        max_physical_speed_px_per_sec=1.0,
        birth_suppression_strength=1.0,
    )
    state, _ = _seed_confirmed_tracks(3, config)

    Track(
        state,
        _test_batch(
            "teleport-but-suppressed",
            0.2,
            [
                _test_observation("a", 0.2, 300.0, 300.0),
                _test_observation("b", 0.2, 350.0, 300.0),
                _test_observation("c", 0.2, 400.0, 300.0),
            ],
        ),
        config,
    )

    assert len(state["tracks"]) == 3
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 3) == 3


def test_track_v2_hard_suppression_limits_births_to_overflow_when_motion_rejects_all():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        confirmation_hits=2,
        detector_miss_tolerance_sec=2.0,
        motion_tolerance_px=1.0,
        localization_jitter_px=0.0,
        motion_tolerance_growth_px_per_sec=0.0,
        max_physical_speed_px_per_sec=1.0,
        birth_suppression_strength=1.0,
    )
    state, _ = _seed_confirmed_tracks(2, config)

    Track(
        state,
        _test_batch(
            "one-overflow",
            0.2,
            [
                _test_observation("a", 0.2, 300.0, 300.0),
                _test_observation("b", 0.2, 350.0, 300.0),
                _test_observation("c", 0.2, 400.0, 300.0),
            ],
        ),
        config,
    )

    assert len(state["tracks"]) == 3
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 3) == 2
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 1) == 1


def test_track_v2_intermediate_suppression_reduces_explainable_births():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        confirmation_hits=2,
        detector_miss_tolerance_sec=2.0,
        motion_tolerance_px=1.0,
        localization_jitter_px=0.0,
        motion_tolerance_growth_px_per_sec=0.0,
        max_physical_speed_px_per_sec=1.0,
        birth_suppression_strength=0.5,
    )
    state, _ = _seed_confirmed_tracks(4, TrackV2Config(confirmation_hits=2, detector_miss_tolerance_sec=2.0))

    Track(
        state,
        _test_batch(
            "half-suppressed",
            0.2,
            [
                _test_observation("a", 0.2, 300.0, 300.0),
                _test_observation("b", 0.2, 350.0, 300.0),
                _test_observation("c", 0.2, 400.0, 300.0),
                _test_observation("d", 0.2, 450.0, 300.0),
            ],
        ),
        config,
    )

    assert len(state["tracks"]) == 6
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 3) == 2
    assert sum(1 for track in state["tracks"] if len(track["path"]) == 1) == 2


def test_track_v2_moderate_walker_keeps_identity():
    from track import Track, TrackV2Config

    config = TrackV2Config(
        motion_tolerance_px=18.0,
        localization_jitter_px=8.0,
        motion_tolerance_growth_px_per_sec=8.0,
        max_physical_speed_px_per_sec=260.0,
    )
    state = {"tracks": []}
    positions = [(20.0, 20.0), (32.0, 21.0), (45.0, 19.0), (57.0, 22.0), (70.0, 21.0)]

    for index, (x, y) in enumerate(positions):
        timestamp = index * 0.1
        Track(state, _test_batch(f"moderate-{index}", timestamp, [_test_observation(index, timestamp, x, y)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == len(positions)


def test_track_v2_replay_draw_returns_decision_counts():
    import pytest
    cv2 = pytest.importorskip("cv2")
    from track import Track, TrackV2Config

    config = TrackV2Config()
    state = {"tracks": []}
    batch = _test_batch("f0", 0.0, [_test_observation("a", 0.0, 10.0, 10.0)])
    previous_track_ids = set()
    active_track_ids = confirmed_active_track_ids(state, batch, config)
    Track(state, batch, config)
    frame = cv2.UMat(64, 64, cv2.CV_8UC3).get()

    debug = draw_tracking_state(frame, state, batch, active_track_ids, previous_track_ids)

    assert debug == {"active": 0, "observations": 1, "matched": 0, "births": 1, "unmatched_active": 0}
