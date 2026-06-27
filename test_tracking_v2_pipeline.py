import math
import tempfile
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

from events import detect_events
from track import TrackV2, TrackV2Config


VIDEO_PATH = "videoplayback.mp4"
OUTPUT_DIR = Path("output")
OUTPUT_PATH = OUTPUT_DIR / "trackv2_output.mp4"
MAX_CONTACT_SHEET_CROPS = 100
CONTACT_TILE_SIZE = 96
CONTACT_HEADER_HEIGHT = 120
CONTACT_PADDING = 8
PRINT_ASSIGNMENT_MAP = False
CARD9_SYNTHETIC_LINE_CONFIG = {
    "point_a": [400.0, 0.0],
    "point_b": [400.0, 100.0],
}
SECTION_WIDTH = 41


def print_section_header(title):
    print("\n" + "=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


def run_contact_sheet_self_tests_section():
    print_section_header("CONTACT SHEET SELF TESTS")
    run_contact_sheet_self_tests()
    print("PASS")


def run_card9_event_scenario_tests_section():
    print_section_header("CARD 9 SYNTHETIC TESTS")
    try:
        run_card9_event_scenario_tests()
    except AssertionError as exc:
        print("FAIL")
        print("\nReason:")
        print(exc)
        print("\nContinuing to full pipeline...")
        return False

    print("PASS")
    return True


def scenario_observation(detection_id, timestamp, center):
    return {
        "detection_id": detection_id,
        "timestamp": timestamp,
        "center": center,
        "bbox": [center[0] - 10, center[1] - 20, center[0] + 10, center[1] + 20],
        "confidence": 0.9,
        "embedding": [1.0, 0.0],
    }


def clamp_bbox_to_frame(bbox, frame_shape):
    height, width = frame_shape[:2]
    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(math.floor(x1))))
    y1 = max(0, min(height, int(math.floor(y1))))
    x2 = max(0, min(width, int(math.ceil(x2))))
    y2 = max(0, min(height, int(math.ceil(y2))))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_observation(frame, observation):
    clamped_bbox = clamp_bbox_to_frame(observation.get("bbox"), frame.shape)
    if clamped_bbox is None:
        return None

    x1, y1, x2, y2 = clamped_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def collect_track_crop(track_crops, display_id, observation, frame, frame_idx, timestamp):
    crop = crop_observation(frame, observation)
    if crop is None:
        return False

    track_crops[display_id].append({
        "crop": crop,
        "frame_index": frame_idx,
        "timestamp": timestamp,
    })
    return True


def evenly_sample_entries(entries, max_entries=MAX_CONTACT_SHEET_CROPS):
    if len(entries) <= max_entries:
        return list(entries)
    if max_entries <= 1:
        return [entries[0]]

    sampled = []
    last_index = len(entries) - 1
    for sample_idx in range(max_entries):
        source_index = round(sample_idx * last_index / (max_entries - 1))
        sampled.append(entries[source_index])
    return sampled


def contact_sheet_grid_size(count):
    if count <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def resize_to_tile(image, tile_size=CONTACT_TILE_SIZE):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(tile_size / width, tile_size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    tile = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
    y_offset = (tile_size - resized_h) // 2
    x_offset = (tile_size - resized_w) // 2
    tile[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized
    return tile


def write_contact_sheet(output_dir, display_id, entries, max_crops=MAX_CONTACT_SHEET_CROPS):
    if not entries:
        return None

    chronological_entries = sorted(entries, key=lambda entry: (entry["frame_index"], entry["timestamp"]))
    sampled_entries = evenly_sample_entries(chronological_entries, max_crops)
    rows, cols = contact_sheet_grid_size(len(sampled_entries))

    sheet_h = CONTACT_HEADER_HEIGHT + rows * CONTACT_TILE_SIZE + (rows + 1) * CONTACT_PADDING
    sheet_w = cols * CONTACT_TILE_SIZE + (cols + 1) * CONTACT_PADDING
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    first_frame = chronological_entries[0]["frame_index"]
    last_frame = chronological_entries[-1]["frame_index"]
    duration = chronological_entries[-1]["timestamp"] - chronological_entries[0]["timestamp"]
    header_lines = [
        f"TRACK: {display_id:03d}",
        f"DETECTIONS: {len(chronological_entries)}",
        f"FIRST FRAME: {first_frame}",
        f"LAST FRAME: {last_frame}",
        f"DURATION: {duration:.1f}s",
    ]
    for line_idx, line in enumerate(header_lines):
        cv2.putText(
            sheet,
            line,
            (CONTACT_PADDING, 22 + line_idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for idx, entry in enumerate(sampled_entries):
        tile = resize_to_tile(entry["crop"])
        if tile is None:
            continue
        row = idx // cols
        col = idx % cols
        y = CONTACT_HEADER_HEIGHT + CONTACT_PADDING + row * CONTACT_TILE_SIZE
        x = CONTACT_PADDING + col * CONTACT_TILE_SIZE
        sheet[y:y + CONTACT_TILE_SIZE, x:x + CONTACT_TILE_SIZE] = tile
        cv2.putText(
            sheet,
            str(entry["frame_index"]),
            (x + 3, y + CONTACT_TILE_SIZE - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"track_{display_id:03d}.png"
    if not cv2.imwrite(str(path), sheet):
        raise IOError(f"Failed to write contact sheet: {path}")
    return path


def write_contact_sheets(output_dir, track_crops):
    written_paths = []
    for display_id in sorted(track_crops):
        path = write_contact_sheet(output_dir, display_id, track_crops[display_id])
        if path is not None:
            written_paths.append(path)
    return written_paths


def print_event_table(events, title="EVENT TABLE (Card 9)"):
    print(title)
    if not events:
        print("- none")
        return

    for event in events:
        print(
            f"[EVENT] track_id={event['runtime_track_id']} "
            f"type={event['event_type']} "
            f"direction={event['direction']} "
            f"timestamp={event['timestamp']}"
        )


def run_contact_sheet_self_tests():
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_output_dir = Path(tmp_dir) / "output"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[:, :] = (20, 40, 60)

        _run_contact_sheet_self_tests_in_dir(test_output_dir, frame)

    print("CONTACT SHEET SELF TESTS PASSED")


def _run_contact_sheet_self_tests_in_dir(test_output_dir, frame):
    entries_by_track = defaultdict(list)
    assert collect_track_crop(entries_by_track, 1, {"bbox": [5, 5, 20, 20]}, frame, 3, 0.3)
    assert not collect_track_crop(entries_by_track, 99, {"bbox": [10, 10, 10, 20]}, frame, 4, 0.4)

    for idx in range(5):
        sample_frame = np.full((40, 40, 3), idx * 40, dtype=np.uint8)
        assert collect_track_crop(entries_by_track, 2, {"bbox": [-5, -5, 30, 30]}, sample_frame, idx, idx * 0.1)

    large_entries = []
    for idx in range(500):
        large_entries.append({
            "crop": np.full((8, 8, 3), idx % 255, dtype=np.uint8),
            "frame_index": idx,
            "timestamp": idx * 0.1,
        })
    entries_by_track[3] = large_entries

    sampled = evenly_sample_entries(large_entries, 100)
    assert len(sampled) == 100
    assert sampled[0]["frame_index"] == 0
    assert sampled[-1]["frame_index"] == 499
    assert [entry["frame_index"] for entry in sampled] == sorted(entry["frame_index"] for entry in sampled)

    paths = write_contact_sheets(test_output_dir, entries_by_track)
    expected_names = ["track_001.png", "track_002.png", "track_003.png"]
    assert [path.name for path in paths] == expected_names
    for path in paths:
        assert path.exists() and path.stat().st_size > 0

    tiny_video_path = test_output_dir / "trackv2_output.mp4"
    writer = cv2.VideoWriter(str(tiny_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (40, 40))
    assert writer.isOpened(), "Self-test video writer did not open"
    writer.write(frame)
    writer.release()
    assert tiny_video_path.exists() and tiny_video_path.stat().st_size > 0


def run_trackv2_scenario_tests():
    scenario_config = TrackV2Config(
        tentative_hits_to_activate=1,
        unmatched_detection_buffer_frames=1,
        max_misses_active=2,
        max_misses_tentative=1,
        min_track_lifetime_sec=0.0,
    )

    doorway_tracker = TrackV2(scenario_config)
    doorway_ids = []
    for frame_idx in range(5):
        _, assignments = doorway_tracker.update({
            frame_idx * 0.1: [
                scenario_observation(
                    f"doorway-{frame_idx}",
                    frame_idx * 0.1,
                    [100 + frame_idx * 6, 100 + ((-1) ** frame_idx) * 3],
                )
            ]
        })
        doorway_ids.extend(assignments.values())
    assert len(set(doorway_ids)) == 1, "Doorway continuity scenario fragmented one person"

    reentry_tracker = TrackV2(scenario_config)
    _, first_assignment = reentry_tracker.update({
        0.0: [scenario_observation("exit-0", 0.0, [50, 50])]
    })
    first_track_id = next(iter(first_assignment.values()))
    reentry_tracker.update({0.1: []})
    reentry_tracker.update({0.2: []})
    reentry_tracker.update({3.5: []})
    _, second_assignment = reentry_tracker.update({
        3.6: [scenario_observation("exit-1", 3.6, [50, 50])]
    })
    second_track_id = next(iter(second_assignment.values()))
    assert first_track_id != second_track_id, "Exit + re-entry scenario reused a closed track"

    single_tracker = TrackV2(scenario_config)
    single_ids = []
    for frame_idx in range(6):
        _, assignments = single_tracker.update({
            frame_idx * 0.1: [
                scenario_observation(f"single-{frame_idx}", frame_idx * 0.1, [200 + frame_idx, 100])
            ]
        })
        single_ids.extend(assignments.values())
    assert len(set(single_ids)) == 1, "Single detection stability scenario created multiple tracks"

    jitter_tracker = TrackV2(scenario_config)
    jitter_ids = []
    jitter_centers = [[300, 200], [302, 199], [299, 201], [303, 200], [301, 198]]
    for frame_idx, center in enumerate(jitter_centers):
        _, assignments = jitter_tracker.update({
            frame_idx * 0.1: [scenario_observation(f"jitter-{frame_idx}", frame_idx * 0.1, center)]
        })
        jitter_ids.extend(assignments.values())
    assert len(set(jitter_ids)) == 1, "Jitter resistance scenario fragmented one track"

    print("TRACKV2 SCENARIO TESTS PASSED")


def _event_test_tracker():
    return TrackV2(TrackV2Config(
        max_speed_px_per_sec=10000.0,
        base_motion_gate=10000.0,
        tentative_hits_to_activate=1,
        unmatched_detection_buffer_frames=1,
        max_misses_active=2,
        max_misses_tentative=1,
        min_track_lifetime_sec=0.0,
    ))


def _run_event_trajectory(points, detection_prefix="event"):
    tracker = _event_test_tracker()
    for frame_idx, center in enumerate(points):
        timestamp = frame_idx * 0.1
        tracker.update({
            timestamp: [
                scenario_observation(
                    f"{detection_prefix}-{frame_idx}",
                    timestamp,
                    center,
                )
            ]
        })
    return tracker.tracks


def run_card9_event_scenario_tests():
    crossing_tracks = _run_event_trajectory(
        [[390, 50], [395, 50], [405, 50], [410, 50]],
        "crossing",
    )
    crossing_events = detect_events(crossing_tracks, CARD9_SYNTHETIC_LINE_CONFIG)
    assert len(crossing_events) == 1, "Card 9 crossing scenario did not emit exactly one event"
    assert crossing_events[0]["event_type"] == "ENTRY", "Card 9 crossing scenario emitted wrong event type"
    assert crossing_events[0]["direction"] == "IN", "Card 9 crossing scenario emitted wrong direction"
    assert crossing_events[0]["runtime_track_id"] == crossing_tracks[0].runtime_track_id

    non_crossing_tracks = _run_event_trajectory(
        [[410, 10], [410, 30], [410, 50], [410, 70]],
        "parallel",
    )
    assert detect_events(non_crossing_tracks, CARD9_SYNTHETIC_LINE_CONFIG) == [], (
        "Card 9 no-crossing scenario emitted an event"
    )

    oscillation_tracks = _run_event_trajectory(
        [[390, 50], [405, 50], [390, 50]],
        "oscillation",
    )
    assert detect_events(oscillation_tracks, CARD9_SYNTHETIC_LINE_CONFIG) == [], (
        "Card 9 oscillation scenario emitted an event"
    )

    reverse_oscillation_tracks = _run_event_trajectory(
        [[410, 50], [395, 50], [410, 50]],
        "reverse-oscillation",
    )
    assert detect_events(reverse_oscillation_tracks, CARD9_SYNTHETIC_LINE_CONFIG) == [], (
        "Card 9 reverse oscillation scenario emitted an event"
    )

    two_point_crossing_tracks = _run_event_trajectory(
        [[395, 50], [405, 50]],
        "two-point-crossing",
    )
    assert detect_events(two_point_crossing_tracks, CARD9_SYNTHETIC_LINE_CONFIG) == [], (
        "Card 9 two-point crossing scenario emitted an event without enough terminal evidence"
    )

    terminal_entry_tracks = _run_event_trajectory(
        [[390, 50], [395, 50], [405, 50]],
        "terminal-entry",
    )
    terminal_entry_events = detect_events(terminal_entry_tracks, CARD9_SYNTHETIC_LINE_CONFIG)
    assert len(terminal_entry_events) == 1, "Card 9 terminal entry scenario did not emit exactly one event"
    assert terminal_entry_events[0]["event_type"] == "ENTRY", (
        "Card 9 terminal entry scenario emitted wrong event type"
    )
    assert terminal_entry_events[0]["direction"] == "IN", (
        "Card 9 terminal entry scenario emitted wrong direction"
    )

    terminal_exit_tracks = _run_event_trajectory(
        [[410, 50], [405, 50], [395, 50]],
        "terminal-exit",
    )
    terminal_exit_events = detect_events(terminal_exit_tracks, CARD9_SYNTHETIC_LINE_CONFIG)
    assert len(terminal_exit_events) == 1, "Card 9 terminal exit scenario did not emit exactly one event"
    assert terminal_exit_events[0]["event_type"] == "EXIT", (
        "Card 9 terminal exit scenario emitted wrong event type"
    )
    assert terminal_exit_events[0]["direction"] == "OUT", (
        "Card 9 terminal exit scenario emitted wrong direction"
    )

    multi_tracker = _event_test_tracker()
    multi_trajectories = {
        "crossing": [[390, 50], [395, 50], [405, 50], [410, 50]],
        "right-side": [[420, 20], [420, 25], [420, 30], [420, 35]],
        "left-side": [[380, 20], [380, 25], [380, 30], [380, 35]],
    }
    assigned_ids_by_prefix = defaultdict(set)
    for frame_idx in range(4):
        timestamp = frame_idx * 0.1
        observations = [
            scenario_observation(prefix, timestamp, points[frame_idx])
            for prefix, points in multi_trajectories.items()
        ]
        _, assignments = multi_tracker.update({timestamp: observations})
        for detection_id, runtime_track_id in assignments.items():
            assigned_ids_by_prefix[detection_id].add(runtime_track_id)

    multi_events = detect_events(multi_tracker.tracks, CARD9_SYNTHETIC_LINE_CONFIG)
    crossing_track_id = next(iter(assigned_ids_by_prefix["crossing"]))
    assert len(multi_events) == 1, "Card 9 multi-track scenario emitted wrong event count"
    assert multi_events[0]["runtime_track_id"] == crossing_track_id, (
        "Card 9 multi-track scenario emitted an event for the wrong track"
    )

    deterministic_once = detect_events(multi_tracker.tracks, CARD9_SYNTHETIC_LINE_CONFIG)
    deterministic_twice = detect_events(multi_tracker.tracks, CARD9_SYNTHETIC_LINE_CONFIG)
    deterministic_reversed = detect_events(list(reversed(multi_tracker.tracks)), CARD9_SYNTHETIC_LINE_CONFIG)
    assert deterministic_once == deterministic_twice == deterministic_reversed, (
        "Card 9 event detection is not deterministic"
    )

    print("CARD9 EVENT SCENARIO TESTS PASSED")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section_header("FULL PIPELINE")

    from detection.detection_engine import detect
    from embed.embed_engine import embed
    from build_observation import build_observation


    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 10

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_config = {
        "point_a": [frame_w / 2.0, 0.0],
        "point_b": [frame_w / 2.0, float(frame_h)],
    }

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    embedder = embed()
    tracker = TrackV2(TrackV2Config())

    runtime_to_display_id = {}
    next_display_id = 1
    previous_assignment_by_detection = {}
    previous_frame_assignments = []
    stable_assignment_count = 0
    comparable_assignment_count = 0
    identity_switch_count = 0
    track_fragmentation_count = 0
    track_rebirth_violations = 0
    frame_idx = 0
    track_crops = defaultdict(list)
    latest_events = []

    print("TRACKV2 METHOD LOG:")
    print("- enforced strict motion-first continuity")
    print("- disabled bbox influence on identity decisions")
    print("- added 3-frame unmatched buffering before birth")
    print("- added 3-second cooldown for closed tracks")
    print("- added forced continuity priority over track creation")
    run_trackv2_scenario_tests()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx * (1.0 / fps)
        frame_packet = {
            "frame_id": f"frame-{frame_idx}",
            "timestamp": timestamp,
            "image": frame,
        }

        detections = detect(frame_packet)
        observations_by_ts = defaultdict(list)
        observations_by_ts.setdefault(timestamp, [])

        for det in detections:
            emb_result = embedder.embed({
                "detection_id": det["detection_id"],
                "image": det["image"],
            })
            obs = build_observation(det, emb_result)
            observations_by_ts[timestamp].append(obs)

        tracks, assignment_map = tracker.update(observations_by_ts)
        latest_events = detect_events(tracks, line_config)
        closed_tracks = {
            track.runtime_track_id: track
            for track in tracks
            if track.state == "CLOSED" and track.closed_timestamp is not None
        }

        for runtime_track_id in assignment_map.values():
            closed_track = closed_tracks.get(runtime_track_id)
            if closed_track is None:
                continue

            if timestamp - closed_track.closed_timestamp < tracker.config.closed_track_cooldown_sec:
                track_rebirth_violations += 1

        for runtime_track_id in assignment_map.values():
            if runtime_track_id not in runtime_to_display_id:
                runtime_to_display_id[runtime_track_id] = next_display_id
                next_display_id += 1

        display_assignment_map = {
            detection_id: runtime_to_display_id[runtime_track_id]
            for detection_id, runtime_track_id in assignment_map.items()
        }

        current_frame_assignments = []
        for obs in observations_by_ts[timestamp]:
            runtime_track_id = assignment_map.get(obs["detection_id"])
            if runtime_track_id is None:
                continue

            collect_track_crop(
                track_crops,
                runtime_to_display_id[runtime_track_id],
                obs,
                frame,
                frame_idx,
                timestamp,
            )

            current_frame_assignments.append({
                "detection_id": obs["detection_id"],
                "center": obs["center"],
                "runtime_track_id": runtime_track_id,
                "display_id": runtime_to_display_id[runtime_track_id],
            })

        for detection_id, runtime_track_id in assignment_map.items():
            previous_track_id = previous_assignment_by_detection.get(detection_id)
            if previous_track_id is not None:
                comparable_assignment_count += 1
                if previous_track_id == runtime_track_id:
                    stable_assignment_count += 1
                else:
                    identity_switch_count += 1
                    track_fragmentation_count += 1
                    print(
                        "POTENTIAL ID SWITCH DETECTED "
                        f"frame={frame_idx} detection_id={detection_id} "
                        f"from=Track {runtime_to_display_id[previous_track_id]} "
                        f"to=Track {runtime_to_display_id[runtime_track_id]}"
                    )
            previous_assignment_by_detection[detection_id] = runtime_track_id

        for current in current_frame_assignments:
            if not previous_frame_assignments:
                continue

            nearest_previous = min(
                previous_frame_assignments,
                key=lambda previous: (
                    (previous["center"][0] - current["center"][0]) ** 2
                    + (previous["center"][1] - current["center"][1]) ** 2
                ),
            )
            squared_distance = (
                (nearest_previous["center"][0] - current["center"][0]) ** 2
                + (nearest_previous["center"][1] - current["center"][1]) ** 2
            )

            if squared_distance > tracker.config.base_motion_gate ** 2:
                continue

            comparable_assignment_count += 1
            if nearest_previous["runtime_track_id"] == current["runtime_track_id"]:
                stable_assignment_count += 1
            else:
                identity_switch_count += 1
                track_fragmentation_count += 1
                print(
                    "POTENTIAL ID SWITCH DETECTED "
                    f"frame={frame_idx} previous_detection_id={nearest_previous['detection_id']} "
                    f"current_detection_id={current['detection_id']} "
                    f"from=Track {nearest_previous['display_id']} "
                    f"to=Track {current['display_id']}"
                )

        active_count = sum(1 for track in tracks if track.state == "ACTIVE")
        tentative_count = sum(1 for track in tracks if track.state == "TENTATIVE")
        closed_count = sum(1 for track in tracks if track.state == "CLOSED")

        print(
            f"frame={frame_idx} active={active_count} tentative={tentative_count} "
            f"closed={closed_count} new_tracks={tracker.last_new_tracks_created} "
            f"assignments={len(assignment_map)} "
            f"tracks_continued={tracker.last_debug_report['tracks_continued']} "
            f"tracks_rejected_by_motion_gate={tracker.last_debug_report['tracks_rejected_by_motion_gate']} "
            f"forced_continuations={tracker.last_debug_report['forced_continuations']}"
        )

        if PRINT_ASSIGNMENT_MAP:
            print(f"assignment_map={display_assignment_map}")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            runtime_track_id = assignment_map.get(det["detection_id"], "?")
            label = (
                f"Track {runtime_to_display_id[runtime_track_id]}"
                if runtime_track_id != "?"
                else "Track ?"
            )

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                label,
                (int(x1), max(20, int(y1))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(frame)
        previous_frame_assignments = current_frame_assignments
        frame_idx += 1

    cap.release()
    writer.release()

    contact_sheet_paths = write_contact_sheets(OUTPUT_DIR, track_crops)

    if comparable_assignment_count == 0:
        stability_score = 100.0
    else:
        stability_score = 100.0 * stable_assignment_count / comparable_assignment_count

    print("\n========================")
    print("TRACKV2 DONE")
    print("========================")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"CONTACT SHEETS: {len(contact_sheet_paths)}")
    print(f"TOTAL TRACKS: {len(tracker.tracks)}")
    print(f"POTENTIAL ID SWITCHES: {identity_switch_count}")
    print(f"TRACK STABILITY SCORE: {stability_score:.2f}%")
    print(f"TRACK CONTINUITY SCORE: {stability_score:.2f}%")
    print(f"TRACK FRAGMENTATION COUNT: {track_fragmentation_count}")
    print(f"TRACK REBIRTH VIOLATIONS: {track_rebirth_violations}")
    print_event_table(latest_events)


if __name__ == "__main__":
    run_contact_sheet_self_tests_section()
    run_card9_event_scenario_tests_section()
    main()
