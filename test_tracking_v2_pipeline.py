import cv2
from collections import defaultdict

from detection.detection_engine import detect
from embed.embed_engine import embed
from build_observation import build_observation
from trackv2 import TrackV2, TrackV2Config


VIDEO_PATH = "videoplayback.mp4"
OUTPUT_PATH = "trackv2_output.mp4"
PRINT_ASSIGNMENT_MAP = False


def scenario_observation(detection_id, timestamp, center):
    return {
        "detection_id": detection_id,
        "timestamp": timestamp,
        "center": center,
        "bbox": [center[0] - 10, center[1] - 20, center[0] + 10, center[1] + 20],
        "confidence": 0.9,
        "embedding": [1.0, 0.0],
    }


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


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 10

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
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

    print("\n========================")
    print("RUNNING TRACKV2 PIPELINE")
    print("========================\n")
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

        for det in detections:
            emb_result = embedder.embed({
                "detection_id": det["detection_id"],
                "image": det["image"],
            })
            obs = build_observation(det, emb_result)
            observations_by_ts[timestamp].append(obs)

        tracks, assignment_map = tracker.update(observations_by_ts)
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

    if comparable_assignment_count == 0:
        stability_score = 100.0
    else:
        stability_score = 100.0 * stable_assignment_count / comparable_assignment_count

    print("\n========================")
    print("TRACKV2 DONE")
    print("========================")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"TOTAL TRACKS: {len(tracker.tracks)}")
    print(f"POTENTIAL ID SWITCHES: {identity_switch_count}")
    print(f"TRACK STABILITY SCORE: {stability_score:.2f}%")
    print(f"TRACK CONTINUITY SCORE: {stability_score:.2f}%")
    print(f"TRACK FRAGMENTATION COUNT: {track_fragmentation_count}")
    print(f"TRACK REBIRTH VIOLATIONS: {track_rebirth_violations}")


if __name__ == "__main__":
    main()
