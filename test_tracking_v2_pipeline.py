import cv2
from collections import defaultdict

from detection.detection_engine import detect
from embed.embed_engine import embed
from build_observation import build_observation
from trackv2 import TrackV2, TrackV2Config


VIDEO_PATH = "videoplayback.mp4"
OUTPUT_PATH = "trackv2_output.mp4"
PRINT_ASSIGNMENT_MAP = False


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

    previous_assignment_by_detection = {}
    previous_frame_assignments = []
    stable_assignment_count = 0
    comparable_assignment_count = 0
    identity_switch_count = 0
    frame_idx = 0

    print("\n========================")
    print("RUNNING TRACKV2 PIPELINE")
    print("========================\n")

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

        current_frame_assignments = []
        for obs in observations_by_ts[timestamp]:
            runtime_track_id = assignment_map.get(obs["detection_id"])
            if runtime_track_id is None:
                continue

            current_frame_assignments.append({
                "detection_id": obs["detection_id"],
                "center": obs["center"],
                "runtime_track_id": runtime_track_id,
            })

        for detection_id, runtime_track_id in assignment_map.items():
            previous_track_id = previous_assignment_by_detection.get(detection_id)
            if previous_track_id is not None:
                comparable_assignment_count += 1
                if previous_track_id == runtime_track_id:
                    stable_assignment_count += 1
                else:
                    identity_switch_count += 1
                    print(
                        "POTENTIAL ID SWITCH DETECTED "
                        f"frame={frame_idx} detection_id={detection_id} "
                        f"from={previous_track_id} to={runtime_track_id}"
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
                print(
                    "POTENTIAL ID SWITCH DETECTED "
                    f"frame={frame_idx} previous_detection_id={nearest_previous['detection_id']} "
                    f"current_detection_id={current['detection_id']} "
                    f"from={nearest_previous['runtime_track_id']} "
                    f"to={current['runtime_track_id']}"
                )

        active_count = sum(1 for track in tracks if track.state == "ACTIVE")
        tentative_count = sum(1 for track in tracks if track.state == "TENTATIVE")
        closed_count = sum(1 for track in tracks if track.state == "CLOSED")

        print(
            f"frame={frame_idx} active={active_count} tentative={tentative_count} "
            f"closed={closed_count} new_tracks={tracker.last_new_tracks_created} "
            f"assignments={len(assignment_map)}"
        )

        if PRINT_ASSIGNMENT_MAP:
            print(f"assignment_map={assignment_map}")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            runtime_track_id = assignment_map.get(det["detection_id"], "?")
            label = f"T{runtime_track_id[:8]}" if runtime_track_id != "?" else "T?"

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


if __name__ == "__main__":
    main()
