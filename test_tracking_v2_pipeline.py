import argparse
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from events import detect_events
from events.line_overlay import draw_line_overlay
from track import TrackV2, TrackV2Config

DEFAULT_VIDEO_PATH = "videoplayback.mp4"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_VIDEO_NAME = "trackv2_output.mp4"
MAX_CONTACT_SHEET_CROPS = 100
CONTACT_TILE_SIZE = 96
CONTACT_HEADER_HEIGHT = 120
CONTACT_PADDING = 8
PROGRESS_BAR_WIDTH = 30
SEX_LABELS = {
    0: "Male",
    1: "Female",
}
RACE_LABELS = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
}
AGE_LABELS = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+",
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


def collect_track_crop(
    track_crops, display_id, observation, frame, frame_idx, timestamp
):
    crop = crop_observation(frame, observation)
    if crop is None:
        return

    track_crops[display_id].append(
        {
            "crop": crop,
            "frame_index": frame_idx,
            "timestamp": timestamp,
        }
    )


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
    tile[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return tile


def write_contact_sheet(
    output_dir, display_id, entries, max_crops=MAX_CONTACT_SHEET_CROPS
):
    if not entries:
        return None

    chronological_entries = sorted(
        entries, key=lambda entry: (entry["frame_index"], entry["timestamp"])
    )
    sampled_entries = evenly_sample_entries(chronological_entries, max_crops)
    rows, cols = contact_sheet_grid_size(len(sampled_entries))

    sheet_h = (
        CONTACT_HEADER_HEIGHT + rows * CONTACT_TILE_SIZE + (rows + 1) * CONTACT_PADDING
    )
    sheet_w = cols * CONTACT_TILE_SIZE + (cols + 1) * CONTACT_PADDING
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    first_frame = chronological_entries[0]["frame_index"]
    last_frame = chronological_entries[-1]["frame_index"]
    duration = (
        chronological_entries[-1]["timestamp"] - chronological_entries[0]["timestamp"]
    )
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
        sheet[y : y + CONTACT_TILE_SIZE, x : x + CONTACT_TILE_SIZE] = tile
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


def print_progress(
    frame_idx, total_frames, active_count, tentative_count, closed_count, event_count
):
    if total_frames > 0:
        current = min(frame_idx + 1, total_frames)
        ratio = current / total_frames
        filled = int(PROGRESS_BAR_WIDTH * ratio)
        bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
        suffix = f"{current}/{total_frames}"
    else:
        bar = "█" * (frame_idx % (PROGRESS_BAR_WIDTH + 1))
        bar = bar.ljust(PROGRESS_BAR_WIDTH, "░")
        suffix = f"frame {frame_idx}"

    print(
        f"\rProcessing {bar} {suffix} "
        f"active={active_count} tentative={tentative_count} closed={closed_count} events={event_count}",
        end="",
        flush=True,
    )


def print_event_table(events, runtime_to_display_id):
    print("\n\nEVENT TABLE")
    print("===========")
    if not events:
        print("- none")
        return

    for event in events:
        runtime_track_id = event["runtime_track_id"]
        display_id = runtime_to_display_id.get(runtime_track_id)
        track_label = (
            f"Track {display_id}" if display_id is not None else runtime_track_id
        )
        sex = SEX_LABELS[event["sex"]]
        race = RACE_LABELS[event["race"]]
        age = AGE_LABELS[event["age"]]
        print(
            f"{event['timestamp']:.3f}s | {track_label} | "
            f"{event['event_type']} | {event['direction']} | {event['event_id']} | "
            f"sex={sex} race={race} age={age}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full backend component pipeline on one video."
    )
    parser.add_argument(
        "input", nargs="?", default=DEFAULT_VIDEO_PATH, help="Input video path"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output video and crop sheets",
    )
    parser.add_argument(
        "--output-video",
        default=DEFAULT_OUTPUT_VIDEO_NAME,
        help="Output video filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.input
    output_dir = Path(args.output_dir)
    output_video_path = output_dir / args.output_video
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 10

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    line_config = {
        "point_a": [frame_w / 2.0, 0.0],
        "point_b": [frame_w / 2.0, float(frame_h)],
    }

    writer = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open output video writer: {output_video_path}")

    from detection.detection_engine import detect
    from demographics import DemographicsEngine
    from embed.embed_engine import embed
    from build_observation import build_observation

    embedder = embed()
    demographics_engine = DemographicsEngine()
    tracker = TrackV2(TrackV2Config())
    runtime_to_display_id = {}
    best_detection_by_runtime_track = {}
    demographics_by_event_id = {}
    next_display_id = 1
    track_crops = defaultdict(list)
    latest_events = []
    frame_idx = 0

    print("FULL PIPELINE")
    print("=============")
    print(f"input:  {video_path}")
    print(f"output: {output_video_path}")
    print(f"line:   {line_config['point_a']} -> {line_config['point_b']}")

    try:
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
                emb_result = embedder.embed(
                    {
                        "detection_id": det["detection_id"],
                        "image": det["image"],
                    }
                )
                obs = build_observation(det, emb_result)
                observations_by_ts[timestamp].append(obs)

            tracks, assignment_map = tracker.update(observations_by_ts)
            latest_events = detect_events(tracks, line_config)

            for runtime_track_id in assignment_map.values():
                if runtime_track_id not in runtime_to_display_id:
                    runtime_to_display_id[runtime_track_id] = next_display_id
                    next_display_id += 1

            for det in detections:
                runtime_track_id = assignment_map.get(det["detection_id"])
                if runtime_track_id is None:
                    continue

                current_best = best_detection_by_runtime_track.get(runtime_track_id)
                if (
                    current_best is None
                    or float(det["confidence"]) > float(current_best["confidence"])
                ):
                    best_detection_by_runtime_track[runtime_track_id] = det

            enriched_events = []
            for event in latest_events:
                event = dict(event)
                demographics = demographics_by_event_id.get(event["event_id"])
                if demographics is None:
                    best_detection = best_detection_by_runtime_track.get(
                        event["runtime_track_id"]
                    )
                    if best_detection is None:
                        continue
                    demographics = demographics_engine.predict(best_detection["image"])
                    demographics_by_event_id[event["event_id"]] = demographics

                event.update(demographics)
                enriched_events.append(event)
            latest_events = enriched_events

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

            draw_line_overlay(frame, line_config)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                runtime_track_id = assignment_map.get(det["detection_id"])
                label = "Track ?"
                if runtime_track_id is not None:
                    label = f"Track {runtime_to_display_id[runtime_track_id]}"

                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
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

            active_count = sum(1 for track in tracks if track.state == "ACTIVE")
            tentative_count = sum(1 for track in tracks if track.state == "TENTATIVE")
            closed_count = sum(1 for track in tracks if track.state == "CLOSED")
            print_progress(
                frame_idx,
                total_frames,
                active_count,
                tentative_count,
                closed_count,
                len(latest_events),
            )
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    contact_sheet_paths = write_contact_sheets(output_dir, track_crops)

    print("\n\nSUMMARY")
    print("=======")
    print(f"frames:         {frame_idx}")
    print(f"tracks:         {len(tracker.tracks)}")
    print(f"contact sheets: {len(contact_sheet_paths)}")
    print(f"video:          {output_video_path}")
    print_event_table(latest_events, runtime_to_display_id)


if __name__ == "__main__":
    main()
