"""Replay Detect -> Embed -> Observe -> Track over an input video.

This script intentionally stops at Track V2. It does not run Event,
Demographic, Assemble, or Output stages.
"""

import argparse
from pathlib import Path


DEFAULT_VIDEO_PATH = "videoplayback.mp4"


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
    return parser.parse_args()


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
            tracking_state = Track(tracking_state, observation_batch, config)
            update_track_summary(track_summary, tracking_state)

            frame_index += 1
            print(f"\rprocessed frames: {frame_index}", end="", flush=True)
    finally:
        cap.release()

    print_track_summary(track_summary, frame_index)


if __name__ == "__main__":
    main()
