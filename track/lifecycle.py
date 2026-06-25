import uuid
from typing import Dict, Iterable, List, Set

from track.config import TrackV2Config
from track.models import RuntimeTrackV2

TENTATIVE = "TENTATIVE"
ACTIVE = "ACTIVE"
CLOSED = "CLOSED"


def create_track(observation: Dict, frame_index: int = 0) -> RuntimeTrackV2:
    center = [float(observation["center"][0]), float(observation["center"][1])]
    bbox = [float(v) for v in observation["bbox"]]

    return RuntimeTrackV2(
        runtime_track_id=str(uuid.uuid4()),
        state=TENTATIVE,
        current_center=center,
        current_bbox=bbox,
        velocity=[0.0, 0.0],
        first_seen_timestamp=float(observation["timestamp"]),
        last_seen_timestamp=float(observation["timestamp"]),
        hit_count=1,
        miss_count=0,
        detection_history=[observation["detection_id"]],
        center_history=[center],
        last_embedding=observation.get("embedding"),
        closed_timestamp=None,
        created_frame_index=frame_index,
        last_matched_frame_index=frame_index,
        last_unmatched_frame_index=None,
    )


def matchable_tracks(tracks: Iterable[RuntimeTrackV2]) -> List[RuntimeTrackV2]:
    active = [track for track in tracks if track.state == ACTIVE]
    tentative = [track for track in tracks if track.state == TENTATIVE]
    return active + tentative


def promote_if_ready(track: RuntimeTrackV2, config: TrackV2Config) -> None:
    if track.state == TENTATIVE and track.hit_count >= config.tentative_hits_to_activate:
        track.state = ACTIVE


def close_track(track: RuntimeTrackV2, current_timestamp: float) -> None:
    if track.state == CLOSED:
        return

    track.state = CLOSED
    track.closed_timestamp = float(current_timestamp)


def close_stale_tracks(
    tracks: Iterable[RuntimeTrackV2],
    matched_track_ids: Set[str],
    config: TrackV2Config,
    current_timestamp: float,
    frame_index: int,
) -> None:
    for track in tracks:
        if track.state == CLOSED or track.runtime_track_id in matched_track_ids:
            continue

        track.miss_count += 1
        track.last_unmatched_frame_index = frame_index

        track_age = float(current_timestamp) - float(track.first_seen_timestamp)
        if track_age < config.min_track_lifetime_sec:
            continue

        if track.state == TENTATIVE and track.miss_count > config.max_misses_tentative:
            close_track(track, current_timestamp)
        elif track.state == ACTIVE and track.miss_count > config.max_misses_active:
            close_track(track, current_timestamp)
