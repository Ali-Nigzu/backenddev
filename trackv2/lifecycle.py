import uuid
from typing import Dict, Iterable, List, Set

from trackv2.config import TrackV2Config
from trackv2.models import RuntimeTrackV2

TENTATIVE = "TENTATIVE"
ACTIVE = "ACTIVE"
CLOSED = "CLOSED"


def create_track(observation: Dict) -> RuntimeTrackV2:
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
    )


def matchable_tracks(tracks: Iterable[RuntimeTrackV2]) -> List[RuntimeTrackV2]:
    return [track for track in tracks if track.state != CLOSED]


def promote_if_ready(track: RuntimeTrackV2, config: TrackV2Config) -> None:
    if track.state == TENTATIVE and track.hit_count >= config.tentative_hits_to_activate:
        track.state = ACTIVE


def close_stale_tracks(
    tracks: Iterable[RuntimeTrackV2],
    matched_track_ids: Set[str],
    config: TrackV2Config,
) -> None:
    for track in tracks:
        if track.state == CLOSED or track.runtime_track_id in matched_track_ids:
            continue

        track.miss_count += 1

        if track.state == TENTATIVE and track.miss_count > config.max_misses_tentative:
            track.state = CLOSED
        elif track.state == ACTIVE and track.miss_count > config.max_misses_active:
            track.state = CLOSED
