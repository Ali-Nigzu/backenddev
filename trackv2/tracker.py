from typing import Dict, List, Tuple

from trackv2.config import TrackV2Config
from trackv2.lifecycle import create_track, close_stale_tracks, matchable_tracks, promote_if_ready
from trackv2.matching import assign_matches
from trackv2.models import RuntimeTrackV2
from trackv2.motion import compute_velocity


class TrackV2:
    def __init__(self, config: TrackV2Config | None = None):
        self.config = config or TrackV2Config()
        self.tracks: List[RuntimeTrackV2] = []
        self.last_new_tracks_created = 0

    def update(self, observations_by_ts: Dict[float, List[Dict]]) -> Tuple[List[RuntimeTrackV2], Dict[str, str]]:
        assignment_map: Dict[str, str] = {}
        self.last_new_tracks_created = 0

        for timestamp in sorted(observations_by_ts.keys()):
            observations = observations_by_ts[timestamp]
            eligible_tracks = matchable_tracks(self.tracks)

            matches, unmatched_track_indices, unmatched_observation_indices, gated_observation_indices = assign_matches(
                eligible_tracks,
                observations,
                self.config,
            )

            matched_track_ids = set()

            for track_index, observation_index in matches:
                track = eligible_tracks[track_index]
                observation = observations[observation_index]
                previous_center = track.current_center
                dt = float(observation["timestamp"]) - float(track.last_seen_timestamp)
                measured_velocity = compute_velocity(previous_center, observation["center"], dt)

                track.velocity = [
                    self.config.velocity_smoothing * track.velocity[0]
                    + (1.0 - self.config.velocity_smoothing) * measured_velocity[0],
                    self.config.velocity_smoothing * track.velocity[1]
                    + (1.0 - self.config.velocity_smoothing) * measured_velocity[1],
                ]
                track.current_center = [float(observation["center"][0]), float(observation["center"][1])]
                track.current_bbox = [float(v) for v in observation["bbox"]]
                track.last_embedding = observation.get("embedding")
                track.last_seen_timestamp = float(observation["timestamp"])
                track.hit_count += 1
                track.miss_count = 0
                track.detection_history.append(observation["detection_id"])
                track.center_history.append(track.current_center)
                promote_if_ready(track, self.config)

                matched_track_ids.add(track.runtime_track_id)
                assignment_map[observation["detection_id"]] = track.runtime_track_id

            close_stale_tracks(eligible_tracks, matched_track_ids, self.config)

            for observation_index in sorted(unmatched_observation_indices):
                if observation_index in gated_observation_indices:
                    continue

                track = create_track(observations[observation_index])
                promote_if_ready(track, self.config)
                self.tracks.append(track)
                self.last_new_tracks_created += 1
                assignment_map[observations[observation_index]["detection_id"]] = track.runtime_track_id

        return list(self.tracks), assignment_map
