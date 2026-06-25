from typing import Dict, List, Tuple

from track.config import TrackV2Config
from track.lifecycle import create_track, close_stale_tracks, matchable_tracks, promote_if_ready
from track.matching import assign_matches
from track.models import RuntimeTrackV2
from track.motion import compute_velocity, distance, predict_center


class TrackV2:
    def __init__(self, config: TrackV2Config | None = None):
        self.config = config or TrackV2Config()
        self.tracks: List[RuntimeTrackV2] = []
        self.pending_births: List[Dict] = []
        self.frame_index = 0
        self.last_update_timestamp: float | None = None
        self.last_new_tracks_created = 0
        self.last_debug_report = self._empty_debug_report()

    def _empty_debug_report(self) -> Dict[str, int]:
        return {
            "new_tracks_created": 0,
            "tracks_continued": 0,
            "tracks_rejected_by_motion_gate": 0,
            "forced_continuations": 0,
        }

    def _strict_motion_gate(self, track: RuntimeTrackV2, observation: Dict) -> bool:
        dt = float(observation["timestamp"]) - float(track.last_seen_timestamp)
        if dt < 0:
            return False

        predicted = predict_center(track.current_center, track.velocity, dt)
        motion_distance = distance(predicted, observation["center"])
        allowed_distance = self.config.strict_motion_gate_multiplier * (
            self.config.base_motion_gate + self.config.max_speed_px_per_sec * dt
        )
        return motion_distance <= allowed_distance

    def _recent_unmatched_track_is_plausible(self, observation: Dict) -> bool:
        for track in matchable_tracks(self.tracks):
            if track.last_unmatched_frame_index is None:
                continue

            frames_since_unmatched = self.frame_index - track.last_unmatched_frame_index
            if frames_since_unmatched > self.config.max_association_gap_frames:
                continue

            if self._strict_motion_gate(track, observation):
                return True

        return False

    def _record_pending_birth(self, observation: Dict) -> bool:
        closest_pending = None
        closest_distance = float("inf")

        for pending in self.pending_births:
            pending_distance = distance(pending["center"], observation["center"])
            if pending_distance < closest_distance:
                closest_distance = pending_distance
                closest_pending = pending

        if closest_pending is not None and closest_distance <= self.config.base_motion_gate:
            closest_pending["center"] = observation["center"]
            closest_pending["observation"] = observation
            closest_pending["last_seen_frame"] = self.frame_index
            closest_pending["count"] += 1
            return closest_pending["count"] >= self.config.unmatched_detection_buffer_frames

        self.pending_births.append({
            "center": observation["center"],
            "observation": observation,
            "first_seen_frame": self.frame_index,
            "last_seen_frame": self.frame_index,
            "count": 1,
        })
        return self.config.unmatched_detection_buffer_frames <= 1

    def _prune_pending_births(self) -> None:
        self.pending_births = [
            pending for pending in self.pending_births
            if self.frame_index - pending["last_seen_frame"] <= self.config.unmatched_detection_buffer_frames
        ]

    def update(
        self,
        observations_by_ts: Dict[float, List[Dict]],
        current_timestamp: float | None = None,
    ) -> Tuple[List[RuntimeTrackV2], Dict[str, str]]:
        assignment_map: Dict[str, str] = {}
        self.last_new_tracks_created = 0
        self.last_debug_report = self._empty_debug_report()

        timestamps = sorted(observations_by_ts.keys())
        if not timestamps:
            if current_timestamp is not None:
                timestamps = [float(current_timestamp)]
            elif self.last_update_timestamp is not None:
                timestamps = [self.last_update_timestamp + 1.0]
            else:
                timestamps = [0.0]

        for timestamp in timestamps:
            observations = observations_by_ts.get(timestamp, [])
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
                track.last_matched_frame_index = self.frame_index
                track.last_unmatched_frame_index = None
                track.hit_count += 1
                track.miss_count = 0
                track.detection_history.append(observation["detection_id"])
                track.center_history.append(track.current_center)
                promote_if_ready(track, self.config)

                matched_track_ids.add(track.runtime_track_id)
                assignment_map[observation["detection_id"]] = track.runtime_track_id
                self.last_debug_report["tracks_continued"] += 1

            self.last_debug_report["tracks_rejected_by_motion_gate"] += max(
                0,
                len(eligible_tracks) * len(observations) - len(gated_observation_indices),
            )

            close_stale_tracks(
                eligible_tracks,
                matched_track_ids,
                self.config,
                current_timestamp=float(timestamp),
                frame_index=self.frame_index,
            )

            for observation_index in sorted(unmatched_observation_indices):
                observation = observations[observation_index]

                if observation_index in gated_observation_indices:
                    self.last_debug_report["forced_continuations"] += 1
                    continue

                if self._recent_unmatched_track_is_plausible(observation):
                    self.last_debug_report["forced_continuations"] += 1
                    continue

                if not self._record_pending_birth(observation):
                    continue

                track = create_track(observation, frame_index=self.frame_index)
                promote_if_ready(track, self.config)
                self.tracks.append(track)
                self.last_new_tracks_created += 1
                self.last_debug_report["new_tracks_created"] += 1
                assignment_map[observation["detection_id"]] = track.runtime_track_id

            self._prune_pending_births()
            self.last_update_timestamp = float(timestamp)
            self.frame_index += 1

        return list(self.tracks), assignment_map
