"""Focused synthetic tests for the stateless tracking stage."""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

from events import Event
from track import Track
from track import track as track_module


def detection(
    frame_id, detection_id, x, y=50.0, confidence=0.9, width=20.0, height=40.0
):
    return {
        "detection_id": detection_id,
        "bbox": {
            "x1": x - width / 2.0,
            "y1": y - height / 2.0,
            "x2": x + width / 2.0,
            "y2": y + height / 2.0,
        },
        "centre": {"x": x, "y": y},
        "confidence": confidence,
    }


def frame(index, timestamp, detections=None):
    frame_id = f"frame-{index:02d}"
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "detections": detections or [],
    }


def batch(points, confidence=0.9):
    frames = []
    for index, (timestamp, x) in enumerate(points):
        frame_id = f"frame-{index:02d}"
        frames.append(
            frame(
                index,
                timestamp,
                [detection(frame_id, f"det-{index}", x, confidence=confidence)],
            )
        )
    return {"detections": frames}


class TrackTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(Track()({"detections": []}), {"tracks": []})

    def test_determinism_and_canonical_frame_order(self):
        detection_batch = batch([(0.0, 10.0), (0.3, 13.0), (0.7, 17.0)])
        detection_batch["detections"].reverse()
        first = Track()(copy.deepcopy(detection_batch))
        second = Track()(copy.deepcopy(detection_batch))
        self.assertEqual(first, second)
        self.assertEqual(
            [point["timestamp"] for point in first["tracks"][0]["path"]],
            [0.0, 0.3, 0.7],
        )

    def test_single_pedestrian_and_bootstrap_activation(self):
        result = Track()(batch([(0.0, 10.0), (0.3, 13.0)]))
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(result["tracks"][0]["track_id"], "1")
        self.assertEqual(len(result["tracks"][0]["path"]), 2)

    def test_later_birth_requires_normal_activation_hits(self):
        frames = [frame(index, index * 0.1) for index in range(10)]
        for index, x in enumerate((20.0, 22.0), start=10):
            frame_id = f"frame-{index:02d}"
            frames.append(
                frame(index, index * 0.1, [detection(frame_id, f"det-{index}", x)])
            )
        self.assertEqual(Track()({"detections": frames}), {"tracks": []})
        frame_id = "frame-12"
        frames.append(frame(12, 1.2, [detection(frame_id, "det-12", 24.0)]))
        self.assertEqual(len(Track()({"detections": frames})["tracks"]), 1)

    def test_tentative_expiry_is_not_returned(self):
        frames = [frame(0, 0.0, [detection("frame-00", "det-0", 10.0)])]
        frames.append(frame(1, 1.1))
        self.assertEqual(Track()({"detections": frames}), {"tracks": []})

    def test_low_confidence_recovers_active_track(self):
        frames = batch([(0.0, 10.0), (0.3, 13.0)])["detections"]
        frames.append(
            frame(2, 0.6, [detection("frame-02", "det-2", 16.0, confidence=0.3)])
        )
        result = Track()({"detections": frames})
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"][0]["path"]), 3)

    def test_low_confidence_detection_cannot_create_track(self):
        only_low = frame(0, 0.0, [detection("frame-00", "det-0", 10.0, confidence=0.3)])
        self.assertEqual(Track()({"detections": [only_low]}), {"tracks": []})

    def test_short_disappearance_recovers_same_track(self):
        frames = batch([(0.0, 10.0), (0.3, 13.0)])["detections"]
        frames.extend(
            [frame(2, 0.8), frame(3, 1.3, [detection("frame-03", "det-3", 23.0)])]
        )
        result = Track()({"detections": frames})
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1"])
        self.assertEqual(len(result["tracks"][0]["path"]), 3)

    def test_long_disappearance_closes_old_track_and_births_new_id(self):
        frames = batch([(0.0, 10.0), (0.3, 13.0)])["detections"]
        frames.append(frame(2, 2.31, [detection("frame-02", "det-2", 40.0)]))
        frames.append(frame(3, 2.61, [detection("frame-03", "det-3", 43.0)]))
        result = Track()({"detections": frames})
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1", "2"])
        self.assertEqual([len(track["path"]) for track in result["tracks"]], [2, 2])

    def test_crossing_pedestrians_keep_trajectory_direction(self):
        frames = []
        for index, (left_to_right, right_to_left) in enumerate(
            [(20.0, 80.0), (30.0, 70.0), (42.0, 58.0), (55.0, 45.0), (68.0, 32.0)]
        ):
            frame_id = f"frame-{index:02d}"
            frames.append(
                frame(
                    index,
                    index * 0.3,
                    [
                        detection(frame_id, f"a-{index}", left_to_right),
                        detection(frame_id, f"b-{index}", right_to_left),
                    ],
                )
            )
        result = Track()({"detections": frames})
        self.assertEqual(len(result["tracks"]), 2)
        paths = [
            [point["centre"]["x"] for point in track["path"]]
            for track in result["tracks"]
        ]
        self.assertTrue(
            all(later > earlier for earlier, later in zip(paths[0], paths[0][1:]))
        )
        self.assertTrue(
            all(later < earlier for earlier, later in zip(paths[1], paths[1][1:]))
        )

    def test_irregular_intervals_and_zero_dt(self):
        result = Track()(
            batch([(0.0, 10.0), (0.31, 13.1), (0.70, 17.0), (1.08, 20.8), (1.62, 26.2)])
        )
        self.assertEqual(len(result["tracks"]), 1)
        zero_dt = Track()(batch([(0.0, 10.0), (0.0, 10.0)]))
        self.assertEqual(len(zero_dt["tracks"]), 1)

    def test_best_crop_uses_earliest_highest_confidence(self):
        frames = []
        for index, confidence in enumerate((0.8, 0.9, 0.9)):
            frame_id = f"frame-{index:02d}"
            frames.append(
                frame(
                    index,
                    index * 0.3,
                    [
                        detection(
                            frame_id,
                            f"det-{index}",
                            10 + index * 2,
                            confidence=confidence,
                        )
                    ],
                )
            )
        track = Track()({"detections": frames})["tracks"][0]
        self.assertEqual(track["best_crop"]["frame_id"], "frame-01")
        self.assertEqual(track["best_crop_confidence"], 0.9)

    def test_event_accepts_track_batch_without_adapter(self):
        result = Track()(batch([(index * 0.2, 50.0) for index in range(6)]))
        event_batch = Event()(
            result,
            {"point_a": {"x": 0.0, "y": 50.0}, "point_b": {"x": 100.0, "y": 50.0}},
        )
        self.assertEqual(event_batch, {"events": []})

    def test_invalid_timestamps_fail_clearly(self):
        for timestamp in (math.nan, math.inf, -math.inf):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                Track()({"detections": [frame(0, timestamp)]})

    def test_math_stays_finite_positive_and_gated_pairs_are_rejected(self):
        source = detection("frame-00", "det-0", 10.0)
        state, covariance = track_module._new_kalman_state(source)
        internal = {
            "state": state,
            "covariance": covariance,
            "last_prediction_timestamp": 0.0,
        }
        track_module._predict(internal, 1.7)
        self.assertGreater(internal["state"][2], 0.0)
        self.assertGreater(internal["state"][3], 0.0)
        self.assertTrue(np.isfinite(internal["covariance"]).all())
        _residual, innovation_covariance, distance = track_module._innovation(
            internal, source
        )
        self.assertTrue(np.isfinite(innovation_covariance).all())
        self.assertTrue(math.isfinite(distance))
        far = detection("frame-01", "det-far", 100_000.0)
        matches, unmatched_tracks, unmatched_detections = track_module._associate(
            [internal], [far], 1.0
        )
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_detections, [0])

    def test_refinement_disabled_preserves_pass_one_fragments(self):
        frames = batch(
            [(0.0, 10.0), (0.1, 12.0), (0.4, 18.0), (0.5, 20.0), (0.6, 22.0)]
        )["detections"]
        with (
            patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15),
            patch.object(track_module, "REFINE_ENABLED", False),
        ):
            result = Track()({"detections": frames})
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1", "2"])
        self.assertEqual([len(track["path"]) for track in result["tracks"]], [2, 3])

    def test_refinement_merges_short_active_fragmentation(self):
        frames = batch(
            [(0.0, 10.0), (0.1, 12.0), (0.4, 18.0), (0.5, 20.0), (0.6, 22.0)]
        )["detections"]
        with patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15):
            result = Track()({"detections": frames})
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1"])
        self.assertEqual(
            [point["centre"]["x"] for point in result["tracks"][0]["path"]],
            [10.0, 12.0, 18.0, 20.0, 22.0],
        )

    def test_refinement_absorbs_tentative_bridge(self):
        points = [
            (0.0, 10.0),
            (0.1, 12.0),
            (0.4, 18.0),
            (0.5, 20.0),
            (0.8, 26.0),
            (0.9, 28.0),
            (1.0, 30.0),
        ]
        frames = batch(points)["detections"]
        with (
            patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15),
            patch.object(track_module, "TENTATIVE_TIMEOUT_SECONDS", 0.15),
            patch.object(track_module, "BOOTSTRAP_WINDOW_FRAMES", 2),
        ):
            result = Track()({"detections": frames})
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1"])
        self.assertEqual(len(result["tracks"][0]["path"]), len(points))

    def test_refinement_absorbs_tentative_tail(self):
        points = [(0.0, 10.0), (0.1, 12.0), (0.4, 18.0), (0.5, 20.0)]
        with (
            patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15),
            patch.object(track_module, "BOOTSTRAP_WINDOW_FRAMES", 2),
        ):
            result = Track()(batch(points))
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"][0]["path"]), 4)

    def test_tentative_only_fragments_cannot_create_public_track(self):
        points = [(0.0, 10.0), (0.1, 12.0), (0.4, 18.0), (0.5, 20.0)]
        with (
            patch.object(track_module, "BOOTSTRAP_WINDOW_FRAMES", 0),
            patch.object(track_module, "TENTATIVE_TIMEOUT_SECONDS", 0.15),
        ):
            self.assertEqual(Track()(batch(points)), {"tracks": []})

    def test_refinement_never_links_beyond_hard_horizon(self):
        points = [(0.0, 10.0), (0.1, 12.0), (1.61, 42.2), (1.71, 44.2), (1.81, 46.2)]
        with patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15):
            result = Track()(batch(points))
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1", "2"])

    def test_refinement_rejects_physically_impossible_short_gap(self):
        points = [(0.0, 10.0), (0.1, 12.0), (0.4, 1000.0), (0.5, 1002.0), (0.6, 1004.0)]
        with patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15):
            result = Track()(batch(points))
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1", "2"])

    def test_refinement_allows_a_smooth_turn(self):
        samples = [
            (0.0, 10.0, 50.0),
            (0.1, 12.0, 50.0),
            (0.4, 18.0, 51.0),
            (0.5, 20.0, 52.0),
            (0.6, 22.0, 54.0),
        ]
        frames = []
        for index, (timestamp, x, y) in enumerate(samples):
            frame_id = f"frame-{index:02d}"
            frames.append(
                frame(index, timestamp, [detection(frame_id, f"det-{index}", x, y=y)])
            )
        with patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15):
            result = Track()({"detections": frames})
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"][0]["path"]), len(samples))

    def test_refinement_rejects_implausible_box_scale_change(self):
        frames = []
        for index, (timestamp, x, width, height) in enumerate(
            [
                (0.0, 10.0, 20.0, 40.0),
                (0.1, 12.0, 20.0, 40.0),
                (0.4, 18.0, 60.0, 120.0),
                (0.5, 20.0, 60.0, 120.0),
                (0.6, 22.0, 60.0, 120.0),
            ]
        ):
            frame_id = f"frame-{index:02d}"
            frames.append(
                frame(
                    index,
                    timestamp,
                    [
                        detection(
                            frame_id, f"det-{index}", x, width=width, height=height
                        )
                    ],
                )
            )
        with patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15):
            result = Track()({"detections": frames})
        self.assertEqual([track["track_id"] for track in result["tracks"]], ["1", "2"])

    def test_refinement_global_assignment_is_not_greedy(self):
        summaries = [{}, {}, {}, {}]
        candidate_costs = {(0, 2): 0.20, (0, 3): 0.10, (1, 2): 0.10}
        with patch.object(
            track_module, "_candidate_costs", return_value=candidate_costs
        ):
            self.assertEqual(
                track_module._assign_fragment_links(summaries), {0: 3, 1: 2}
            )

    def test_merged_crop_and_path_are_real_and_deterministic(self):
        frames = []
        for index, (timestamp, x, confidence) in enumerate(
            [
                (0.0, 10.0, 0.8),
                (0.1, 12.0, 0.8),
                (0.4, 18.0, 0.95),
                (0.5, 20.0, 0.95),
                (0.6, 22.0, 0.9),
            ]
        ):
            frame_id = f"frame-{index:02d}"
            frames.append(
                frame(
                    index,
                    timestamp,
                    [detection(frame_id, f"det-{index}", x, confidence=confidence)],
                )
            )
        with patch.object(track_module, "ACTIVE_TIMEOUT_SECONDS", 0.15):
            first = Track()({"detections": copy.deepcopy(frames)})
            second = Track()({"detections": copy.deepcopy(frames)})
        self.assertEqual(first, second)
        track = first["tracks"][0]
        self.assertEqual(track["best_crop"]["frame_id"], "frame-02")
        real_points = {
            (
                float(item["timestamp"]),
                float(item["detections"][0]["centre"]["x"]),
                float(item["detections"][0]["centre"]["y"]),
            )
            for item in frames
        }
        self.assertTrue(
            all(
                (point["timestamp"], point["centre"]["x"], point["centre"]["y"])
                in real_points
                for point in track["path"]
            )
        )


if __name__ == "__main__":
    unittest.main()
