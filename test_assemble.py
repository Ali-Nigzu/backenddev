"""Focused tests for final storage-row assembly."""

import copy
import unittest
from datetime import datetime

from assemble import Assemble


def event(track_id="1", timestamp=1.333, event_type=1):
    return {
        "track_id": track_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "best_crop": {"frame_id": "frame", "bbox": {}},
    }


def demographic(track_id="1", age=35, sex=1):
    return {"track_id": track_id, "age": age, "sex": sex}


def assemble(events, demographics, start="2026-08-15T11:34:55.000Z"):
    return Assemble()(
        {"events": events}, {"results": demographics}, start
    )


class AssembleTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(assemble([], []), {"rows": []})

    def test_exact_storage_schema_types_device_and_id_range(self):
        row = assemble([event()], [demographic()])["rows"][0]
        self.assertEqual(
            set(row),
            {"device_id", "event_id", "event", "timestamp", "sex", "age_bucket"},
        )
        for field in ("device_id", "event_id", "event", "sex", "age_bucket"):
            self.assertIs(type(row[field]), int)
        self.assertIs(type(row["timestamp"]), str)
        self.assertEqual(row["device_id"], 0)
        self.assertLessEqual(1, row["event_id"])
        self.assertLessEqual(row["event_id"], 2**63 - 1)
        self.assertNotIn("ts", row)
        self.assertNotIn("org_id", row)
        self.assertNotIn("site_id", row)

    def test_deterministic(self):
        events = [event(), event("2", 2.0, 0)]
        demographics = [demographic(), demographic("2", 19, 0)]
        first = assemble(copy.deepcopy(events), copy.deepcopy(demographics))
        second = assemble(copy.deepcopy(events), copy.deepcopy(demographics))
        self.assertEqual(first, second)

    def test_unrelated_earlier_event_does_not_change_ids(self):
        events = [event("a", 1.0), event("b", 2.0)]
        demographics = [demographic("a"), demographic("b")]
        original = assemble(events, demographics)["rows"]
        prefixed = assemble(
            [event("x", 0.5, 0), *events],
            [demographic("x", 18, 0), *demographics],
        )["rows"]
        self.assertEqual(
            [row["event_id"] for row in original],
            [row["event_id"] for row in prefixed[1:]],
        )

    def test_different_tracks_distinguish_same_final_properties(self):
        rows = assemble(
            [event("a"), event("b")],
            [demographic("a"), demographic("b")],
        )["rows"]
        self.assertNotEqual(rows[0]["event_id"], rows[1]["event_id"])

    def test_exact_duplicate_identities_get_distinct_deterministic_ids(self):
        events = [event(), event()]
        first = assemble(events, [demographic()])
        second = assemble(copy.deepcopy(events), [demographic()])
        self.assertNotEqual(
            first["rows"][0]["event_id"], first["rows"][1]["event_id"]
        )
        self.assertEqual(first, second)

    def test_relative_time_becomes_canonical_utc_during_bst(self):
        row = assemble(
            [event(timestamp=1.333)],
            [demographic()],
            "2026-08-15T11:34:55.000Z",
        )["rows"][0]
        self.assertEqual(row["timestamp"], "2026-08-15T11:34:56.333Z")
        parsed = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        self.assertIsNotNone(parsed.tzinfo)


if __name__ == "__main__":
    unittest.main()
