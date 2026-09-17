import copy
import importlib
import unittest
from datetime import datetime, timezone

from snapshot import organisation_engine
from snapshot import site_engine


orchestration = importlib.import_module("snapshot.snapshot")


def instant(hour, minute=0):
    return datetime(2026, 9, 17, hour, minute, tzinfo=timezone.utc)


def stamp(hour, minute=0):
    return site_engine.stamp(instant(hour, minute))


def make_site(site_id=2, enabled=False, created_at=None, capacity=100):
    return {
        "id": site_id,
        "name": f"Site {site_id}",
        "organisation_id": 1,
        "max_capacity": capacity,
        "enabled": enabled,
        "created_at": created_at or stamp(10),
    }


def make_device(device_id=3, site_id=2, enabled=False, analyzed_until=None,
                created_at=None):
    return {
        "id": device_id,
        "name": f"Device {device_id}",
        "site_id": site_id,
        "enabled": enabled,
        "analyzed_until": analyzed_until or stamp(13),
        "created_at": created_at or stamp(10),
    }


def make_event(event_id, device_id=3, site_id=2, at=None, event_type=1,
               age=0, sex=0):
    return {
        "organisation_id": 1,
        "site_id": site_id,
        "device_id": device_id,
        "event_id": str(event_id),
        "event": event_type,
        "timestamp": at or stamp(12),
        "sex": sex,
        "age_bucket": age,
    }


def compute_site(site, devices, events=None, now=None):
    now = now or instant(14)
    snapshot = {"ts": site["created_at"], "state": {}}
    return site_engine.compute_site(
        site, devices, snapshot, "REBUILD", None, events or [], now)


def compute_organisation(sites, devices_by_site, events=None, now=None,
                         enabled=False):
    now = now or instant(14)
    horizons = {}
    for site in sites:
        value = site_engine.site_horizons(site, devices_by_site[site["id"]], now)
        horizons[site["id"]] = None if value is None else {"stable_until": value[1]}
    return organisation_engine.compute(
        sites, devices_by_site, horizons, None,
        min(site["created_at"] for site in sites), events or [], "REBUILD",
        now, enabled)


def assert_integer_payload(test, value):
    if isinstance(value, dict):
        for child in value.values():
            assert_integer_payload(test, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_integer_payload(test, child)
    elif isinstance(value, float):
        test.fail(f"float found in payload: {value!r}")


class SnapshotRepairTests(unittest.TestCase):
    def test_01_disabled_device_advancement_generates_range(self):
        site = make_site()
        device = make_device(analyzed_until=stamp(13))
        context = {
            "organisation": {"id": 1, "enabled": False},
            "sites": [site],
            "devices_by_site": {2: [device]},
        }
        previous = {2: {"device_watermarks": {
            "3": {"consumed_until": stamp(11)}
        }}}
        ranges = orchestration._ranges(
            context, {2: "INCREMENTAL"}, previous, "TIME_ONLY", instant(14))
        self.assertEqual(1, len(ranges))
        self.assertEqual((instant(11), instant(13)), (ranges[0].start, ranges[0].end))

    def test_02_disabled_device_events_contribute(self):
        site = make_site(enabled=True)
        device = make_device(enabled=False)
        _, payload, _ = compute_site(site, [device], [make_event("entry")])
        self.assertEqual(1, sum(payload["entrances_96"]))

    def test_03_disabled_site_does_not_suppress_events(self):
        site = make_site(enabled=False)
        device = make_device(enabled=True)
        _, payload, _ = compute_site(site, [device], [make_event("entry")])
        self.assertEqual(1, sum(payload["entrances_96"]))

    def test_04_disabled_organisation_does_not_suppress_events(self):
        site = make_site(enabled=False)
        device = make_device(enabled=False)
        _, payload, state = compute_organisation(
            [site], {2: [device]}, [make_event("entry")], enabled=False)
        self.assertEqual(1, sum(payload["entrances_96"]))
        self.assertFalse(state["metadata"]["organisation_enabled"])

    def test_05_all_devices_disabled_still_have_horizon(self):
        site = make_site(enabled=False)
        devices = [make_device(3, enabled=False), make_device(4, enabled=False)]
        latest, stable = site_engine.site_horizons(site, devices, instant(14))
        self.assertEqual(instant(13), latest)
        self.assertEqual(instant(13), stable)

    def test_06_no_horizon_movement_has_no_range(self):
        site = make_site()
        device = make_device(analyzed_until=stamp(11))
        context = {
            "organisation": {"id": 1, "enabled": False},
            "sites": [site], "devices_by_site": {2: [device]},
        }
        previous = {2: {"device_watermarks": {
            "3": {"consumed_until": stamp(11)}
        }}}
        self.assertEqual([], orchestration._ranges(
            context, {2: "TIME_ONLY"}, previous, "TIME_ONLY", instant(14)))

    def test_07_first_run_uses_creation_to_horizon(self):
        site = make_site(created_at=stamp(10))
        device = make_device(created_at=stamp(10, 30), analyzed_until=stamp(13))
        context = {
            "organisation": {"id": 1, "enabled": False},
            "sites": [site], "devices_by_site": {2: [device]},
        }
        ranges = orchestration._ranges(
            context, {2: "REBUILD"}, {2: None}, "TIME_ONLY", instant(14))
        self.assertEqual((instant(10, 30), instant(13)),
                         (ranges[0].start, ranges[0].end))

    def test_08_future_horizon_is_capped_and_resumes(self):
        site = make_site()
        device = make_device(analyzed_until=stamp(15))
        context = {
            "organisation": {"id": 1, "enabled": False},
            "sites": [site], "devices_by_site": {2: [device]},
        }
        previous = {2: {"device_watermarks": {
            "3": {"consumed_until": stamp(13)}
        }}}
        first = orchestration._ranges(
            context, {2: "INCREMENTAL"}, previous, "TIME_ONLY", instant(14))[0]
        previous[2]["device_watermarks"]["3"]["consumed_until"] = stamp(14)
        second = orchestration._ranges(
            context, {2: "INCREMENTAL"}, previous, "TIME_ONLY",
            instant(14, 30))[0]
        self.assertEqual((instant(13), instant(14)), (first.start, first.end))
        self.assertEqual((instant(14), instant(14, 30)), (second.start, second.end))

    def test_09_machine_and_timestamp_advance_beyond_source_horizon(self):
        site = make_site()
        device = make_device(analyzed_until=stamp(13))
        result_ts, _, state = compute_site(site, [device], now=instant(14))
        self.assertEqual(instant(14), result_ts)
        self.assertEqual(stamp(14), state["current_machine"]["cursor_ts"])
        self.assertEqual(stamp(13), state["stable_until"])

    def test_10_safe_v1_state_is_reused(self):
        site = make_site()
        device = make_device(analyzed_until=stamp(13))
        _, _, state = compute_site(site, [device], now=instant(14))
        classification, previous = site_engine.classify_site(
            site, [device], {"ts": stamp(14), "state": state}, instant(15))
        self.assertEqual("TIME_ONLY", classification)
        self.assertIs(state, previous)
        self.assertEqual(1, state["version"])

    def test_11_unsafe_checkpoint_rebuilds_only_affected_site(self):
        bad_site = make_site(2)
        safe_site = make_site(1)
        bad_device = make_device(3, 2, analyzed_until=stamp(11))
        safe_device = make_device(1, 1, analyzed_until=stamp(13))
        _, _, bad_state = compute_site(bad_site, [bad_device], now=instant(12))
        bad_state["stable_until"] = stamp(12)
        classification, _ = site_engine.classify_site(
            bad_site, [bad_device], {"ts": stamp(12), "state": bad_state}, instant(14))
        self.assertEqual("REBUILD", classification)
        context = {
            "organisation": {"id": 1, "enabled": False},
            "sites": [safe_site, bad_site],
            "devices_by_site": {1: [safe_device], 2: [bad_device]},
        }
        ranges = orchestration._ranges(
            context, {1: "NO_OP", 2: "REBUILD"}, {1: {}, 2: bad_state},
            "TIME_ONLY", instant(14))
        self.assertEqual({2}, {value.site_id for value in ranges})
        _, _, org_state = compute_organisation(
            [bad_site], {2: [bad_device]}, now=instant(12))
        org_state["stable_until"] = stamp(12)
        org_context = {
            "organisation": {"id": 1, "enabled": False},
            "sites": [bad_site],
            "devices_by_site": {2: [bad_device]},
            "organisation_snapshot": {"ts": stamp(12), "state": org_state},
        }
        org_classification, _ = orchestration._org_classification(
            org_context, {2: {"stable_until": instant(11)}}, instant(14))
        self.assertEqual("REBUILD", org_classification)

    def test_12_production_shaped_disabled_site_consumes_three_devices(self):
        site = make_site(2, enabled=False)
        old_devices = [make_device(value, 2, False, stamp(11)) for value in (3, 4, 5)]
        _, _, old_state = compute_site(site, old_devices, now=instant(12))
        devices = [make_device(value, 2, False, stamp(13)) for value in (3, 4, 5)]
        classification, previous = site_engine.classify_site(
            site, devices, {"ts": stamp(12), "state": old_state}, instant(14))
        self.assertEqual("INCREMENTAL", classification)
        events = [make_event(f"entry-{value}", value, 2, stamp(12, value))
                  for value in (3, 4, 5)]
        _, payload, _ = site_engine.compute_site(
            site, devices, {"ts": stamp(12), "state": old_state},
            classification, previous, events, instant(14))
        self.assertEqual(3, sum(payload["entrances_96"]))

    def test_13_occupancy_96_is_96_integer_triples(self):
        site = make_site()
        _, payload, _ = compute_site(site, [make_device()])
        self.assertEqual(96, len(payload["occupancy_96"]))
        self.assertTrue(all(len(value) == 3 for value in payload["occupancy_96"]))
        self.assertTrue(all(type(number) is int for value in payload["occupancy_96"]
                            for number in value))

    def test_14_occupancy_average_excludes_zero_time(self):
        block = {
            "occupancy_area_person_seconds": [1800],
            "occupancy_seconds": [300],
            "occupancy_min_positive": [6],
            "occupancy_max": [6],
        }
        self.assertEqual([[6, 6, 6]], site_engine._occupancy(block))

    def test_15_five_minutes_at_six_emits_six_not_two(self):
        site = make_site()
        machine = site_engine.new_machine(instant(10), site)
        index = 95
        machine["q15"]["occupancy_area_person_seconds"][index] = 6 * 5 * 60
        machine["q15"]["occupancy_seconds"][index] = 5 * 60
        machine["q15"]["occupancy_min_positive"][index] = 6
        machine["q15"]["occupancy_max"][index] = 6
        payload = site_engine.derive_payload(
            machine, site, [make_device()],
            {"device_watermarks": {"3": {"name": "Device 3"}}})
        self.assertEqual([6, 6, 6], payload["occupancy_96"][index])

    def test_16_empty_occupancy_bucket_is_zero_triple(self):
        site = make_site()
        _, payload, _ = compute_site(site, [make_device()])
        self.assertIn([0, 0, 0], payload["occupancy_96"])

    def test_17_named_period_uses_positive_occupancy_average(self):
        site = make_site()
        machine = site_engine.new_machine(instant(10), site)
        machine["today"]["occupancy_area_person_seconds"][9] = 1800
        machine["today"]["occupancy_seconds"][9] = 300
        machine["today"]["occupancy_min_positive"][9] = 6
        machine["today"]["occupancy_max"][9] = 6
        payload = site_engine.derive_payload(
            machine, site, [make_device()],
            {"device_watermarks": {"3": {"name": "Device 3"}}})
        self.assertEqual([6, 6, 6], payload["today"]["occupancy"][9])

    def test_18_round_half_up_is_deterministic(self):
        self.assertEqual(3, site_engine._round_half_up(5, 2))
        self.assertEqual(2, site_engine._round_half_up(4, 2))
        self.assertEqual(2, site_engine._round_half_up(3, 2))

    def test_19_dwell_is_integer_minutes(self):
        site = make_site()
        machine = site_engine.new_machine(instant(10), site)
        cases = {
            90: (1629, 1, 27),
            91: (1649, 1, 27),
            92: (1650, 1, 28),
            93: (3300, 2, 28),
            # 148 / 5 = 29.6 seconds: direct minute rounding is zero. Rounding
            # seconds first would produce 30 seconds and incorrectly emit one.
            94: (148, 5, 0),
        }
        for index, (total, count, _) in cases.items():
            machine["q15"]["dwell_sum_seconds"][index] = total
            machine["q15"]["dwell_count"][index] = count
        payload = site_engine.derive_payload(
            machine, site, [make_device()],
            {"device_watermarks": {"3": {"name": "Device 3"}}})
        self.assertEqual(96, len(payload["dwell_time_96"]))
        self.assertEqual(0, payload["dwell_time_96"][95])
        for index, (_, _, expected) in cases.items():
            self.assertEqual(expected, payload["dwell_time_96"][index])
        self.assertTrue(all(type(value) is int for value in payload["dwell_time_96"]))

    def test_20_capacity_is_integer_pair_from_precise_state(self):
        site = make_site(capacity=2)
        machine = site_engine.new_machine(instant(10), site)
        machine["q15"]["occupancy_area_person_seconds"][95] = 3
        machine["q15"]["occupancy_seconds"][95] = 2
        machine["q15"]["rolling_peak_occupancy"][95] = 1
        payload = site_engine.derive_payload(
            machine, site, [make_device()],
            {"device_watermarks": {"3": {"name": "Device 3"}}})
        self.assertEqual([75, 50], payload["capacity"][95])

    def test_21_age_percentages_are_integer_and_sum_to_100(self):
        values = site_engine._pct([1, 1, 1, 0, 0, 0])
        self.assertEqual([34, 33, 33, 0, 0, 0], values)
        self.assertEqual(100, sum(values))

    def test_22_sex_percentages_are_integer_and_sum_to_100(self):
        values = site_engine._pct([1, 1])
        self.assertEqual([50, 50], values)
        self.assertTrue(all(type(value) is int for value in values))

    def test_23_traffic_percentages_sum_to_100(self):
        site = make_site()
        devices = [make_device(3), make_device(4)]
        machine = site_engine.new_machine(instant(10), site)
        machine["q15"]["traffic_counts"][95] = {"3": 1, "4": 2}
        state = {"device_watermarks": {
            "3": {"name": "Device 3"}, "4": {"name": "Device 4"}}}
        payload = site_engine.derive_payload(machine, site, devices, state)
        self.assertEqual([33, 67], payload["traffic_split_96"][95])
        self.assertEqual(100, sum(payload["traffic_split_96"][95]))

    def test_24_empty_percentages_are_integer_zeros(self):
        self.assertEqual([0, 0, 0], site_engine._pct([0, 0, 0]))

    def test_25_site_payload_contains_no_float(self):
        site = make_site(enabled=False)
        events = [make_event("in", at=stamp(12)),
                  make_event("out", at=stamp(12, 5), event_type=0)]
        _, payload, _ = compute_site(site, [make_device(enabled=False)], events)
        self.assertEqual([5], [value for value in payload["dwell_time_96"] if value])
        assert_integer_payload(self, payload)

    def test_26_organisation_payload_contains_no_float(self):
        site = make_site(enabled=False)
        events = [make_event("in", at=stamp(12)),
                  make_event("out", at=stamp(12, 5), event_type=0)]
        _, payload, _ = compute_organisation(
            [site], {2: [make_device(enabled=False)]}, events, enabled=False)
        self.assertEqual([5], [value for value in payload["dwell_time_96"] if value])
        assert_integer_payload(self, payload)

    def test_27_repeated_incremental_calculation_does_not_double_count(self):
        site = make_site()
        first_device = make_device(analyzed_until=stamp(11))
        first_event = make_event("first", at=stamp(10, 30))
        _, first_payload, state = compute_site(
            site, [first_device], [first_event], now=instant(12))
        second_device = make_device(analyzed_until=stamp(13))
        second_event = make_event("second", at=stamp(12, 30))
        classification, previous = site_engine.classify_site(
            site, [second_device], {"ts": stamp(12), "state": state}, instant(14))
        _, second_payload, second_state = site_engine.compute_site(
            site, [second_device], {"ts": stamp(12), "state": state},
            classification, previous, [second_event], instant(14))
        classification, previous = site_engine.classify_site(
            site, [second_device], {"ts": stamp(14), "state": second_state}, instant(15))
        _, third_payload, _ = site_engine.compute_site(
            site, [second_device], {"ts": stamp(14), "state": second_state},
            classification, previous, [], instant(15))
        self.assertEqual(1, sum(first_payload["entrances_96"]))
        self.assertEqual(2, sum(second_payload["entrances_96"]))
        self.assertEqual(2, sum(third_payload["entrances_96"]))

    def test_28_site_and_organisation_aggregation_remain_correct(self):
        sites = [make_site(1, enabled=False), make_site(2, enabled=False)]
        devices_by_site = {
            1: [make_device(1, 1, False)],
            2: [make_device(2, 2, False)],
        }
        events = [make_event("one", 1, 1, stamp(12)),
                  make_event("two", 2, 2, stamp(12, 1))]
        _, payload, _ = compute_organisation(
            sites, devices_by_site, events, enabled=False)
        self.assertEqual(2, sum(payload["entrances_96"]))
        nonempty = [row for row in payload["traffic_split_96"] if sum(row)]
        self.assertEqual([[50, 50]], nonempty)
        self.assertEqual([1, 2], [axis["site_id"] for axis in payload["traffic_devices"]])


if __name__ == "__main__":
    unittest.main()

