from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import importlib

import pytest

from snapshot import organisation_engine as org
from snapshot import site_engine as site
from snapshot.models import SourceRange
from snapshot.source import coalesce_ranges

UTC = timezone.utc


def dt(hour, minute=0, second=0, day=1):
    return datetime(2026, 1, day, hour, minute, second, tzinfo=UTC)


def site_record(site_id=1, capacity=10):
    return {"id": site_id, "name": str(site_id), "max_capacity": capacity,
            "created_at": site.stamp(dt(0)), "destination": "p.d.t",
            "bigquery_destination": "p.d.t"}


def device(device_id=1, site_id=1, horizon=None, name="camera", status="enabled"):
    return {"id": device_id, "site_id": site_id, "name": name, "status": status,
            "created_at": site.stamp(dt(0)), "analyzed_until": None if horizon is None else site.stamp(horizon),
            "analysis_config": {}}


def event(at, direction, site_id=1, device_id=1, event_id=1):
    return {"destination": "p.d.t", "site_id": site_id, "device_id": device_id,
            "event_id": event_id, "event": direction, "timestamp": site.stamp(at),
            "sex": 0, "age_bucket": 2}


@pytest.mark.parametrize(("value", "expected"), [
    (dt(12), dt(12)), (dt(12, 14, 59), dt(12)), (dt(12, 15), dt(12, 15)),
    (dt(12, 29, 59), dt(12, 15)), (dt(12, 48, 44), dt(12, 45)),
])
def test_floor_to_q15(value, expected):
    assert site.floor_to_q15(value) == expected


def test_q15_exact_boundary_without_event_and_many_rolls():
    machine = site.new_machine(dt(0, 7), {})
    site.advance(machine, dt(12, 15), {})
    assert site.validate_q15(machine)
    assert site.parse_ts(machine["q15"]["window_start"]) + 95 * site.Q15 == dt(12, 15)
    assert all(len(machine["q15"][key]) == 96 for key in site.Q15_ARRAYS)


def test_partial_bucket_does_not_integrate_future():
    machine = site.new_machine(dt(12), {})
    machine["occupancy"] = 1
    machine["entry_fifo"] = [site.stamp(dt(12))]
    site.advance(machine, dt(12, 7), {})
    assert machine["q15"]["occupancy_seconds"][-1] == 7 * 60


def test_null_device_horizon_uses_created_at():
    value = device()
    value["created_at"] = site.stamp(dt(9))
    other = device(2, horizon=dt(12))
    assert site.site_horizons(site_record(), [value, other]) == (dt(12), dt(9))


def test_composite_identity_and_conflict():
    devices = [device(1, horizon=dt(13)), device(2, horizon=dt(13))]
    first = event(dt(12), 1, device_id=1, event_id=7)
    second = event(dt(12), 1, device_id=2, event_id=7)
    assert len(site.merge_events([first, first, second], devices, dt(0), dt(13))) == 2
    conflicting = dict(first, sex=1)
    with pytest.raises(ValueError, match="Conflicting"):
        site.merge_events([first, conflicting], devices, dt(0), dt(13))


def test_site_fifo_exact_expiry_and_late_exit():
    record = site_record()
    machine = site.new_machine(dt(8), record)
    site.apply_event(machine, event(dt(8), 1), record)
    site.apply_event(machine, event(dt(12), 0, event_id=2), record)
    assert machine["occupancy"] == 0
    assert machine["q15"]["dwell_count"][-1] == 1
    machine = site.new_machine(dt(8), record)
    site.apply_event(machine, event(dt(8), 1), record)
    site.apply_event(machine, event(dt(12, 0, 1), 0, event_id=2), record)
    assert machine["occupancy"] == 0
    assert sum(machine["q15"]["dwell_count"]) == 0


def test_stable_boundary_preserves_exact_expiry_for_exit():
    machine = site.new_machine(dt(8), {})
    runtimes = {"1": {"occupancy": 0, "entry_fifo": []}}
    org.apply_event(machine, runtimes, event(dt(8), 1))
    org.advance(machine, runtimes, dt(12), include_target_expiry=False)
    assert runtimes["1"]["entry_fifo"] == [site.stamp(dt(8))]
    current, current_runtime = org.current_from(machine, runtimes, [event(dt(12), 0, event_id=2)], dt(12))
    assert current_runtime["1"]["occupancy"] == 0
    assert current["q15"]["dwell_count"][-1] == 1


def _org_timeline(events, target):
    machine = site.new_machine(dt(9), {})
    runtimes = {"1": {"occupancy": 0, "entry_fifo": []}, "2": {"occupancy": 0, "entry_fifo": []}}
    current, runtime = org.current_from(machine, runtimes, sorted(events, key=site.event_order), target)
    return current, runtime


@pytest.mark.parametrize(("events", "target", "area", "seconds", "average", "maximum"), [
    ([event(dt(10), 1, 1, 1, 1), event(dt(10), 1, 2, 2, 2), event(dt(11), 0, 1, 1, 3), event(dt(11), 0, 2, 2, 4)], dt(11), 7200, 3600, 2, 2),
    ([event(dt(10), 1, 1, 1, 1), event(dt(10, 30), 0, 1, 1, 2), event(dt(10, 30), 1, 2, 2, 3), event(dt(11), 0, 2, 2, 4)], dt(11), 3600, 3600, 1, 1),
    ([event(dt(10), 1, 1, 1, 1), event(dt(10, 30), 1, 2, 2, 2), event(dt(11), 0, 1, 1, 3), event(dt(11, 30), 0, 2, 2, 4)], dt(11, 30), 7200, 5400, 4/3, 2),
])
def test_exact_organisation_occupancy(events, target, area, seconds, average, maximum):
    machine, _ = _org_timeline(events, target)
    block = machine["today"]
    assert sum(block["occupancy_area_person_seconds"]) == area
    assert sum(block["occupancy_seconds"]) == seconds
    assert sum(block["occupancy_area_person_seconds"]) / sum(block["occupancy_seconds"]) == pytest.approx(average)
    assert max(block["occupancy_max"]) == maximum


def test_cross_site_exit_is_unmatched_and_unknown_persists_until_expiry():
    machine, runtimes = _org_timeline([event(dt(10), 1, 1), event(dt(11), 0, 2, 2, 2)], dt(13))
    assert runtimes["1"]["occupancy"] == 1
    assert runtimes["2"]["occupancy"] == 0
    org.advance(machine, runtimes, dt(14), include_target_expiry=True)
    assert runtimes["1"]["occupancy"] == 0


def test_horizons_and_no_enabled_capacity_rule():
    horizons = {1: {"ts": dt(12, 5), "stable_until": dt(12, 5)},
                2: {"ts": dt(12), "stable_until": dt(12)},
                3: {"ts": dt(11, 52), "stable_until": dt(11, 52)}, 4: None}
    assert org.organisation_horizons(horizons) == (dt(12, 5), dt(11, 52))
    sites = [site_record(1, 10), site_record(4, 100)]
    state = {"device_watermarks": {}, "metadata": {"device_names": {}}}
    payload = org.derive_payload(site.new_machine(dt(12), {}), sites, state)
    assert payload["capacity"][-1] == [0.0, 0.0]
    with pytest.raises(ValueError, match="no analytically relevant"):
        org.organisation_horizons({4: None})


def test_source_ranges_coalesce_without_crossing_devices():
    ranges = [
        SourceRange("p.d.t", 1, 1, dt(9), dt(10)),
        SourceRange("p.d.t", 1, 1, dt(10), dt(11)),
        SourceRange("p.d.t", 1, 2, dt(9), dt(11)),
    ]
    result = coalesce_ranges(ranges)
    assert result == [SourceRange("p.d.t", 1, 1, dt(9), dt(11)),
                      SourceRange("p.d.t", 1, 2, dt(9), dt(11))]


def test_site_v2_and_corrupt_v3_are_not_reusable():
    record = site_record()
    devices = [device(horizon=dt(12))]
    machine = site.new_machine(dt(0), record)
    site.advance(machine, dt(12), record, expire_at_target=False)
    state = site._build_state(dt(12), devices, record, machine, [])
    assert site.validate_site_state(state, site.stamp(dt(12)), devices) is state
    legacy = dict(state, engine_version=2)
    assert site.validate_site_state(legacy, site.stamp(dt(12)), devices) is None
    corrupt = dict(state)
    corrupt["stable_machine"] = dict(machine)
    corrupt["stable_machine"]["q15"] = dict(machine["q15"], entrances=[0])
    assert site.validate_site_state(corrupt, site.stamp(dt(12)), devices) is None


def test_active_time_empty_period_does_not_dilute_average():
    machine = site.new_machine(dt(10), {})
    runtimes = {"1": {"occupancy": 0, "entry_fifo": []}}
    current, _ = org.current_from(machine, runtimes,
        [event(dt(10), 1), event(dt(10, 30), 0, event_id=2)], dt(11))
    block = current["today"]
    assert sum(block["occupancy_area_person_seconds"]) == 1800
    assert sum(block["occupancy_seconds"]) == 1800


def _organisation_payload(sites, events, target=dt(12, 14)):
    machine = site.new_machine(dt(12), {})
    runtimes = {str(record["id"]): {"occupancy": 0, "entry_fifo": []} for record in sites}
    current, _ = org.current_from(machine, runtimes, sorted(events, key=site.event_order), target)
    state = {"device_watermarks": {}, "metadata": {
        "site_names": {str(record["id"]): record["name"] for record in sites},
        "device_names": {},
    }}
    return org.derive_payload(current, sites, state)


def test_one_site_organisation_traffic_axis_and_percentages():
    payload = _organisation_payload([site_record(1)], [event(dt(12, 1), 1)])
    assert payload["traffic_devices"] == [{"site_id": 1, "name": "1"}]
    assert all(len(bucket) == 1 for bucket in payload["traffic_split_96"])
    assert payload["traffic_split_96"][-1] == [100.0]
    assert payload["traffic_split_96"][0] == [0.0]


def test_organisation_traffic_is_raw_site_split_in_sorted_order():
    sites = [site_record(3), site_record(1), site_record(2)]
    events = []
    counts = {1: 2, 2: 6, 3: 2}
    event_id = 0
    for site_id, count in counts.items():
        for _ in range(count):
            event_id += 1
            events.append(event(dt(12, 1), 0, site_id=site_id,
                                device_id=site_id * 10, event_id=event_id))
    payload = _organisation_payload(sites, events)
    assert payload["traffic_devices"] == [
        {"site_id": 1, "name": "1"}, {"site_id": 2, "name": "2"},
        {"site_id": 3, "name": "3"},
    ]
    assert payload["traffic_split_96"][-1] == [20.0, 60.0, 20.0]
    assert all(len(bucket) == 3 for bucket in payload["traffic_split_96"])


def test_no_device_site_keeps_zero_organisation_traffic_slot():
    payload = _organisation_payload(
        [site_record(3), site_record(1), site_record(2)],
        [event(dt(12, 1), 1, site_id=1), event(dt(12, 2), 0, site_id=2, event_id=2)],
    )
    assert payload["traffic_split_96"][-1] == [50.0, 50.0, 0.0]
    assert payload["traffic_split_96"][0] == [0.0, 0.0, 0.0]


def test_site_traffic_stays_device_based_while_org_aggregates_by_site():
    record = site_record(1)
    devices = [device(10, horizon=dt(12, 15), name="A"),
               device(11, horizon=dt(12, 15), name="B")]
    machine = site.new_machine(dt(12), record)
    rows = []
    for index in range(2):
        rows.append(event(dt(12, 1), 0, 1, 10, index + 1))
    for index in range(6):
        rows.append(event(dt(12, 1), 0, 1, 11, index + 10))
    for row in sorted(rows, key=site.event_order):
        site.apply_event(machine, row, record)
    site_state = site._build_state(dt(12), devices, record, site.new_machine(dt(12), record), rows)
    site_payload = site.derive_payload(machine, record, devices, site_state)
    assert site_payload["traffic_split_96"][-1] == [25.0, 75.0]

    org_rows = rows + [event(dt(12, 1), 0, 2, 20, 100), event(dt(12, 1), 0, 2, 20, 101)]
    org_payload = _organisation_payload([record, site_record(2)], org_rows)
    assert org_payload["traffic_split_96"][-1] == [80.0, 20.0]


def test_snapshot_success_does_not_print_stats(monkeypatch, capsys):
    orchestration = importlib.import_module("snapshot.snapshot")

    class Connector:
        def close(self):
            pass

    resources = orchestration.Resources(object(), Connector(), object())

    @contextmanager
    def fake_connection(_resources):
        yield object()

    monkeypatch.setattr(orchestration, "_resources", lambda: resources)
    monkeypatch.setattr(orchestration, "connection", fake_connection)
    monkeypatch.setattr(orchestration, "_run_attempt", lambda *args: True)
    assert orchestration.Snapshot(1) is True
    assert capsys.readouterr().out == ""


def test_noop_attempt_does_no_bq_or_writes_and_prints_nothing(monkeypatch, capsys):
    orchestration = importlib.import_module("snapshot.snapshot")

    class SQL:
        def commit(self):
            pass

    context = {"sites": [], "devices_by_site": {}, "organisation_snapshot": {}}
    horizon = {1: {"ts": dt(12), "stable_until": dt(12)}}
    monkeypatch.setattr(orchestration, "load_context", lambda *args: context)
    monkeypatch.setattr(orchestration, "_site_plan", lambda *args: ({}, {}, horizon))
    monkeypatch.setattr(orchestration, "_org_classification", lambda *args: ("NO_OP", {}))
    monkeypatch.setattr(orchestration, "fetch_events",
                        lambda *args: pytest.fail("no-op queried BigQuery"))
    monkeypatch.setattr(orchestration, "persist",
                        lambda *args: pytest.fail("no-op attempted persistence"))
    resources = orchestration.Resources(object(), object(), object())
    assert orchestration._run_attempt(resources, SQL(), 1, orchestration.AttemptStats()) is True
    assert capsys.readouterr().out == ""
