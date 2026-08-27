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
    assert org.organisation_horizons(horizons) == dt(11, 52)
    assert org.organisation_horizons(horizons, dt(11, 50)) == dt(11, 50)
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


def _site_snapshot(record, devices, now, rows):
    blank = {"ts": site.stamp(now), "state": {}}
    result = site.compute_site(record, devices, blank, "REBUILD", None, rows, now)
    return {"ts": site.stamp(result[0]), "payload": result[1], "state": result[2]}


def test_site_v4_wall_clock_time_only_and_q15_roll():
    record = site_record()
    devices = [device(horizon=dt(12))]
    snapshot = _site_snapshot(record, devices, dt(12, 14, 59), [])
    assert snapshot["state"]["engine_version"] == 4
    classification, previous = site.classify_site(record, devices, snapshot, dt(12, 15))
    assert classification == "TIME_ONLY"
    ts, payload, state = site.compute_site(record, devices, snapshot, classification,
                                           previous, [], dt(12, 15))
    assert ts == dt(12, 15)
    assert state["view_until"] == site.stamp(dt(12, 15))
    assert state["stable_until"] == site.stamp(dt(12))
    assert site.parse_ts(state["current_machine"]["q15"]["window_start"]) + 95 * site.Q15 == dt(12, 15)
    assert all(len(state["current_machine"]["q15"][key]) == 96 for key in site.Q15_ARRAYS)
    assert payload["entrances_96"] == [0] * 96


def test_source_horizon_ahead_of_now_is_retained_but_stable_is_capped():
    record = site_record()
    devices = [device(horizon=dt(12, 2))]
    snapshot = _site_snapshot(record, devices, dt(12), [])
    state = snapshot["state"]
    assert state["stable_until"] == site.stamp(dt(12))
    assert state["device_watermarks"]["1"]["source_horizon"] == site.stamp(dt(12, 2))
    assert state["stable_machine"]["cursor_ts"] == state["view_until"]


def test_known_future_row_is_retained_until_wall_clock_reaches_it_without_refetch():
    record = site_record()
    devices = [device(horizon=dt(12, 2))]
    future = event(dt(12, 1), 1)
    snapshot = _site_snapshot(record, devices, dt(12), [future])
    assert snapshot["state"]["provisional_events"] == []
    assert snapshot["state"]["pending_events"] == [future]
    classification, previous = site.classify_site(record, devices, snapshot, dt(12, 1, 1))
    assert classification == "PROMOTE_ONLY"
    result = site.compute_site(record, devices, snapshot, classification, previous, [], dt(12, 1, 1))
    assert result[2]["current_machine"]["occupancy"] == 1
    assert result[2]["pending_events"] == []
    assert sum(result[2]["current_machine"]["all_time"]["entrances"]) == 1


def test_time_only_open_occupancy_and_expiry_without_source_replay():
    record = site_record()
    devices = [device(horizon=dt(9))]
    snapshot = _site_snapshot(record, devices, dt(11), [event(dt(8), 1)])
    assert snapshot["state"]["current_machine"]["occupancy"] == 1
    classification, previous = site.classify_site(record, devices, snapshot, dt(12))
    assert classification == "TIME_ONLY"
    result = site.compute_site(record, devices, snapshot, classification, previous, [], dt(12))
    at_noon = {"ts": site.stamp(result[0]), "payload": result[1], "state": result[2]}
    assert at_noon["state"]["current_machine"]["occupancy"] == 0
    assert sum(at_noon["state"]["current_machine"]["q15"]["dwell_count"]) == 0
    classification, previous = site.classify_site(record, devices, at_noon, dt(13))
    result = site.compute_site(record, devices, at_noon, classification, previous, [], dt(13))
    assert result[2]["current_machine"]["occupancy"] == 0


def test_late_exit_rebuilds_current_cache_and_removes_provisional_expiry():
    record = site_record()
    first_devices = [device(horizon=dt(9))]
    snapshot = _site_snapshot(record, first_devices, dt(13), [event(dt(8), 1)])
    assert snapshot["state"]["current_machine"]["occupancy"] == 0
    advanced_devices = [device(horizon=dt(11))]
    classification, previous = site.classify_site(record, advanced_devices, snapshot, dt(13))
    assert classification == "INCREMENTAL"
    result = site.compute_site(record, advanced_devices, snapshot, classification, previous,
                               [event(dt(10), 0, event_id=2)], dt(13))
    current = result[2]["current_machine"]
    assert current["occupancy"] == 0 and current["entry_fifo"] == []
    assert sum(current["q15"]["dwell_count"]) == 1
    assert sum(current["q15"]["dwell_sum_seconds"]) == 7200
    assert sum(current["today"]["occupancy_area_person_seconds"]) == 7200


def test_late_entry_changes_past_and_current_state():
    record = site_record()
    initial_devices = [device(horizon=dt(19))]
    snapshot = _site_snapshot(record, initial_devices, dt(20), [])
    advanced_devices = [device(horizon=dt(20))]
    classification, previous = site.classify_site(record, advanced_devices, snapshot, dt(20, 5))
    result = site.compute_site(record, advanced_devices, snapshot, classification, previous,
                               [event(dt(19, 30), 1)], dt(20, 5))
    current = result[2]["current_machine"]
    assert current["occupancy"] == 1
    assert sum(current["q15"]["entrances"]) == 1
    assert sum(current["today"]["entrances"]) == 1
    assert sum(current["all_time"]["entrances"]) == 1


def test_no_enabled_device_site_advances_with_null_stable_and_capacity():
    record = site_record(capacity=25)
    devices = [device(status="disabled")]
    snapshot = _site_snapshot(record, devices, dt(12, 14, 59), [])
    assert snapshot["state"]["stable_until"] is None
    classification, previous = site.classify_site(record, devices, snapshot, dt(12, 15))
    result = site.compute_site(record, devices, snapshot, classification, previous, [], dt(12, 15))
    assert result[0] == dt(12, 15)
    assert result[2]["stable_until"] is None
    assert result[2]["provisional_events"] == []
    assert result[1]["capacity"][-1] == [0.0, 0.0]


def test_current_cache_time_advance_equals_full_reconstruction():
    record = site_record()
    devices = [device(horizon=dt(10))]
    rows = [event(dt(9), 1), event(dt(9, 30), 0, event_id=2)]
    snapshot = _site_snapshot(record, devices, dt(11), rows)
    classification, previous = site.classify_site(record, devices, snapshot, dt(12, 15))
    fast = site.compute_site(record, devices, snapshot, classification, previous, [], dt(12, 15))
    rebuilt = _site_snapshot(record, devices, dt(12, 15), rows)
    assert fast[1] == rebuilt["payload"]
    assert fast[2]["current_machine"] == rebuilt["state"]["current_machine"]


def _org_snapshot(sites, devices_by_site, horizons, now, rows):
    result = org.compute(sites, devices_by_site, horizons, None, site.stamp(now),
                         rows, "REBUILD", now)
    return {"ts": site.stamp(result[0]), "payload": result[1], "state": result[2]}


def test_organisation_v2_wall_clock_cache_and_source_cap():
    sites = [site_record(1), site_record(2)]
    devices_by_site = {1: [device(1, 1, dt(12, 2))], 2: [device(2, 2, dt(11, 30))]}
    horizons = {1: {"stable_until": dt(12, 2)}, 2: {"stable_until": dt(11, 30)}}
    snapshot = _org_snapshot(sites, devices_by_site, horizons, dt(12), [])
    state = snapshot["state"]
    assert state["engine_version"] == 2
    assert state["view_until"] == site.stamp(dt(12))
    assert state["stable_until"] == site.stamp(dt(11, 30))
    assert "site_horizons" not in state
    assert state["site_source_horizons"]["1"] == {"stable_until": site.stamp(dt(12, 2))}
    assert org.validate_state(state, snapshot["ts"], sites, devices_by_site) is state
    result = org.compute(sites, devices_by_site, horizons, state, snapshot["ts"], [],
                         "TIME_ONLY", dt(12, 15))
    assert result[0] == dt(12, 15)
    assert result[2]["stable_until"] == site.stamp(dt(11, 30))
    assert result[2]["current_machine"]["cursor_ts"] == site.stamp(dt(12, 15))


def test_organisation_late_exit_repairs_site_local_runtime_and_expiry():
    sites = [site_record(1), site_record(2)]
    old_devices = {1: [device(1, 1, dt(9))], 2: [device(2, 2, dt(9))]}
    old_horizons = {1: {"stable_until": dt(9)}, 2: {"stable_until": dt(9)}}
    snapshot = _org_snapshot(sites, old_devices, old_horizons, dt(13),
                             [event(dt(8), 1, site_id=1)])
    assert snapshot["state"]["current_site_runtime"]["1"]["occupancy"] == 0
    new_devices = {1: [device(1, 1, dt(11))], 2: [device(2, 2, dt(11))]}
    new_horizons = {1: {"stable_until": dt(11)}, 2: {"stable_until": dt(11)}}
    result = org.compute(sites, new_devices, new_horizons, snapshot["state"], snapshot["ts"],
                         [event(dt(10), 0, site_id=1, event_id=2)], "INCREMENTAL", dt(13))
    state = result[2]
    assert state["current_site_runtime"]["1"] == {"occupancy": 0, "entry_fifo": []}
    assert state["current_site_runtime"]["2"] == {"occupancy": 0, "entry_fifo": []}
    assert sum(state["current_machine"]["q15"]["dwell_sum_seconds"]) == 7200
    assert state["current_machine"]["occupancy"] == 0


def test_organisation_v2_validation_accepts_site_local_positive_occupancy():
    sites = [site_record(1)]
    devices_by_site = {1: [device(1, 1, dt(11))]}
    horizons = {1: {"stable_until": dt(11)}}
    snapshot = _org_snapshot(sites, devices_by_site, horizons, dt(12),
                             [event(dt(10), 1, site_id=1)])
    assert snapshot["state"]["current_machine"]["occupancy"] == 1
    assert org.validate_state(snapshot["state"], snapshot["ts"], sites, devices_by_site) is snapshot["state"]


def test_orchestration_time_only_plans_zero_source_ranges():
    orchestration = importlib.import_module("snapshot.snapshot")
    record = site_record(1)
    devices = [device(1, 1, dt(12))]
    site_snapshot = _site_snapshot(record, devices, dt(12), [])
    horizons = {1: {"stable_until": dt(12)}}
    org_snapshot = _org_snapshot([record], {1: devices}, horizons, dt(12), [])
    context = {
        "sites": [record], "devices_by_site": {1: devices},
        "site_snapshots": {1: site_snapshot}, "organisation_snapshot": org_snapshot,
        "snapshot_now": dt(12, 5),
    }
    classifications, previous, planned_horizons = orchestration._site_plan(context, dt(12, 5))
    org_classification, org_previous = orchestration._org_classification(
        context, planned_horizons, dt(12, 5))
    assert classifications == {1: "TIME_ONLY"}
    assert org_classification == "TIME_ONLY"
    assert orchestration._ranges(context, classifications, previous, org_classification,
                                 org_previous, planned_horizons) == []
