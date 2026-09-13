"""Pure organisation Snapshot engine.

Organisation history is intentionally retrospective over current membership because the
database has no effective-dated organisation membership relation.
"""

import copy
from datetime import timedelta

from . import site_engine as engine

def organisation_horizons(site_results, snapshot_now=None):
    relevant = [value for value in site_results.values() if value is not None]
    if not relevant:
        if snapshot_now is None:
            raise ValueError("snapshot_now is required when there are no active analytical sites")
        return snapshot_now
    stable = min(value["stable_until"] for value in relevant)
    return stable if snapshot_now is None else min(stable, snapshot_now)


def membership(sites):
    return {"site_ids": sorted(int(s["id"]) for s in sites)}


def device_watermarks(devices_by_site, view_until=None):
    result = {}
    for site_id, devices in devices_by_site.items():
        for device in devices:
            consumed_until = engine.device_horizon(device, view_until)
            result[str(device["id"])] = {
                "site_id": int(site_id), "enabled": device["enabled"],
                "created_at": device["created_at"],
                "consumed_until": engine.stamp(consumed_until),
            }
    return result


def metadata(sites, organisation_enabled=None):
    return {
        "organisation_enabled": organisation_enabled,
        "site_capacities": {str(s["id"]): int(s["max_capacity"]) for s in sites},
        "site_names": {str(s["id"]): s["name"] for s in sites},
        "site_enabled": {str(s["id"]): s["enabled"] for s in sites},
    }


def _expire_before(machine, runtimes, target, include_target):
    while True:
        candidates = []
        for site_id, runtime in runtimes.items():
            if runtime["entry_fifo"]:
                candidates.append((engine.parse_ts(runtime["entry_fifo"][0]) + engine.MAX_OPEN_VISIT, int(site_id)))
        if not candidates:
            return
        expiry, site_id = min(candidates)
        if expiry > target or (expiry == target and not include_target):
            return
        engine.advance(machine, expiry, {}, expire_at_target=False)
        runtime = runtimes[str(site_id)]
        runtime["entry_fifo"].pop(0)
        runtime["occupancy"] -= 1
        machine["occupancy"] -= 1


def apply_event(machine, runtimes, event):
    instant = engine.parse_ts(event["timestamp"])
    _expire_before(machine, runtimes, instant, include_target=False)
    engine.advance(machine, instant, {}, expire_at_target=False)
    site_id = str(event["site_id"])
    runtime = runtimes.setdefault(site_id, {"occupancy": 0, "entry_fifo": []})
    active = engine._active(machine, instant, {})
    q, q_index = active[0]
    if event["event"] == 1:
        runtime["occupancy"] += 1
        runtime["entry_fifo"].append(event["timestamp"])
        machine["occupancy"] += 1
        for block, index in active:
            engine._increment(block, "entrances", index)
        local = instant.astimezone(engine._zone())
        age, sex = event["age_bucket"], event["sex"]
        for block in (machine["today"], machine["yesterday"]):
            if block["local_date"] == local.date().isoformat():
                block["age_counts"][age] += 1; block["sex_counts"][sex] += 1
        machine["week"]["age_counts_by_day"][local.weekday()][age] += 1
        machine["week"]["sex_counts_by_day"][local.weekday()][sex] += 1
        machine["month"]["age_counts_by_week"][active[3][1]][age] += 1
        machine["month"]["sex_counts_by_week"][active[3][1]][sex] += 1
        machine["quarter"]["age_counts_by_week"][active[4][1]][age] += 1
        machine["quarter"]["sex_counts_by_week"][active[4][1]][sex] += 1
        machine["year"]["age_counts_by_month"][local.month - 1][age] += 1
        machine["year"]["sex_counts_by_month"][local.month - 1][sex] += 1
        machine["all_time"]["age_counts"][age] += 1
        machine["all_time"]["sex_counts"][sex] += 1
    else:
        for block, index in active:
            engine._increment(block, "exits", index)
        if runtime["entry_fifo"]:
            entered = engine.parse_ts(runtime["entry_fifo"].pop(0))
            runtime["occupancy"] -= 1
            machine["occupancy"] -= 1
            q["dwell_sum_seconds"][q_index] += int((instant - entered).total_seconds())
            q["dwell_count"][q_index] += 1
    # Site machines use device IDs for traffic.  At organisation scope the
    # corresponding authoritative traffic dimension is the member site ID.
    site_key = str(event["site_id"])
    q["traffic_counts"][q_index][site_key] = q["traffic_counts"][q_index].get(site_key, 0) + 1
    engine._refresh_peaks(machine)


def advance(machine, runtimes, target, include_target_expiry=True):
    _expire_before(machine, runtimes, target, include_target_expiry)
    engine.advance(machine, target, {}, expire_at_target=False)


def current_from(stable_machine, stable_runtime, provisional, target):
    machine, runtimes = copy.deepcopy(stable_machine), copy.deepcopy(stable_runtime)
    groups = {}
    for event in provisional:
        groups.setdefault(event["timestamp"], []).append(event)
    for timestamp in sorted(groups, key=engine.parse_ts):
        instant = engine.parse_ts(timestamp)
        _expire_before(machine, runtimes, instant, include_target=False)
        for event in sorted(groups[timestamp], key=engine.event_order):
            apply_event(machine, runtimes, event)
        _expire_before(machine, runtimes, instant, include_target=True)
    advance(machine, runtimes, target, include_target_expiry=True)
    return machine, runtimes


def _traffic_sites(sites):
    """Adapt sorted site identities to the shared traffic payload renderer."""
    return [{"id": int(value["id"]), "name": value["name"]}
            for value in sorted(sites, key=lambda value: int(value["id"]))]


def derive_payload(machine, sites, state):
    capacity = sum(int(site["max_capacity"]) for site in sites)
    synthetic = {"max_capacity": capacity}
    # Reuse the mathematical renderer, but provide sites as its traffic axis.
    # This aggregates raw event counts by site before percentages are derived.
    render_state = {"device_watermarks": {
        str(value["id"]): {"name": value["name"]} for value in _traffic_sites(sites)}}
    payload = engine.derive_payload(machine, synthetic, _traffic_sites(sites), render_state)
    payload["traffic_devices"] = [
        {"site_id": identity["device_id"], "name": identity["name"]}
        for identity in payload["traffic_devices"]
    ]
    return payload


def build_state(stable, sites, devices_by_site, stable_machine, runtimes,
                provisional, view_until, current_machine, current_runtimes,
                organisation_enabled):
    return {
        "version": engine.SNAPSHOT_STATE_VERSION,
        "traffic_dimension": "site",
        "view_until": engine.stamp(view_until),
        "stable_until": engine.stamp(stable),
        "membership": membership(sites),
        "device_watermarks": device_watermarks(devices_by_site, view_until),
        "metadata": metadata(sites, organisation_enabled),
        "stable_machine": stable_machine,
        "stable_site_runtime": runtimes,
        "provisional_events": provisional,
        "current_machine": current_machine,
        "current_site_runtime": current_runtimes,
    }


def compute(sites, devices_by_site, horizons, old_state, old_ts, events, classification,
            snapshot_now=None, organisation_enabled=None):
    latest = engine.parse_ts(old_ts) if snapshot_now is None else snapshot_now
    stable = organisation_horizons(horizons, latest)
    if classification in ("TIME_ONLY", "METADATA_ONLY"):
        stable_machine = old_state["stable_machine"]
        runtimes = old_state["stable_site_runtime"]
        current = copy.deepcopy(old_state["current_machine"])
        current_runtimes = copy.deepcopy(old_state["current_site_runtime"])
        advance(current, current_runtimes, latest, include_target_expiry=True)
        state = build_state(engine.parse_ts(old_state["stable_until"]), sites, devices_by_site,
                            stable_machine, runtimes,
                            old_state["provisional_events"], latest,
                            current, current_runtimes, organisation_enabled)
        return latest, derive_payload(current, sites, state), state
    if classification == "REBUILD":
        start = min(engine.parse_ts(site["created_at"]) for site in sites)
        stable_machine = engine.new_machine(start, {})
        runtimes = {str(site["id"]): {"occupancy": 0, "entry_fifo": []} for site in sites}
        combined = events
    else:
        old_stable = engine.parse_ts(old_state["stable_until"])
        stable_machine = copy.deepcopy(old_state["stable_machine"])
        runtimes = copy.deepcopy(old_state["stable_site_runtime"])
        combined = old_state["provisional_events"] + events
        start = old_stable
    merged = {}
    source_devices = {int(value["id"]): value for values in devices_by_site.values()
                      for value in values}
    for event in combined:
        instant = engine.parse_ts(event["timestamp"])
        device = source_devices.get(int(event["device_id"]))
        if device is None or not start <= instant < engine.device_horizon(device, latest):
            continue
        merged.setdefault(engine.event_identity(event), event)
    ordered = sorted(merged.values(), key=engine.event_order)
    for event in ordered:
        if engine.parse_ts(event["timestamp"]) < stable:
            apply_event(stable_machine, runtimes, event)
    # Stable interval is half-open; retain exact-boundary expiries for provisional events.
    advance(stable_machine, runtimes, stable, include_target_expiry=False)
    provisional = [event for event in ordered if stable <= engine.parse_ts(event["timestamp"]) < latest]
    current, current_runtimes = current_from(stable_machine, runtimes, provisional, latest)
    state = build_state(stable, sites, devices_by_site, stable_machine, runtimes,
                        provisional, latest, current, current_runtimes,
                        organisation_enabled)
    return latest, derive_payload(current, sites, state), state
