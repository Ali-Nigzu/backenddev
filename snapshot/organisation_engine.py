"""Pure organisation Snapshot engine.

Organisation history is intentionally retrospective over current membership because the
database has no effective-dated organisation membership relation.
"""

import copy
from datetime import timedelta

from . import site_engine as engine

ORGANISATION_ENGINE_VERSION = 2


def organisation_horizons(site_results, snapshot_now=None):
    relevant = [value for value in site_results.values() if value is not None]
    if not relevant:
        raise ValueError("Organisation has no analytically relevant sites")
    stable = min(value["stable_until"] for value in relevant)
    return stable if snapshot_now is None else min(stable, snapshot_now)


def membership(sites, relevant_ids):
    return {"site_ids": sorted(int(s["id"]) for s in sites), "relevant_site_ids": sorted(relevant_ids)}


def device_watermarks(sites, devices_by_site, view_until=None):
    destinations = {int(site["id"]): site["destination"] for site in sites}
    result = {}
    for site_id, devices in devices_by_site.items():
        for device in devices:
            horizon = engine.device_horizon(device)
            result[str(device["id"])] = {
                "site_id": int(site_id), "status": device["status"],
                "created_at": device["created_at"], "analyzed_until": device["analyzed_until"],
                "source_horizon": engine.stamp(horizon),
                "destination": destinations[int(site_id)], "analysis_config": device.get("analysis_config"),
            }
    return result


def metadata(sites, devices_by_site):
    return {
        "site_capacities": {str(s["id"]): int(s["max_capacity"]) for s in sites},
        "site_names": {str(s["id"]): s["name"] for s in sites},
        "device_names": {str(d["id"]): d["name"] for values in devices_by_site.values() for d in values},
    }


def _runtime(machine):
    return {"occupancy": int(machine["occupancy"]), "entry_fifo": list(machine["entry_fifo"])}


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
    if machine["occupancy"] != sum(runtime["occupancy"] for runtime in runtimes.values()):
        raise ValueError("Organisation occupancy/runtime mismatch")


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


def validate_state(state, row_ts, sites, devices_by_site):
    required = {"engine_version", "view_until", "stable_until", "membership", "site_source_horizons",
                "device_watermarks", "metadata", "stable_machine", "stable_site_runtime",
                "provisional_events", "pending_events", "traffic_dimension", "current_machine",
                "current_site_runtime"}
    if not isinstance(state, dict) or state.get("engine_version") != ORGANISATION_ENGINE_VERSION or not required <= set(state):
        return None
    if state["traffic_dimension"] != "site":
        return None
    try:
        stable, latest = engine.parse_ts(state["stable_until"]), engine.parse_ts(row_ts)
        if state["view_until"] != engine.stamp(latest) or stable > latest:
            return None
        for machine, cursor in ((state["stable_machine"], stable),
                                (state["current_machine"], latest)):
            if (machine.get("cursor_ts") != engine.stamp(cursor)
                    or not engine.validate_q15(machine)
                    or not engine._calendar_shapes(machine)
                    or int(machine.get("occupancy", -1)) < 0
                    or machine.get("entry_fifo") != []):
                return None
        expected_membership = membership(sites, [sid for sid, values in devices_by_site.items() if any(d["status"] == "enabled" for d in values)])
        if state["membership"] != expected_membership:
            return None
        expected_horizons = {}
        for site in sites:
            values = [engine.device_horizon(device) for device in devices_by_site[int(site["id"])]
                      if device["status"] == "enabled"]
            expected_horizons[str(site["id"])] = (None if not values else {
                "stable_until": engine.stamp(min(values))})
        if state["site_source_horizons"] != expected_horizons:
            return None
        for runtime_key, machine_key, cursor in (("stable_site_runtime", "stable_machine", stable),
                                                  ("current_site_runtime", "current_machine", latest)):
            if set(state[runtime_key]) != {str(site["id"]) for site in sites}:
                return None
            total = 0
            for runtime in state[runtime_key].values():
                fifo = [engine.parse_ts(value) for value in runtime["entry_fifo"]]
                if fifo != sorted(fifo) or runtime["occupancy"] != len(fifo) or runtime["occupancy"] < 0:
                    return None
                if fifo and fifo[0] + engine.MAX_OPEN_VISIT < cursor:
                    return None
                total += runtime["occupancy"]
            if total != state[machine_key]["occupancy"]:
                return None
        identities = set()
        horizons = {int(key): engine.device_horizon({**value, "id": int(key)}) for key, value in state["device_watermarks"].items() if value["status"] == "enabled"}
        for event in state["provisional_events"]:
            event = engine.normalize_event_dict(event)
            instant = engine.parse_ts(event["timestamp"])
            identity = engine.event_identity(event)
            if not stable <= instant < latest or instant >= horizons[event["device_id"]] or identity in identities:
                return None
            identities.add(identity)
        for event in state["pending_events"]:
            event = engine.normalize_event_dict(event)
            instant = engine.parse_ts(event["timestamp"])
            identity = engine.event_identity(event)
            if instant < latest or instant >= horizons[event["device_id"]] or identity in identities:
                return None
            identities.add(identity)
    except (KeyError, TypeError, ValueError):
        return None
    return state


def build_state(stable, sites, devices_by_site, site_horizons, stable_machine, runtimes,
                provisional, view_until=None, current_machine=None, current_runtimes=None,
                pending=None):
    view_until = stable if view_until is None else view_until
    current_machine = copy.deepcopy(stable_machine) if current_machine is None else current_machine
    current_runtimes = copy.deepcopy(runtimes) if current_runtimes is None else current_runtimes
    relevant = [int(key) for key, value in site_horizons.items() if value is not None]
    return {
        "engine_version": ORGANISATION_ENGINE_VERSION,
        "traffic_dimension": "site",
        "view_until": engine.stamp(view_until),
        "stable_until": engine.stamp(stable),
        "membership": membership(sites, relevant),
        "site_source_horizons": {str(key): None if value is None else {
            "stable_until": engine.stamp(value["stable_until"])}
            for key, value in site_horizons.items()},
        "device_watermarks": device_watermarks(sites, devices_by_site, view_until),
        "metadata": metadata(sites, devices_by_site),
        "stable_machine": stable_machine,
        "stable_site_runtime": runtimes,
        "provisional_events": provisional,
        "pending_events": pending or [],
        "current_machine": current_machine,
        "current_site_runtime": current_runtimes,
    }


def compute(sites, devices_by_site, horizons, old_state, old_ts, events, classification,
            snapshot_now=None):
    latest = engine.parse_ts(old_ts) if snapshot_now is None else snapshot_now
    stable = organisation_horizons(horizons, latest)
    if classification in ("TIME_ONLY", "METADATA_ONLY"):
        stable_machine = copy.deepcopy(old_state["stable_machine"])
        runtimes = copy.deepcopy(old_state["stable_site_runtime"])
        current = copy.deepcopy(old_state["current_machine"])
        current_runtimes = copy.deepcopy(old_state["current_site_runtime"])
        advance(current, current_runtimes, latest, include_target_expiry=True)
        state = build_state(engine.parse_ts(old_state["stable_until"]), sites, devices_by_site,
                            horizons, stable_machine, runtimes,
                            copy.deepcopy(old_state["provisional_events"]), latest,
                            current, current_runtimes, copy.deepcopy(old_state["pending_events"]))
        return latest, derive_payload(current, sites, state), state
    if classification == "REBUILD":
        start = min(engine.parse_ts(site["created_at"]) for site in sites if horizons[int(site["id"])] is not None)
        stable_machine = engine.new_machine(start, {})
        runtimes = {str(site["id"]): {"occupancy": 0, "entry_fifo": []} for site in sites}
        combined = events
    else:
        old_stable = engine.parse_ts(old_state["stable_until"])
        stable_machine = copy.deepcopy(old_state["stable_machine"])
        runtimes = copy.deepcopy(old_state["stable_site_runtime"])
        combined = old_state["provisional_events"] + old_state["pending_events"] + events
        start = old_stable
    merged = {}
    source_devices = {int(value["id"]): value for values in devices_by_site.values()
                      for value in values if value["status"] == "enabled"}
    for raw in combined:
        event = engine.normalize_event_dict(raw)
        instant = engine.parse_ts(event["timestamp"])
        device = source_devices.get(int(event["device_id"]))
        if device is None or not start <= instant < engine.device_horizon(device):
            continue
        previous = merged.setdefault(engine.event_identity(event), event)
        if previous != event:
            raise ValueError("Conflicting organisation event identity")
    ordered = sorted(merged.values(), key=engine.event_order)
    for event in ordered:
        if engine.parse_ts(event["timestamp"]) < stable:
            apply_event(stable_machine, runtimes, event)
    # Stable interval is half-open; retain exact-boundary expiries for provisional events.
    advance(stable_machine, runtimes, stable, include_target_expiry=False)
    provisional = [event for event in ordered if stable <= engine.parse_ts(event["timestamp"]) < latest]
    pending = [event for event in ordered if engine.parse_ts(event["timestamp"]) >= latest]
    current, current_runtimes = current_from(stable_machine, runtimes, provisional, latest)
    state = build_state(stable, sites, devices_by_site, horizons, stable_machine, runtimes,
                        provisional, latest, current, current_runtimes, pending)
    return latest, derive_payload(current, sites, state), state
