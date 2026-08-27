"""Pure organisation Snapshot engine.

Organisation history is intentionally retrospective over current membership because the
database has no effective-dated organisation membership relation.
"""

import copy
from datetime import timedelta

from . import site_engine as engine

ORGANISATION_ENGINE_VERSION = 1


def organisation_horizons(site_results):
    relevant = [value for value in site_results.values() if value is not None]
    if not relevant:
        raise ValueError("Organisation has no analytically relevant sites")
    return max(value["ts"] for value in relevant), min(value["stable_until"] for value in relevant)


def membership(sites, relevant_ids):
    return {"site_ids": sorted(int(s["id"]) for s in sites), "relevant_site_ids": sorted(relevant_ids)}


def device_watermarks(sites, devices_by_site):
    destinations = {int(site["id"]): site["destination"] for site in sites}
    result = {}
    for site_id, devices in devices_by_site.items():
        for device in devices:
            result[str(device["id"])] = {
                "site_id": int(site_id), "status": device["status"],
                "created_at": device["created_at"], "analyzed_until": device["analyzed_until"],
                "destination": destinations[int(site_id)], "analysis_config": device.get("analysis_config"),
            }
    return result


def metadata(sites, devices_by_site):
    return {
        "site_capacities": {str(s["id"]): int(s["max_capacity"]) for s in sites},
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
    device = str(event["device_id"])
    q["traffic_counts"][q_index][device] = q["traffic_counts"][q_index].get(device, 0) + 1
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


def _org_devices(state):
    return [{"id": int(key), "name": state["metadata"]["device_names"].get(key, key)}
            for key in state["device_watermarks"]]


def derive_payload(machine, sites, state):
    capacity = sum(int(site["max_capacity"]) for site in sites)
    synthetic = {"max_capacity": capacity}
    # The site renderer is deliberately shared to preserve the public contract exactly.
    render_state = {"device_watermarks": {
        key: {"name": state["metadata"]["device_names"].get(key, key)}
        for key in state["device_watermarks"]}}
    return engine.derive_payload(machine, synthetic, _org_devices(state), render_state)


def validate_state(state, row_ts, sites, devices_by_site):
    required = {"engine_version", "stable_until", "membership", "site_horizons",
                "device_watermarks", "metadata", "stable_machine", "stable_site_runtime",
                "provisional_events"}
    if not isinstance(state, dict) or state.get("engine_version") != ORGANISATION_ENGINE_VERSION or not required <= set(state):
        return None
    try:
        stable, latest = engine.parse_ts(state["stable_until"]), engine.parse_ts(row_ts)
        if stable > latest or state["stable_machine"].get("cursor_ts") != state["stable_until"]:
            return None
        if not engine.validate_q15(state["stable_machine"]):
            return None
        expected_membership = membership(sites, [sid for sid, values in devices_by_site.items() if any(d["status"] == "enabled" for d in values)])
        if state["membership"] != expected_membership:
            return None
        total = 0
        for runtime in state["stable_site_runtime"].values():
            fifo = [engine.parse_ts(value) for value in runtime["entry_fifo"]]
            if fifo != sorted(fifo) or runtime["occupancy"] != len(fifo) or runtime["occupancy"] < 0:
                return None
            if fifo and fifo[0] + engine.MAX_OPEN_VISIT < stable:
                return None
            total += runtime["occupancy"]
        if total != state["stable_machine"]["occupancy"]:
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
    except (KeyError, TypeError, ValueError):
        return None
    return state


def build_state(stable, sites, devices_by_site, site_horizons, stable_machine, runtimes, provisional):
    relevant = [int(key) for key, value in site_horizons.items() if value is not None]
    return {
        "engine_version": ORGANISATION_ENGINE_VERSION,
        "stable_until": engine.stamp(stable),
        "membership": membership(sites, relevant),
        "site_horizons": {str(key): None if value is None else {
            "ts": engine.stamp(value["ts"]), "stable_until": engine.stamp(value["stable_until"])}
            for key, value in site_horizons.items()},
        "device_watermarks": device_watermarks(sites, devices_by_site),
        "metadata": metadata(sites, devices_by_site),
        "stable_machine": stable_machine,
        "stable_site_runtime": runtimes,
        "provisional_events": provisional,
    }


def compute(sites, devices_by_site, horizons, old_state, old_ts, events, classification):
    latest, stable = organisation_horizons(horizons)
    if classification == "REBUILD":
        start = min(engine.parse_ts(site["created_at"]) for site in sites if horizons[int(site["id"])] is not None)
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
    for raw in combined:
        event = engine.normalize_event_dict(raw)
        if not start <= engine.parse_ts(event["timestamp"]) < latest:
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
    state = build_state(stable, sites, devices_by_site, horizons, stable_machine, runtimes, provisional)
    current, _ = current_from(stable_machine, runtimes, provisional, latest)
    return latest, derive_payload(current, sites, state), state
