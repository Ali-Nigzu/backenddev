import copy
from datetime import datetime, timedelta, timezone

SITE_ENGINE_VERSION = 4
ENGINE_VERSION = SITE_ENGINE_VERSION
MAX_RETRIES = 2
Q15 = timedelta(minutes=15)
MAX_OPEN_VISIT = timedelta(hours=4)


def _zone():
    return timezone.utc


def _dt(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _stamp(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zeros(size):
    return [0 for _ in range(size)]


def _bucket(size):
    return {
        "entrances": _zeros(size), "exits": _zeros(size),
        "occupancy_area_person_seconds": _zeros(size),
        "occupancy_seconds": _zeros(size),
        "occupancy_min_positive": _zeros(size),
        "occupancy_max": _zeros(size),
    }


def floor_to_q15(value):
    value = value.astimezone(timezone.utc)
    return value.replace(minute=value.minute - value.minute % 15, second=0, microsecond=0)


def _machine(start, site):
    local = start.astimezone(_zone())
    monday = (local - timedelta(days=local.weekday())).date()
    q_start = floor_to_q15(start) - timedelta(minutes=15 * 95)
    machine = {
        "cursor_ts": _stamp(start), "occupancy": 0, "entry_fifo": [],
        "q15": {
            "window_start": _stamp(q_start), "entrances": _zeros(96), "exits": _zeros(96),
            "occupancy_area_person_seconds": _zeros(96), "occupancy_seconds": _zeros(96),
            "occupancy_min_positive": _zeros(96), "occupancy_max": _zeros(96),
            "rolling_peak_occupancy": _zeros(96), "dwell_sum_seconds": _zeros(96),
            "dwell_count": _zeros(96), "traffic_counts": [{} for _ in range(96)],
        },
        "today": {"local_date": local.date().isoformat(), **_bucket(24), "age_counts": _zeros(6), "sex_counts": _zeros(2)},
        "yesterday": {"local_date": (local.date() - timedelta(days=1)).isoformat(), **_bucket(24), "age_counts": _zeros(6), "sex_counts": _zeros(2)},
        "week": {"iso_week_start": monday.isoformat(), **_bucket(7), "age_counts_by_day": [_zeros(6) for _ in range(7)], "sex_counts_by_day": [_zeros(2) for _ in range(7)]},
        "month": {"first_iso_week_start": (monday - timedelta(weeks=3)).isoformat(), **_bucket(4), "age_counts_by_week": [_zeros(6) for _ in range(4)], "sex_counts_by_week": [_zeros(2) for _ in range(4)]},
        "quarter": {"first_iso_week_start": (monday - timedelta(weeks=11)).isoformat(), **_bucket(12), "age_counts_by_week": [_zeros(6) for _ in range(12)], "sex_counts_by_week": [_zeros(2) for _ in range(12)]},
        "year": {"local_year": local.year, **_bucket(12), "age_counts_by_month": [_zeros(6) for _ in range(12)], "sex_counts_by_month": [_zeros(2) for _ in range(12)]},
        "all_time": {"start_year": local.year, **_bucket(1), "age_counts": _zeros(6), "sex_counts": _zeros(2)},
    }
    return machine


def _device_watermarks(devices, view_until=None):
    result = {}
    for device in devices:
        horizon = device_horizon(device)
        result[str(device["id"])] = {
            "name": device["name"], "status": device["status"],
            "created_at": device["created_at"], "analyzed_until": device["analyzed_until"],
            "source_horizon": _stamp(horizon),
            "analysis_config": device.get("analysis_config"),
        }
    return result


def _site_metadata(site):
    return {"max_capacity": site["max_capacity"], "destination": site.get("destination", site.get("bigquery_destination"))}


def _state_v2(state):
    required = {"engine_version", "stable_until", "device_watermarks", "site_metadata", "stable_machine", "provisional_events"}
    if not isinstance(state, dict) or state.get("engine_version") != ENGINE_VERSION or not required.issubset(state):
        return None
    if not isinstance(state["device_watermarks"], dict) or not isinstance(state["site_metadata"], dict) or not isinstance(state["stable_machine"], dict) or not isinstance(state["provisional_events"], list):
        return None
    try:
        _dt(state["stable_until"])
        if state["stable_machine"]["cursor_ts"] != state["stable_until"]:
            return None
        _dt(state["stable_machine"]["cursor_ts"])
    except (KeyError, TypeError, ValueError):
        return None
    return state


def _build_state(stable, devices, site, stable_machine, provisional_events,
                 view_until=None, current_machine=None, pending_events=None):
    # The optional arguments keep the small pure-engine fixtures readable while all
    # production v4 states always contain an explicit wall-clock current cache.
    view_until = stable if view_until is None else view_until
    current_machine = copy.deepcopy(stable_machine) if current_machine is None else current_machine
    return {
        "engine_version": ENGINE_VERSION, "view_until": _stamp(view_until),
        "stable_until": None if stable is None else _stamp(stable),
        "device_watermarks": _device_watermarks(devices, view_until),
        "site_metadata": _site_metadata(site), "stable_machine": stable_machine,
        "provisional_events": provisional_events, "pending_events": pending_events or [],
        "current_machine": current_machine,
    }

def _q_index(machine, value):
    start = _dt(machine["q15"]["window_start"])
    return int((value - start).total_seconds() // 900)


def _roll_q15(machine):
    q = machine["q15"]
    for key in ("entrances", "exits", "occupancy_area_person_seconds", "occupancy_seconds", "occupancy_min_positive", "occupancy_max", "rolling_peak_occupancy", "dwell_sum_seconds", "dwell_count", "traffic_counts"):
        q[key].pop(0)
        q[key].append({} if key == "traffic_counts" else 0)
    q["window_start"] = _stamp(_dt(q["window_start"]) + Q15)


def _roll_calendar(machine, instant, site):
    zone = _zone()
    local = instant.astimezone(zone)
    if local.date().isoformat() != machine["today"]["local_date"]:
        machine["yesterday"] = copy.deepcopy(machine["today"])
        machine["yesterday"]["local_date"] = (local.date() - timedelta(days=1)).isoformat()
        machine["today"] = {"local_date": local.date().isoformat(), **_bucket(24), "age_counts": _zeros(6), "sex_counts": _zeros(2)}
    monday = (local - timedelta(days=local.weekday())).date().isoformat()
    if monday != machine["week"]["iso_week_start"]:
        machine["week"] = {"iso_week_start": monday, **_bucket(7), "age_counts_by_day": [_zeros(6) for _ in range(7)], "sex_counts_by_day": [_zeros(2) for _ in range(7)]}
        for name, size in (("month", 4), ("quarter", 12)):
            block = machine[name]
            for key in ("entrances", "exits", "occupancy_area_person_seconds", "occupancy_seconds", "occupancy_min_positive", "occupancy_max", "age_counts_by_week", "sex_counts_by_week"):
                block[key].pop(0)
                block[key].append(_zeros(6) if key == "age_counts_by_week" else _zeros(2) if key == "sex_counts_by_week" else 0)
            block["first_iso_week_start"] = (datetime.strptime(block["first_iso_week_start"], "%Y-%m-%d").date() + timedelta(days=7)).isoformat()
    if local.year != machine["year"]["local_year"]:
        machine["year"] = {"local_year": local.year, **_bucket(12), "age_counts_by_month": [_zeros(6) for _ in range(12)], "sex_counts_by_month": [_zeros(2) for _ in range(12)]}
    all_time = machine["all_time"]
    while all_time["start_year"] + len(all_time["entrances"]) <= local.year:
        for key in ("entrances", "exits", "occupancy_area_person_seconds", "occupancy_seconds", "occupancy_min_positive", "occupancy_max"):
            all_time[key].append(0)


def _active(machine, instant, site):
    zone = _zone()
    local = instant.astimezone(zone)
    q = machine["q15"]
    index = _q_index(machine, instant)
    local_date = local.date().isoformat()
    day = local.weekday()
    week_index = int((local.date() - datetime.strptime(machine["month"]["first_iso_week_start"], "%Y-%m-%d").date()).days // 7)
    quarter_index = int((local.date() - datetime.strptime(machine["quarter"]["first_iso_week_start"], "%Y-%m-%d").date()).days // 7)
    return [(q, index), (machine["today"] if local_date == machine["today"]["local_date"] else machine["yesterday"], local.hour), (machine["week"], day), (machine["month"], max(0, min(3, week_index))), (machine["quarter"], max(0, min(11, quarter_index))), (machine["year"], local.month - 1), (machine["all_time"], local.year - machine["all_time"]["start_year"])]


def _add_occupancy(block, index, occupancy, seconds):
    if occupancy <= 0:
        return
    block["occupancy_area_person_seconds"][index] += occupancy * seconds
    block["occupancy_seconds"][index] += seconds
    current = block["occupancy_min_positive"][index]
    block["occupancy_min_positive"][index] = occupancy if current == 0 else min(current, occupancy)
    block["occupancy_max"][index] = max(block["occupancy_max"][index], occupancy)


def _next_expiry(machine):
    if not machine["entry_fifo"]:
        return None
    return _dt(machine["entry_fifo"][0]) + MAX_OPEN_VISIT


def _expire_entries(machine, instant):
    while machine["entry_fifo"] and _next_expiry(machine) <= instant:
        machine["entry_fifo"].pop(0)
        machine["occupancy"] = max(0, machine["occupancy"] - 1)


def _next_boundary(value, target, site, machine, expire_at_target):
    zone = _zone()
    values = [target]
    quarter = datetime.fromtimestamp(((int(value.timestamp()) // 900) + 1) * 900, tz=timezone.utc)
    values.append(quarter)
    local = value.astimezone(zone)
    values.append((local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).astimezone(timezone.utc))
    expiry = _next_expiry(machine)
    if expiry and expiry > value and (expiry < target or expire_at_target and expiry == target):
        values.append(expiry)
    return min(v for v in values if v > value)


def _refresh_peaks(machine):
    maxima = machine["q15"]["occupancy_max"]
    machine["q15"]["rolling_peak_occupancy"] = [max(maxima[max(0, i - 95):i + 1]) for i in range(96)]


def _ensure_q15_interval(machine, instant):
    q_end = _dt(machine["q15"]["window_start"]) + timedelta(minutes=96 * 15)
    if instant >= q_end:
        _roll_q15(machine)


def advance(machine, target, site, expire_at_target=True):
    cursor = _dt(machine["cursor_ts"])
    if expire_at_target or cursor != target:
        _expire_entries(machine, cursor)
    while cursor < target:
        _ensure_q15_interval(machine, cursor)
        boundary = _next_boundary(cursor, target, site, machine, expire_at_target)
        elapsed = int((boundary - cursor).total_seconds())
        for block, index in _active(machine, cursor, site):
            _add_occupancy(block, index, machine["occupancy"], elapsed)
        cursor = boundary
        machine["cursor_ts"] = _stamp(cursor)
        _roll_calendar(machine, cursor, site)
        if cursor < target and cursor.minute % 15 == 0 and cursor.second == 0:
            _roll_q15(machine)
        if cursor < target or expire_at_target:
            _expire_entries(machine, cursor)
    _refresh_peaks(machine)


def _increment(block, key, index):
    block[key][index] += 1


def apply_event(machine, event, site):
    instant = _dt(event["timestamp"])
    advance(machine, instant, site, expire_at_target=False)
    _ensure_q15_interval(machine, instant)
    active = _active(machine, instant, site)
    q, q_index = active[0]
    if event["event"] == 1:
        machine["occupancy"] += 1
        machine["entry_fifo"].append(event["timestamp"])
        for block, index in active:
            _increment(block, "entrances", index)
        zone = _zone()
        local = instant.astimezone(zone)
        day = local.weekday()
        week_index = active[3][1]
        quarter_index = active[4][1]
        month_index = local.month - 1
        age, sex = event["age_bucket"], event["sex"]
        for block in (machine["today"], machine["yesterday"]):
            if block["local_date"] == local.date().isoformat():
                block["age_counts"][age] += 1; block["sex_counts"][sex] += 1
        machine["week"]["age_counts_by_day"][day][age] += 1; machine["week"]["sex_counts_by_day"][day][sex] += 1
        machine["month"]["age_counts_by_week"][week_index][age] += 1; machine["month"]["sex_counts_by_week"][week_index][sex] += 1
        machine["quarter"]["age_counts_by_week"][quarter_index][age] += 1; machine["quarter"]["sex_counts_by_week"][quarter_index][sex] += 1
        machine["year"]["age_counts_by_month"][month_index][age] += 1; machine["year"]["sex_counts_by_month"][month_index][sex] += 1
        machine["all_time"]["age_counts"][age] += 1; machine["all_time"]["sex_counts"][sex] += 1
    else:
        for block, index in active:
            _increment(block, "exits", index)
        if machine["entry_fifo"]:
            entered = _dt(machine["entry_fifo"].pop(0))
            machine["occupancy"] = max(0, machine["occupancy"] - 1)
            q["dwell_sum_seconds"][q_index] += int((instant - entered).total_seconds())
            q["dwell_count"][q_index] += 1
    device_id = str(event["device_id"])
    q["traffic_counts"][q_index][device_id] = q["traffic_counts"][q_index].get(device_id, 0) + 1
    _refresh_peaks(machine)


def _pct(values):
    total = sum(values)
    return [0.0 if total == 0 else value * 100 / total for value in values]


def _occupancy(block):
    return [[0.0 if seconds == 0 else area / seconds, minimum, maximum] for area, seconds, minimum, maximum in zip(block["occupancy_area_person_seconds"], block["occupancy_seconds"], block["occupancy_min_positive"], block["occupancy_max"])]


def _rollup(block, age, sex):
    return {"entrances": block["entrances"], "occupancy": _occupancy(block), "exits": block["exits"], "age_pct": _pct(age), "sex_pct": _pct(sex)}


def derive_payload(machine, site, devices, state):
    q = machine["q15"]
    current_ids = {int(d["id"]): d["name"] for d in devices}
    for values in q["traffic_counts"]:
        for device_id in values:
            current_ids.setdefault(int(device_id), state["device_watermarks"][device_id]["name"])
    axes = [{"device_id": device_id, "name": current_ids[device_id]} for device_id in sorted(current_ids)]
    traffic = []
    for values in q["traffic_counts"]:
        total = sum(values.values())
        traffic.append([0.0 if total == 0 else values.get(str(axis["device_id"]), 0) * 100 / total for axis in axes])
    peak = q["rolling_peak_occupancy"]
    today = _rollup(machine["today"], machine["today"]["age_counts"], machine["today"]["sex_counts"])
    cursor = _dt(machine["cursor_ts"])
    hours = cursor.hour + int(any((cursor.minute, cursor.second, cursor.microsecond)))
    for key in ("entrances", "exits", "occupancy"):
        today[key] = today[key][:hours]
    yesterday = _rollup(machine["yesterday"], machine["yesterday"]["age_counts"], machine["yesterday"]["sex_counts"])
    week = _rollup(machine["week"], [sum(row[i] for row in machine["week"]["age_counts_by_day"]) for i in range(6)], [sum(row[i] for row in machine["week"]["sex_counts_by_day"]) for i in range(2)])
    month = _rollup(machine["month"], [sum(row[i] for row in machine["month"]["age_counts_by_week"]) for i in range(6)], [sum(row[i] for row in machine["month"]["sex_counts_by_week"]) for i in range(2)])
    quarter = _rollup(machine["quarter"], [sum(row[i] for row in machine["quarter"]["age_counts_by_week"]) for i in range(6)], [sum(row[i] for row in machine["quarter"]["sex_counts_by_week"]) for i in range(2)])
    year = _rollup(machine["year"], [sum(row[i] for row in machine["year"]["age_counts_by_month"]) for i in range(6)], [sum(row[i] for row in machine["year"]["sex_counts_by_month"]) for i in range(2)])
    all_time = _rollup(machine["all_time"], machine["all_time"]["age_counts"], machine["all_time"]["sex_counts"])
    averages = [0.0 if seconds == 0 else area / seconds for area, seconds in zip(q["occupancy_area_person_seconds"], q["occupancy_seconds"])]
    return {"entrances_96": q["entrances"], "occupancy_96": averages, "exits_96": q["exits"], "footfall_96": [a + b for a, b in zip(q["entrances"], q["exits"])], "dwell_time_96": [0.0 if count == 0 else total / count for total, count in zip(q["dwell_sum_seconds"], q["dwell_count"])], "traffic_devices": axes, "traffic_split_96": traffic, "capacity": [[average * 100 / site["max_capacity"], hard * 100 / site["max_capacity"]] for average, hard in zip(averages, peak)], "today": today, "yesterday": yesterday, "week": week, "month": month, "quarter": quarter, "year": year, "all_time": all_time}


def _horizons(previous_ts, previous_stable, devices):
    enabled = [d for d in devices if d["status"] == "enabled"]
    horizons = [_dt(d["analyzed_until"]) for d in enabled if d["analyzed_until"]]
    ts = max(horizons) if horizons else previous_ts
    safe = [_dt(d["analyzed_until"]) if d["analyzed_until"] else _dt(d["created_at"]) for d in enabled]
    stable = max(previous_stable, min(safe)) if safe else previous_stable
    return ts, stable


def _event_key(event):
    return event["timestamp"], -event["event"], event["event_id"]


def _merge_events(events, devices, start, end):
    horizons = {d["id"]: _dt(d["analyzed_until"]) for d in devices if d["analyzed_until"]}
    deduplicated = {}
    for event in events:
        try:
            instant, device_id, event_id = _dt(event["timestamp"]), int(event["device_id"]), int(event["event_id"])
            normalised = {"device_id": device_id, "event_id": event_id, "event": int(event["event"]), "timestamp": _stamp(instant), "sex": int(event["sex"]), "age_bucket": int(event["age_bucket"])}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed Snapshot provisional event") from error
        if device_id not in horizons or not start <= instant < end or instant >= horizons[device_id]:
            continue
        previous = deduplicated.setdefault(event_id, normalised)
        if previous != normalised:
            raise ValueError(f"Conflicting event_id: {event_id}")
    return sorted(deduplicated.values(), key=_event_key)


def _provisional_is_safe(state, snapshot_ts):
    try:
        start, end = _dt(state["stable_until"]), _dt(snapshot_ts)
        for event in state["provisional_events"]:
            if not start <= _dt(event["timestamp"]) < end:
                return False
            for key in ("device_id", "event_id", "event", "sex", "age_bucket"):
                int(event[key])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _membership_is_safe(previous, current):
    if set(previous) != set(current):
        return False
    for device_id, watermark in current.items():
        old = previous[device_id]
        if any(old.get(key) != watermark.get(key) for key in ("status", "created_at")):
            return False
        old_horizon, new_horizon = old.get("analyzed_until"), watermark.get("analyzed_until")
        if old_horizon and (not new_horizon or _dt(new_horizon) < _dt(old_horizon)):
            return False
    return True


def _current_from(stable_machine, provisional, ts, site):
    current = copy.deepcopy(stable_machine)
    for event in provisional:
        apply_event(current, event, site)
    advance(current, ts, site)
    return current


# Public names for the I/O and organisation layers.
parse_ts = _dt
stamp = _stamp
new_machine = _machine
derive_site_payload = derive_payload

Q15_ARRAYS = (
    "entrances", "exits", "occupancy_area_person_seconds", "occupancy_seconds",
    "occupancy_min_positive", "occupancy_max", "rolling_peak_occupancy",
    "dwell_sum_seconds", "dwell_count", "traffic_counts",
)


def device_horizon(device):
    """The captured immutable source horizon, including never-analysed devices."""
    return _dt(device["analyzed_until"] or device["created_at"])


def site_horizons(site, devices):
    enabled = [device for device in devices if device["status"] == "enabled"]
    if not enabled:
        return None
    values = [device_horizon(device) for device in enabled]
    return max(values), min(values)


def align_q15(machine, instant):
    """Make the final structural bucket contain instant, even on an exact boundary."""
    wanted = floor_to_q15(instant) - Q15 * 95
    current = _dt(machine["q15"]["window_start"])
    if current > wanted:
        raise ValueError("q15 window is ahead of its cursor")
    while current < wanted:
        _roll_q15(machine)
        current += Q15


_advance = advance


def advance(machine, target, site, expire_at_target=True):
    _advance(machine, target, site, expire_at_target=expire_at_target)
    align_q15(machine, target)


def validate_q15(machine):
    try:
        cursor = _dt(machine["cursor_ts"])
        q = machine["q15"]
        start = _dt(q["window_start"])
        if start != floor_to_q15(start) or start + Q15 * 95 != floor_to_q15(cursor):
            return False
        return all(isinstance(q[key], list) and len(q[key]) == 96 for key in Q15_ARRAYS)
    except (KeyError, TypeError, ValueError):
        return False


def _calendar_shapes(machine):
    sizes = {"today": 24, "yesterday": 24, "week": 7, "month": 4,
             "quarter": 12, "year": 12}
    for name, size in sizes.items():
        block = machine.get(name)
        if not isinstance(block, dict):
            return False
        for key in ("entrances", "exits", "occupancy_area_person_seconds",
                    "occupancy_seconds", "occupancy_min_positive", "occupancy_max"):
            if not isinstance(block.get(key), list) or len(block[key]) != size:
                return False
    return isinstance(machine.get("all_time", {}).get("entrances"), list)


def normalize_event_dict(event, site=None):
    required = ("device_id", "event_id", "event", "timestamp", "sex", "age_bucket")
    if not all(key in event for key in required):
        raise ValueError("Malformed Snapshot event")
    result = {key: int(event[key]) for key in ("device_id", "event_id", "event", "sex", "age_bucket")}
    result["timestamp"] = _stamp(_dt(event["timestamp"]))
    if result["event"] not in (0, 1) or result["sex"] not in (0, 1) or not 0 <= result["age_bucket"] <= 5:
        raise ValueError("Invalid Snapshot event values")
    for key in ("destination", "site_id"):
        if key in event:
            result[key] = event[key] if key == "destination" else int(event[key])
    if site is not None:
        result.setdefault("site_id", int(site["id"]))
        result.setdefault("destination", site.get("destination", site.get("bigquery_destination", "")))
    return result


def event_identity(event):
    return event.get("destination", ""), int(event["device_id"]), int(event["event_id"])


def event_order(event):
    return (_dt(event["timestamp"]), -int(event["event"]), event.get("destination", ""),
            int(event.get("site_id", 0)), int(event["device_id"]), int(event["event_id"]))


def merge_events(events, devices, start, end, site=None):
    horizons = {int(d["id"]): device_horizon(d) for d in devices if d["status"] == "enabled"}
    merged = {}
    for raw in events:
        event = normalize_event_dict(raw, site)
        instant = _dt(event["timestamp"])
        if event["device_id"] not in horizons or not start <= instant < end or instant >= horizons[event["device_id"]]:
            continue
        identity = event_identity(event)
        previous = merged.setdefault(identity, event)
        if previous != event:
            raise ValueError(f"Conflicting event identity: {identity!r}")
    return sorted(merged.values(), key=event_order)


def _machine_is_valid(machine, cursor):
    if machine.get("cursor_ts") != _stamp(cursor) or not validate_q15(machine):
        return False
    if not _calendar_shapes(machine) or int(machine["occupancy"]) < 0:
        return False
    fifo = [_dt(value) for value in machine["entry_fifo"]]
    return (fifo == sorted(fifo) and len(fifo) == int(machine["occupancy"])
            and (not fifo or fifo[0] + MAX_OPEN_VISIT >= cursor))


def validate_site_state(state, snapshot_ts, devices):
    required = {"engine_version", "view_until", "stable_until", "device_watermarks",
                "site_metadata", "stable_machine", "provisional_events", "pending_events",
                "current_machine"}
    if not isinstance(state, dict) or state.get("engine_version") != SITE_ENGINE_VERSION or not required <= set(state):
        return None
    try:
        latest = _dt(snapshot_ts)
        if state["view_until"] != _stamp(latest):
            return None
        enabled = [device for device in devices if device["status"] == "enabled"]
        if not _machine_is_valid(state["current_machine"], latest):
            return None
        if not enabled:
            if state["stable_until"] is not None or state["provisional_events"]:
                return None
            return state
        stable = _dt(state["stable_until"])
        if stable > latest or not _machine_is_valid(state["stable_machine"], stable):
            return None
        future_end = max(device_horizon(device) for device in enabled)
        retained = state["provisional_events"] + state["pending_events"]
        merged = merge_events(retained, devices, stable, future_end)
        if len(merged) != len(retained):
            return None
        if any((_dt(event["timestamp"]) < latest) != (event in state["provisional_events"])
               for event in merged):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return state


def classify_site(site, devices, snapshot, snapshot_now=None):
    previous = validate_site_state(snapshot["state"], snapshot["ts"], devices)
    if previous is None:
        return "REBUILD", None
    old = previous["device_watermarks"]
    latest = _dt(snapshot["ts"])
    snapshot_now = latest if snapshot_now is None else snapshot_now
    current = _device_watermarks(devices, snapshot_now)
    if previous["site_metadata"].get("destination") != _site_metadata(site)["destination"]:
        return "REBUILD", previous
    if set(old) != set(current):
        return "REBUILD", previous
    metadata_only = previous["site_metadata"] != _site_metadata(site)
    for key, value in current.items():
        before = old[key]
        if any(before.get(field) != value.get(field) for field in ("status", "created_at", "analysis_config")):
            return "REBUILD", previous
        if before.get("analyzed_until") and (not value.get("analyzed_until") or _dt(value["analyzed_until"]) < _dt(before["analyzed_until"])):
            return "REBUILD", previous
        if before.get("name") != value.get("name"):
            metadata_only = True
    source_changed = any(old[key].get("source_horizon") != value.get("source_horizon")
                         for key, value in current.items())
    if source_changed:
        return "INCREMENTAL", previous
    horizons = site_horizons(site, devices)
    effective_stable = None if horizons is None else min(horizons[1], snapshot_now)
    if effective_stable is not None and effective_stable > _dt(previous["stable_until"]):
        # A captured source horizon may be ahead of the prior wall clock.  As NOW
        # catches up, promote the already-fetched checkpoint without another read.
        return "PROMOTE_ONLY", previous
    if any(_dt(event["timestamp"]) < snapshot_now for event in previous["pending_events"]):
        return "PROMOTE_ONLY", previous
    if current != old or metadata_only:
        return "METADATA_ONLY", previous
    if snapshot_now != latest:
        return "TIME_ONLY", previous
    return "NO_OP", previous


def compute_site(site, devices, snapshot, classification, previous, supplied_events,
                 snapshot_now=None):
    snapshot_now = (_dt(snapshot["ts"]) if snapshot_now is None else snapshot_now)
    horizons = site_horizons(site, devices)
    start = _dt(site["created_at"])
    if horizons is None:
        if classification == "REBUILD":
            stable_machine = _machine(start, site)
            current = copy.deepcopy(stable_machine)
            advance(current, snapshot_now, site)
        else:
            stable_machine = copy.deepcopy(previous["stable_machine"])
            current = copy.deepcopy(previous["current_machine"])
            advance(current, snapshot_now, site)
        state = _build_state(None, devices, site, stable_machine, [], snapshot_now, current, [])
        return snapshot_now, derive_payload(current, site, devices, state), state

    stable = min(horizons[1], snapshot_now)
    if classification in ("TIME_ONLY", "METADATA_ONLY"):
        current = copy.deepcopy(previous["current_machine"])
        advance(current, snapshot_now, site)
        state = _build_state(_dt(previous["stable_until"]), devices, site,
                             copy.deepcopy(previous["stable_machine"]),
                             copy.deepcopy(previous["provisional_events"]), snapshot_now, current,
                             copy.deepcopy(previous["pending_events"]))
        return snapshot_now, derive_payload(current, site, devices, state), state

    if classification == "REBUILD":
        events = merge_events(supplied_events, devices, start, horizons[0], site)
        stable_machine = _machine(start, site)
    else:
        old_stable = _dt(previous["stable_until"])
        events = merge_events(previous["provisional_events"] + previous["pending_events"] + supplied_events,
                              devices, old_stable, horizons[0], site)
        stable_machine = copy.deepcopy(previous["stable_machine"])
    boundary = stable
    for event in events:
        if _dt(event["timestamp"]) < boundary:
            apply_event(stable_machine, event, site)
    # Strict expiry is essential: an expiry exactly at stable belongs to current replay.
    advance(stable_machine, boundary, site, expire_at_target=False)
    provisional = [event for event in events if boundary <= _dt(event["timestamp"]) < snapshot_now]
    pending = [event for event in events if _dt(event["timestamp"]) >= snapshot_now]
    current = copy.deepcopy(stable_machine)
    for event in provisional:
        apply_event(current, event, site)
    advance(current, snapshot_now, site, expire_at_target=True)
    state = _build_state(boundary, devices, site, stable_machine, provisional,
                         snapshot_now, current, pending)
    return snapshot_now, derive_payload(current, site, devices, state), state
