import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from .inputs import load_production_inputs
except ImportError:
    from inputs import load_production_inputs


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "local"
Q15 = timedelta(minutes=15)


def _zone(site):
    try:
        return ZoneInfo(site["timezone"])
    except Exception:
        return timezone(timedelta(hours=1))


def _dt(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _stamp(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(name):
    with (LOCAL / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(name, value):
    with (LOCAL / name).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


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


def _machine(start, site):
    local = start.astimezone(_zone(site))
    monday = (local - timedelta(days=local.weekday())).date()
    q_start = start - timedelta(minutes=15 * 95)
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


def _device_state(devices):
    return {str(d["id"]): {"name": d["name"], "status": d["status"], "created_at": d["created_at"], "analyzed_until": d["analyzed_until"]} for d in devices}


def initial_state(site, devices):
    start = _dt(site["created_at"])
    machine = _machine(start, site)
    return {"stable_until": _stamp(start), "devices": _device_state(devices), "recent_events": [], "stable_machine": copy.deepcopy(machine), "current_machine": machine}


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
    zone = _zone(site)
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
    zone = _zone(site)
    local = instant.astimezone(zone)
    q = machine["q15"]
    index = _q_index(machine, instant)
    local_date = local.date().isoformat()
    day = local.weekday()
    week_index = int((local.date() - datetime.strptime(machine["month"]["first_iso_week_start"], "%Y-%m-%d").date()).days // 7)
    quarter_index = int((local.date() - datetime.strptime(machine["quarter"]["first_iso_week_start"], "%Y-%m-%d").date()).days // 7)
    return [(q, index), (machine["today"] if local_date == machine["today"]["local_date"] else machine["yesterday"], local.hour), (machine["week"], day), (machine["month"], max(0, min(3, week_index))), (machine["quarter"], max(0, min(11, quarter_index))), (machine["year"], local.month - 1), (machine["all_time"], local.year - machine["all_time"]["start_year"])]


def _add_occupancy(block, index, occupancy, seconds):
    block["occupancy_area_person_seconds"][index] += occupancy * seconds
    block["occupancy_seconds"][index] += seconds
    if occupancy > 0:
        current = block["occupancy_min_positive"][index]
        block["occupancy_min_positive"][index] = occupancy if current == 0 else min(current, occupancy)
    block["occupancy_max"][index] = max(block["occupancy_max"][index], occupancy)


def _next_boundary(value, target, site):
    zone = _zone(site)
    values = [target]
    quarter = datetime.fromtimestamp(((int(value.timestamp()) // 900) + 1) * 900, tz=timezone.utc)
    values.append(quarter)
    local = value.astimezone(zone)
    next_hour = (local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).astimezone(timezone.utc)
    values.append(next_hour)
    return min(v for v in values if v > value)


def _refresh_peaks(machine):
    maxima = machine["q15"]["occupancy_max"]
    machine["q15"]["rolling_peak_occupancy"] = [max(maxima[max(0, i - 95):i + 1]) for i in range(96)]


def _ensure_q15_interval(machine, instant):
    q_end = _dt(machine["q15"]["window_start"]) + timedelta(minutes=96 * 15)
    if instant >= q_end:
        _roll_q15(machine)


def advance(machine, target, site):
    cursor = _dt(machine["cursor_ts"])
    while cursor < target:
        _ensure_q15_interval(machine, cursor)
        boundary = _next_boundary(cursor, target, site)
        elapsed = int((boundary - cursor).total_seconds())
        for block, index in _active(machine, cursor, site):
            _add_occupancy(block, index, machine["occupancy"], elapsed)
        cursor = boundary
        machine["cursor_ts"] = _stamp(cursor)
        _roll_calendar(machine, cursor, site)
        if cursor < target and cursor.minute % 15 == 0 and cursor.second == 0:
            _roll_q15(machine)
    _refresh_peaks(machine)


def _increment(block, key, index):
    block[key][index] += 1


def apply_event(machine, event, site):
    instant = _dt(event["timestamp"])
    advance(machine, instant, site)
    _ensure_q15_interval(machine, instant)
    active = _active(machine, instant, site)
    q, q_index = active[0]
    if event["event"] == 1:
        machine["occupancy"] += 1
        machine["entry_fifo"].append(event["timestamp"])
        for block, index in active:
            _increment(block, "entrances", index)
        zone = _zone(site)
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
            q["dwell_sum_seconds"][q_index] += int((instant - entered).total_seconds())
            q["dwell_count"][q_index] += 1
        machine["occupancy"] = max(0, machine["occupancy"] - 1)
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
            current_ids.setdefault(int(device_id), state["devices"][device_id]["name"])
    axes = [{"device_id": device_id, "name": current_ids[device_id]} for device_id in sorted(current_ids)]
    traffic = []
    for values in q["traffic_counts"]:
        total = sum(values.values())
        traffic.append([0.0 if total == 0 else values.get(str(axis["device_id"]), 0) * 100 / total for axis in axes])
    peak = q["rolling_peak_occupancy"]
    today = _rollup(machine["today"], machine["today"]["age_counts"], machine["today"]["sex_counts"])
    yesterday = _rollup(machine["yesterday"], machine["yesterday"]["age_counts"], machine["yesterday"]["sex_counts"])
    week = _rollup(machine["week"], [sum(row[i] for row in machine["week"]["age_counts_by_day"]) for i in range(6)], [sum(row[i] for row in machine["week"]["sex_counts_by_day"]) for i in range(2)])
    month = _rollup(machine["month"], [sum(row[i] for row in machine["month"]["age_counts_by_week"]) for i in range(6)], [sum(row[i] for row in machine["month"]["sex_counts_by_week"]) for i in range(2)])
    quarter = _rollup(machine["quarter"], [sum(row[i] for row in machine["quarter"]["age_counts_by_week"]) for i in range(6)], [sum(row[i] for row in machine["quarter"]["sex_counts_by_week"]) for i in range(2)])
    year = _rollup(machine["year"], [sum(row[i] for row in machine["year"]["age_counts_by_month"]) for i in range(6)], [sum(row[i] for row in machine["year"]["sex_counts_by_month"]) for i in range(2)])
    all_time = _rollup(machine["all_time"], machine["all_time"]["age_counts"], machine["all_time"]["sex_counts"])
    averages = [0.0 if seconds == 0 else area / seconds for area, seconds in zip(q["occupancy_area_person_seconds"], q["occupancy_seconds"])]
    return {"entrances_96": q["entrances"], "occupancy_96": averages, "exits_96": q["exits"], "footfall_96": [a + b for a, b in zip(q["entrances"], q["exits"])], "dwell_time_96": [0.0 if count == 0 else total / count for total, count in zip(q["dwell_sum_seconds"], q["dwell_count"])], "traffic_devices": axes, "traffic_split_96": traffic, "capacity": [[average * 100 / site["max_capacity"], hard * 100 / site["max_capacity"]] for average, hard in zip(averages, peak)], "today": today, "yesterday": yesterday, "week": week, "month": month, "quarter": quarter, "year": year, "all_time": all_time}


def _horizons(previous, devices, site):
    enabled = [d for d in devices if d["status"] == "enabled"]
    horizons = [_dt(d["analyzed_until"]) for d in enabled if d["analyzed_until"]]
    ts = max(horizons) if horizons else _dt(previous["ts"])
    safe = [_dt(d["analyzed_until"]) if d["analyzed_until"] else _dt(d["created_at"]) for d in enabled]
    stable = max(_dt(previous["state"]["stable_until"]), min(safe)) if safe else _dt(previous["state"]["stable_until"])
    return ts, stable


def Snapshot(site_id):
    try:
        previous = _read("snapshot.json")
        if previous["site_id"] != site_id:
            raise ValueError("snapshot not found")
        site, devices, loaded_events = load_production_inputs(site_id, previous["ts"])
        ts, stable = _horizons(previous, devices, site)
        horizons = {d["id"]: _dt(d["analyzed_until"]) for d in devices if d["analyzed_until"]}
        events = [e for e in loaded_events if e["device_id"] in horizons and _dt(e["timestamp"]) < horizons[e["device_id"]] and _dt(e["timestamp"]) < ts]
        events.sort(key=lambda e: (e["timestamp"], e["event_id"]))
        start = _dt(site["created_at"])
        stable_machine = _machine(start, site)
        for event in events:
            if _dt(event["timestamp"]) < stable:
                apply_event(stable_machine, event, site)
        advance(stable_machine, stable, site)
        recent = [e for e in events if stable <= _dt(e["timestamp"]) < ts]
        current_machine = copy.deepcopy(stable_machine)
        for event in recent:
            apply_event(current_machine, event, site)
        advance(current_machine, ts, site)
        state = {"stable_until": _stamp(stable), "devices": _device_state(devices), "recent_events": recent, "stable_machine": stable_machine, "current_machine": current_machine}
        row = {"site_id": site_id, "ts": _stamp(ts), "payload": derive_payload(current_machine, site, devices, state), "state": state, "updated_at": _stamp(ts)}
        _write("snapshot.json", row)
        return True
    except Exception:
        return False

