import copy
import json
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse


_DESTINATION = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
_LINK = re.compile(r"(?:^|!)1s(?P<project>[A-Za-z0-9_-]+)!2s(?P<dataset>[A-Za-z0-9_]+)!3s(?P<table>[A-Za-z0-9_]+)(?:!|$)")
_INSTANCE = "camosbase:europe-west2:camos-prod-postgres"
_DATABASE = "camos_prod"
_SITE_SQL = "SELECT id, name, organisation_id, bigquery_destination, max_capacity, created_at, updated_at FROM public.sites WHERE id = %s"
_DEVICE_SQL = "SELECT id, name, site_id, gcs_source_uri, status, analysis_interval_minutes, analysis_config, analyzed_until, created_at, updated_at FROM public.devices WHERE site_id = %s ORDER BY id"
_SNAPSHOT_SQL = "SELECT site_id, ts, payload, state, updated_at FROM public.snapshots WHERE site_id = %s"
_SNAPSHOT_UPDATE_SQL = "UPDATE public.snapshots SET ts = %s, payload = %s::jsonb, state = %s::jsonb, updated_at = CURRENT_TIMESTAMP WHERE site_id = %s AND updated_at = %s RETURNING updated_at"



def _credentials():
    from google.oauth2 import service_account
    path = Path(__file__).resolve().parent.parent / "sa.json"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot service account file not found: {path}")
    return service_account.Credentials.from_service_account_file(str(path))


@contextmanager
def _connection(credentials) -> Iterator:
    from google.cloud.sql.connector import Connector
    email = credentials.service_account_email
    if not email.endswith(".gserviceaccount.com"):
        raise RuntimeError("Snapshot credentials must be a service account")
    with Connector(credentials=credentials) as connector:
        connection = connector.connect(_INSTANCE, "pg8000", user=email.removesuffix(".gserviceaccount.com"), db=_DATABASE, enable_iam_auth=True)
        try:
            yield connection
        finally:
            connection.close()


def _destination(value):
    if _DESTINATION.fullmatch(value):
        return value
    workspace = parse_qs(urlparse(value).query).get("ws", [""])[0]
    match = _LINK.search(workspace)
    if not match:
        raise ValueError("Invalid BigQuery destination")
    return ".".join(match.group("project", "dataset", "table"))


def _site(row):
    if row is None:
        raise ValueError("Snapshot site not found")
    keys = ("id", "name", "organisation_id", "bigquery_destination", "max_capacity", "created_at", "updated_at")
    result = dict(zip(keys, row))
    if result["max_capacity"] is None or result["max_capacity"] <= 0:
        raise ValueError("Snapshot site max_capacity must be positive")
    _destination(result["bigquery_destination"])
    result["created_at"] = _stamp(result["created_at"])
    result["updated_at"] = _stamp(result["updated_at"])
    return result


def _devices(rows):
    keys = ("id", "name", "site_id", "gcs_source_uri", "status", "analysis_interval_minutes", "analysis_config", "analyzed_until", "created_at", "updated_at")
    result = []
    for row in rows:
        device = dict(zip(keys, row))
        if isinstance(device["analysis_config"], str):
            device["analysis_config"] = json.loads(device["analysis_config"])
        for key in ("analyzed_until", "created_at", "updated_at"):
            if device[key] is not None:
                device[key] = _stamp(device[key])
        result.append(device)
    return result


def _json_object(value, field):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"Snapshot {field} must be a JSON object")
    return value


def _snapshot(row):
    if row is None:
        raise ValueError("Snapshot row not found")
    keys = ("site_id", "ts", "payload", "state", "updated_at")
    result = dict(zip(keys, row))
    result["state"] = _json_object(result["state"], "state")
    result["ts"] = _stamp(result["ts"])
    result["version"] = result.pop("updated_at")
    return result


def _events(client, table, devices, start, end):
    from google.cloud import bigquery
    if start >= end:
        return []
    ids = [device["id"] for device in devices]
    if not ids:
        return []
    sql = f"SELECT device_id, event_id, event, timestamp, sex, age_bucket FROM `{table}` WHERE device_id IN UNNEST(@device_ids) AND timestamp >= @start AND timestamp < @end ORDER BY timestamp ASC, event DESC, event_id ASC"
    config = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("device_ids", "INT64", ids), bigquery.ScalarQueryParameter("start", "TIMESTAMP", start), bigquery.ScalarQueryParameter("end", "TIMESTAMP", end)])
    horizons = {device["id"]: device["analyzed_until"] for device in devices if device["analyzed_until"]}
    deduplicated = {}
    for row in client.query(sql, job_config=config).result():
        event = {"device_id": int(row["device_id"]), "event_id": int(row["event_id"]), "event": int(row["event"]), "timestamp": _stamp(row["timestamp"]), "sex": int(row["sex"]), "age_bucket": int(row["age_bucket"])}
        if event["device_id"] not in horizons or event["timestamp"] >= horizons[event["device_id"]]:
            continue
        previous = deduplicated.setdefault(event["event_id"], event)
        if previous != event:
            raise ValueError(f"Conflicting event_id: {event['event_id']}")
    return sorted(deduplicated.values(), key=lambda event: (event["timestamp"], -event["event"], event["event_id"]))


def load_production_inputs(site_id):
    """Load only the persistent Snapshot context; BigQuery is queried on demand."""
    credentials = _credentials()
    with _connection(credentials) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(_SITE_SQL, (site_id,))
            site = _site(cursor.fetchone())
            cursor.execute(_DEVICE_SQL, (site_id,))
            devices = _devices(cursor.fetchall())
            cursor.execute(_SNAPSHOT_SQL, (site_id,))
            snapshot = _snapshot(cursor.fetchone())
        finally:
            cursor.close()
    if site["id"] != site_id or snapshot["site_id"] != site_id:
        raise ValueError(f"Snapshot context does not match site_id={site_id}")
    return credentials, site, devices, snapshot


def _load_events(credentials, site, devices, start, end):
    from google.cloud import bigquery
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    return _events(client, _destination(site["bigquery_destination"]), devices, start, end)

class _ConcurrentSnapshotUpdate(RuntimeError):
    pass


def _persist(credentials, site_id, version, ts, payload, state):
    payload_json = json.dumps(payload)
    state_json = json.dumps(state)
    with _connection(credentials) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                _SNAPSHOT_UPDATE_SQL,
                (_dt(ts), payload_json, state_json, site_id, version),
            )
            if cursor.fetchone() is None:
                raise _ConcurrentSnapshotUpdate(f"Concurrent Snapshot update for site_id={site_id}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()


ENGINE_VERSION = 2
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


def _machine(start, site):
    local = start.astimezone(_zone())
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


def _device_watermarks(devices):
    return {str(d["id"]): {"name": d["name"], "status": d["status"], "created_at": d["created_at"], "analyzed_until": d["analyzed_until"]} for d in devices}


def _site_metadata(site):
    return {"max_capacity": site["max_capacity"]}


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


def _build_state(stable, devices, site, stable_machine, provisional_events):
    return {"engine_version": ENGINE_VERSION, "stable_until": _stamp(stable), "device_watermarks": _device_watermarks(devices), "site_metadata": _site_metadata(site), "stable_machine": stable_machine, "provisional_events": provisional_events}

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


def _full_rebuild(credentials, site, devices):
    start = _dt(site["created_at"])
    ts, stable = _horizons(start, start, devices)
    events = _merge_events(_load_events(credentials, site, devices, start, ts), devices, start, ts)
    stable_machine = _machine(start, site)
    for event in events:
        if _dt(event["timestamp"]) < stable:
            apply_event(stable_machine, event, site)
    advance(stable_machine, stable, site)
    provisional = [event for event in events if stable <= _dt(event["timestamp"]) < ts]
    current = _current_from(stable_machine, provisional, ts, site)
    state = _build_state(stable, devices, site, stable_machine, provisional)
    return _stamp(ts), derive_payload(current, site, devices, state), state


def _incremental(credentials, site, devices, snapshot, previous):
    previous_ts, previous_stable = _dt(snapshot["ts"]), _dt(previous["stable_until"])
    ts, stable = _horizons(previous_ts, previous_stable, devices)
    watermarks, metadata = _device_watermarks(devices), _site_metadata(site)
    if ts < previous_ts or stable < previous_stable:
        return _full_rebuild(credentials, site, devices)
    if ts == previous_ts and stable == previous_stable and watermarks == previous["device_watermarks"] and metadata == previous["site_metadata"]:
        return None
    horizons_changed = any(
        previous["device_watermarks"][device_id].get("analyzed_until")
        != watermark.get("analyzed_until")
        for device_id, watermark in watermarks.items()
    )
    if ts == previous_ts and stable == previous_stable and not horizons_changed:
        current = _current_from(previous["stable_machine"], previous["provisional_events"], ts, site)
        state = _build_state(stable, devices, site, previous["stable_machine"], previous["provisional_events"])
        return _stamp(ts), derive_payload(current, site, devices, state), state
    fetched = _load_events(credentials, site, devices, previous_stable, ts)
    events = _merge_events(previous["provisional_events"] + fetched, devices, previous_stable, ts)
    stable_machine = copy.deepcopy(previous["stable_machine"])
    for event in events:
        if _dt(event["timestamp"]) < stable:
            apply_event(stable_machine, event, site)
    advance(stable_machine, stable, site)
    provisional = [event for event in events if stable <= _dt(event["timestamp"]) < ts]
    current = _current_from(stable_machine, provisional, ts, site)
    state = _build_state(stable, devices, site, stable_machine, provisional)
    return _stamp(ts), derive_payload(current, site, devices, state), state


def Snapshot(site_id):
    for attempt in range(MAX_RETRIES + 1):
        credentials, site, devices, snapshot = load_production_inputs(site_id)
        previous = _state_v2(snapshot["state"])
        if previous is None or not _provisional_is_safe(previous, snapshot["ts"]) or not _membership_is_safe(previous["device_watermarks"], _device_watermarks(devices)):
            result = _full_rebuild(credentials, site, devices)
        else:
            result = _incremental(credentials, site, devices, snapshot, previous)
        if result is None:
            return True
        ts, payload, state = result
        try:
            _persist(credentials, site_id, snapshot["version"], ts, payload, state)
            return True
        except _ConcurrentSnapshotUpdate:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Snapshot update repeatedly conflicted for site_id={site_id}") from None
    raise AssertionError("unreachable")
