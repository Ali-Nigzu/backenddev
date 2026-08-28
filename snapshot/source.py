"""BigQuery source range planning and single-pass event normalization."""

from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

from .site_engine import event_identity, event_order, normalize_event_dict, stamp


class SourceRange(NamedTuple):
    destination: str
    site_id: int
    device_id: int
    start: datetime
    end: datetime


def coalesce_ranges(ranges):
    grouped = defaultdict(list)
    for item in ranges:
        if item.start < item.end:
            grouped[(item.destination, item.site_id, item.device_id)].append(item)
    result = []
    for (destination, site_id, device_id), values in grouped.items():
        values.sort(key=lambda value: value.start)
        start, end = values[0].start, values[0].end
        for value in values[1:]:
            if value.start <= end:
                end = max(end, value.end)
            else:
                result.append(SourceRange(destination, site_id, device_id, start, end))
                start, end = value.start, value.end
        result.append(SourceRange(destination, site_id, device_id, start, end))
    return sorted(result, key=lambda value: (value.destination, value.device_id, value.start))


def fetch_events(client, ranges):
    from google.cloud import bigquery
    by_destination = defaultdict(list)
    for item in coalesce_ranges(ranges):
        by_destination[item.destination].append(item)
    merged = {}
    for destination, values in sorted(by_destination.items()):
        clauses, parameters = [], []
        route = {}
        for index, item in enumerate(values):
            clauses.append(f"(device_id=@device_{index} AND timestamp>=@start_{index} AND timestamp<@end_{index})")
            parameters.extend([
                bigquery.ScalarQueryParameter(f"device_{index}", "INT64", item.device_id),
                bigquery.ScalarQueryParameter(f"start_{index}", "TIMESTAMP", item.start),
                bigquery.ScalarQueryParameter(f"end_{index}", "TIMESTAMP", item.end),
            ])
            route.setdefault(item.device_id, []).append(item)
        sql = (f"SELECT device_id,event_id,event,timestamp,sex,age_bucket FROM `{destination}` "
               f"WHERE {' OR '.join(clauses)} ORDER BY timestamp,event DESC,device_id,event_id")
        config = bigquery.QueryJobConfig(query_parameters=parameters)
        for row in client.query(sql, job_config=config).result():
            device_id = int(row["device_id"])
            timestamp = row["timestamp"]
            candidates = route.get(device_id, ())
            matching = candidates if len(candidates) == 1 else [
                item for item in candidates if item.start <= timestamp < item.end]
            if not matching:
                continue
            site_ids = {item.site_id for item in matching}
            if len(site_ids) != 1:
                raise ValueError(f"Device {device_id} routed to multiple sites")
            raw = {"destination": destination, "site_id": site_ids.pop(), "device_id": device_id,
                   "event_id": row["event_id"], "event": row["event"], "timestamp": stamp(timestamp),
                   "sex": row["sex"], "age_bucket": row["age_bucket"]}
            event = normalize_event_dict(raw)
            identity = event_identity(event)
            previous = merged.setdefault(identity, event)
            if previous != event:
                raise ValueError(f"Conflicting event identity: {identity!r}")
    return sorted(merged.values(), key=event_order)
