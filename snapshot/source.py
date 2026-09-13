"""Canonical BigQuery source range planning."""

from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

from .site_engine import event_identity, event_order, stamp


TABLE = "camosbase.camos_prod.events"


class SourceRange(NamedTuple):
    organisation_id: int
    site_id: int
    device_id: int
    start: datetime
    end: datetime


def coalesce_ranges(ranges):
    grouped = defaultdict(list)
    for item in ranges:
        if item.start < item.end:
            grouped[(item.organisation_id, item.site_id, item.device_id)].append(item)
    result = []
    for (organisation_id, site_id, device_id), values in grouped.items():
        values.sort(key=lambda value: value.start)
        start, end = values[0].start, values[0].end
        for value in values[1:]:
            if value.start <= end:
                end = max(end, value.end)
            else:
                result.append(
                    SourceRange(organisation_id, site_id, device_id, start, end)
                )
                start, end = value.start, value.end
        result.append(SourceRange(organisation_id, site_id, device_id, start, end))
    return sorted(
        result,
        key=lambda value: (
            value.organisation_id,
            value.site_id,
            value.device_id,
            value.start,
        ),
    )


def fetch_events(client, ranges):
    from google.cloud import bigquery

    clauses, parameters = [], []
    for index, item in enumerate(coalesce_ranges(ranges)):
        clauses.append(
            f"(organisation_id=@organisation_{index} AND site_id=@site_{index} "
            f"AND device_id=@device_{index} AND timestamp>=@start_{index} "
            f"AND timestamp<@end_{index})"
        )
        parameters.extend(
            [
                bigquery.ScalarQueryParameter(
                    f"organisation_{index}", "INT64", item.organisation_id
                ),
                bigquery.ScalarQueryParameter(f"site_{index}", "INT64", item.site_id),
                bigquery.ScalarQueryParameter(
                    f"device_{index}", "INT64", item.device_id
                ),
                bigquery.ScalarQueryParameter(f"start_{index}", "TIMESTAMP", item.start),
                bigquery.ScalarQueryParameter(f"end_{index}", "TIMESTAMP", item.end),
            ]
        )
    sql = (
        "SELECT organisation_id,site_id,device_id,event_id,event,timestamp,sex,age_bucket "
        f"FROM `{TABLE}` WHERE {' OR '.join(clauses)} "
        "ORDER BY timestamp,event DESC,organisation_id,site_id,device_id,event_id"
    )
    config = bigquery.QueryJobConfig(query_parameters=parameters)
    merged = {}
    for row in client.query(sql, job_config=config).result():
        event = {
            "organisation_id": int(row["organisation_id"]),
            "site_id": int(row["site_id"]),
            "device_id": int(row["device_id"]),
            "event_id": str(row["event_id"]),
            "event": int(row["event"]),
            "timestamp": stamp(row["timestamp"]),
            "sex": int(row["sex"]),
            "age_bucket": int(row["age_bucket"]),
        }
        merged.setdefault(event_identity(event), event)
    return sorted(merged.values(), key=event_order)
