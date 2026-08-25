import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse



_DESTINATION = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
_LINK = re.compile(r"(?:^|!)1s(?P<project>[A-Za-z0-9_-]+)!2s(?P<dataset>[A-Za-z0-9_]+)!3s(?P<table>[A-Za-z0-9_]+)(?:!|$)")
_INSTANCE = "camosbase:europe-west2:camos-prod-postgres"
_DATABASE = "camos_prod"
_SITE_SQL = "SELECT id, name, organisation_id, bigquery_destination, max_capacity, timezone, created_at, updated_at FROM sites WHERE id = %s"
_DEVICE_SQL = "SELECT id, name, site_id, gcs_source_uri, status, analysis_interval_minutes, analysis_config, analyzed_until, created_at, updated_at FROM devices WHERE site_id = %s ORDER BY id"


def _stamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    keys = ("id", "name", "organisation_id", "bigquery_destination", "max_capacity", "timezone", "created_at", "updated_at")
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


def _events(client, table, devices, start, end):
    from google.cloud import bigquery
    ids = [device["id"] for device in devices]
    if not ids:
        return []
    sql = f"SELECT device_id, event_id, event, timestamp, sex, age_bucket FROM `{table}` WHERE device_id IN UNNEST(@device_ids) AND timestamp >= @start AND timestamp < @end ORDER BY timestamp ASC, event_id ASC"
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
    return sorted(deduplicated.values(), key=lambda event: (event["timestamp"], event["event_id"]))


def load_production_inputs(site_id, previous_ts):
    from google.cloud import bigquery
    credentials = _credentials()
    with _connection(credentials) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(_SITE_SQL, (site_id,))
            site = _site(cursor.fetchone())
            cursor.execute(_DEVICE_SQL, (site_id,))
            devices = _devices(cursor.fetchall())
        finally:
            cursor.close()
    horizons = [device["analyzed_until"] for device in devices if device["status"] == "enabled" and device["analyzed_until"]]
    end = max(horizons) if horizons else previous_ts
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    return site, devices, _events(client, _destination(site["bigquery_destination"]), devices, site["created_at"], end)

