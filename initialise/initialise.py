"""Resolve device processing context from PostgreSQL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from initialise.cloud_sql import cloud_sql_connection

_TIMEFRAME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_FIRST_ANALYSIS_START = "2026-08-01T00:00:00.000Z"
_DEVICE_CONTEXT_QUERY = """
SELECT
    devices.id,
    devices.gcs_source_uri,
    devices.analysis_config,
    devices.analyzed_until,
    sites.bigquery_destination
FROM devices
JOIN sites ON sites.id = devices.site_id
WHERE devices.id = %s
"""


def _format_utc_timestamp(value: datetime) -> str:
    utc_value = value.astimezone(timezone.utc)
    return utc_value.strftime(_TIMEFRAME_FORMAT)[:-4] + "Z"


def _decode_analysis_config(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def initialise(device_id: int) -> dict:
    """Return the read-only processing context for a PostgreSQL device ID."""
    timeframe_end = _format_utc_timestamp(datetime.now(timezone.utc))

    try:
        with cloud_sql_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(_DEVICE_CONTEXT_QUERY, (device_id,))
                row = cursor.fetchone()
            finally:
                cursor.close()
    except Exception as exc:
        raise RuntimeError(f"Unable to read processing context for device {device_id}") from exc

    if row is None:
        raise ValueError(f"Device not found: {device_id}")

    resolved_device_id, source_uri, analysis_config, analyzed_until, destination = row
    timeframe_start = (
        _format_utc_timestamp(analyzed_until)
        if analyzed_until is not None
        else _FIRST_ANALYSIS_START
    )
    return {
        "device_id": resolved_device_id,
        "gcs_source_uri": source_uri,
        "analysis_config": _decode_analysis_config(analysis_config),
        "timeframe": {"start": timeframe_start, "end": timeframe_end},
        "bigquery_destination": destination,
    }
