"""Resolve device processing context from PostgreSQL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import google.auth
from google.cloud.sql.connector import Connector

_TIMEFRAME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_FIRST_ANALYSIS_START = "2026-08-01T00:00:00.000Z"
_CLOUD_SQL_INSTANCE = "camosbase:europe-west2:camos-prod-postgres"
_CLOUD_SQL_DATABASE = "camos_prod"
_SERVICE_ACCOUNT_DOMAIN = ".gserviceaccount.com"
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


def _iam_database_user() -> str:
    credentials, _project_id = google.auth.default()
    service_account_email = getattr(
        credentials, "service_account_email", None
    ) or getattr(credentials, "signer_email", None)
    if not service_account_email or not service_account_email.endswith(
        _SERVICE_ACCOUNT_DOMAIN
    ):
        raise RuntimeError(
            "Application Default Credentials must identify a service account"
        )
    return service_account_email.removesuffix(_SERVICE_ACCOUNT_DOMAIN)


def initialise(device_id: int) -> dict:
    """Return the read-only processing context for a PostgreSQL device ID."""
    timeframe_end = _format_utc_timestamp(datetime.now(timezone.utc))
    iam_database_user = _iam_database_user()

    try:
        with Connector() as connector:
            connection = connector.connect(
                _CLOUD_SQL_INSTANCE,
                "pg8000",
                user=iam_database_user,
                db=_CLOUD_SQL_DATABASE,
                enable_iam_auth=True,
            )
            try:
                cursor = connection.cursor()
                try:
                    cursor.execute(_DEVICE_CONTEXT_QUERY, (device_id,))
                    row = cursor.fetchone()
                finally:
                    cursor.close()
            finally:
                connection.close()
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
