"""Append storage-ready OutputBatch rows to an existing BigQuery table."""

from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from google.cloud import bigquery

_ROW_FIELDS = {
    "device_id",
    "event_id",
    "event",
    "timestamp",
    "sex",
    "age_bucket",
}
_INTEGER_FIELDS = ("device_id", "event_id", "event", "sex", "age_bucket")
_DESTINATION_PATTERN = re.compile(
    r"(?:^|!)1s(?P<project>[A-Za-z0-9_-]+)"
    r"!2s(?P<dataset>[A-Za-z0-9_]+)"
    r"!3s(?P<table>[A-Za-z0-9_]+)(?:!|$)"
)
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)


def _parse_bigquery_link(bigquery_link: str) -> tuple[str, str, str]:
    if not isinstance(bigquery_link, str):
        raise ValueError("BigQuery destination must be a console URL string")
    parsed = urlparse(bigquery_link)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "console.cloud.google.com"
        or parsed.path != "/bigquery"
    ):
        raise ValueError("Expected a supported BigQuery console URL")
    workspace_values = parse_qs(parsed.query, keep_blank_values=True).get("ws", [])
    if len(workspace_values) != 1:
        raise ValueError("BigQuery console URL must contain exactly one ws destination")
    match = _DESTINATION_PATTERN.search(workspace_values[0])
    if match is None:
        raise ValueError("Unable to extract BigQuery project, dataset, and table")
    return match.group("project", "dataset", "table")


def _validate_timestamp(value: object, row_index: int) -> None:
    if not isinstance(value, str) or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"OutputBatch.rows[{row_index}].timestamp must be UTC ISO milliseconds ending in Z"
        )
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(
            f"OutputBatch.rows[{row_index}].timestamp is not a valid timestamp"
        ) from exc


def _validate_row(row: object, row_index: int) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"OutputBatch.rows[{row_index}] must be an object")
    if set(row) != _ROW_FIELDS:
        missing = sorted(_ROW_FIELDS - set(row))
        extra = sorted(set(row) - _ROW_FIELDS)
        raise ValueError(
            f"OutputBatch.rows[{row_index}] has invalid fields; "
            f"missing={missing}, extra={extra}"
        )
    for field in _INTEGER_FIELDS:
        if type(row[field]) is not int:
            raise ValueError(
                f"OutputBatch.rows[{row_index}].{field} must be an integer"
            )
    if not 1 <= row["event_id"] <= 2**63 - 1:
        raise ValueError(
            f"OutputBatch.rows[{row_index}].event_id must be a positive signed INT64"
        )
    _validate_timestamp(row["timestamp"], row_index)


def _validate_output_batch(output_batch: object) -> list[dict]:
    if not isinstance(output_batch, dict):
        raise ValueError("OutputBatch must be an object")
    if set(output_batch) != {"rows"}:
        raise ValueError("OutputBatch must contain exactly the rows field")
    rows = output_batch["rows"]
    if not isinstance(rows, list):
        raise ValueError("OutputBatch.rows must be a list")
    for row_index, row in enumerate(rows):
        _validate_row(row, row_index)
    return rows


class Send:
    """Validate and append final rows; insert IDs are only best-effort dedupe."""

    __slots__ = ()

    def __call__(self, output_batch: dict, bigquery_link: str) -> None:
        rows = _validate_output_batch(output_batch)
        if not rows:
            return None

        project, dataset, table = _parse_bigquery_link(bigquery_link)
        table_reference = f"{project}.{dataset}.{table}"
        client = bigquery.Client(project=project)
        errors = client.insert_rows_json(
            table_reference,
            rows,
            row_ids=[str(row["event_id"]) for row in rows],
            skip_invalid_rows=False,
            ignore_unknown_values=False,
        )
        if errors:
            raise RuntimeError(f"BigQuery rejected one or more rows: {errors!r}")
        return None
