"""Append storage-ready OutputBatch rows to an existing BigQuery table."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from google.cloud import bigquery

_DESTINATION_PATTERN = re.compile(
    r"(?:^|!)1s(?P<project>[A-Za-z0-9_-]+)"
    r"!2s(?P<dataset>[A-Za-z0-9_]+)"
    r"!3s(?P<table>[A-Za-z0-9_]+)(?:!|$)"
)


def _parse_bigquery_link(bigquery_link: str) -> tuple[str, str, str]:
    parsed = urlparse(bigquery_link)
    workspace = parse_qs(parsed.query)["ws"][0]
    match = _DESTINATION_PATTERN.search(workspace)
    if match is None:
        raise ValueError("Unable to extract BigQuery project, dataset, and table")
    return match.group("project", "dataset", "table")


class Send:
    """Append final rows; insert IDs are only best-effort dedupe."""

    __slots__ = ()

    def __call__(self, output_batch: dict, bigquery_link: str) -> None:
        rows = output_batch["rows"]
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
