from google.cloud import bigquery

_PROJECT = "camosbase"
_TABLE = "camosbase.camos_prod.events"

class Send:

    __slots__ = ()

    def __call__(self, output_batch: dict) -> None:
        rows = output_batch["rows"]
        if not rows:
            return None

        client = bigquery.Client(project=_PROJECT)
        errors = client.insert_rows_json(
            _TABLE,
            rows,
            row_ids=[str(row["event_id"]) for row in rows],
        )
        if errors:
            raise RuntimeError(f"BigQuery rejected one or more rows: {errors!r}")
        return None
