"""Production Event and demographic assembly stage."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_TIMEFRAME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_UK_TIMEZONE = ZoneInfo("Europe/London")


def _age_to_bucket(age: int) -> int:
    if age <= 4:
        return 0
    if age <= 13:
        return 1
    if age <= 25:
        return 2
    if age <= 45:
        return 3
    if age <= 65:
        return 4
    return 5


def _parse_utc_timeframe_start(value: str) -> datetime:
    return datetime.strptime(
        value,
        _TIMEFRAME_FORMAT,
    ).replace(tzinfo=timezone.utc)


def _create_event_id(event, event_index: int) -> str:
    values = [
        event["track_id"],
        int(event["event_type"]),
        float(event["timestamp"]).hex(),
    ]
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"event-{event_index}-{digest}"


class Assemble:
    """Join Event and demographic batches into final output rows."""

    __slots__ = ()

    def __call__(
        self,
        event_batch: dict,
        demographics_batch: dict,
        timeframe_start: str,
    ) -> dict:
        events = event_batch["events"]
        if not events:
            return {"rows": []}

        timeframe_start_utc = _parse_utc_timeframe_start(timeframe_start)
        demographics_by_track = {
            result["track_id"]: (result["age"], result["sex"])
            for result in demographics_batch["results"]
        }
        rows = []

        for event_index, event in enumerate(events):
            absolute_utc = timeframe_start_utc + timedelta(
                seconds=float(event["timestamp"])
            )
            absolute_uk = absolute_utc.astimezone(_UK_TIMEZONE)
            age, sex = demographics_by_track[event["track_id"]]
            rows.append(
                {
                    "event_id": _create_event_id(event, event_index),
                    "event": event["event_type"],
                    "ts": absolute_uk.isoformat(timespec="milliseconds"),
                    "sex": sex,
                    "age_bucket": _age_to_bucket(age),
                }
            )
        return {"rows": rows}
