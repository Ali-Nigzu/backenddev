"""Production Event and demographic assembly stage."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

_TIMEFRAME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_TEMPORARY_DEVICE_ID = 0
_MAX_SIGNED_INT64 = 2**63 - 1


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


def _create_event_id(identity: tuple, duplicate_occurrence: int | None) -> int:
    values = list(identity)
    if duplicate_occurrence is not None:
        values.append(duplicate_occurrence)
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    unsigned_64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return unsigned_64 % _MAX_SIGNED_INT64 + 1


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
        prepared_events = []

        for event in events:
            absolute_utc = timeframe_start_utc + timedelta(
                seconds=float(event["timestamp"])
            )
            age, sex = demographics_by_track[event["track_id"]]
            timestamp = absolute_utc.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            event_value = int(event["event_type"])
            sex_value = int(sex)
            age_bucket = int(_age_to_bucket(age))
            identity = (
                _TEMPORARY_DEVICE_ID,
                timestamp,
                event_value,
                sex_value,
                age_bucket,
                str(event["track_id"]),
            )
            prepared_events.append(
                (identity, timestamp, event_value, sex_value, age_bucket)
            )

        identity_counts = Counter(item[0] for item in prepared_events)
        identity_occurrences = defaultdict(int)
        rows = []
        for identity, timestamp, event_value, sex_value, age_bucket in prepared_events:
            duplicate_occurrence = None
            if identity_counts[identity] > 1:
                duplicate_occurrence = identity_occurrences[identity]
                identity_occurrences[identity] += 1
            rows.append(
                {
                    "device_id": _TEMPORARY_DEVICE_ID,
                    "event_id": _create_event_id(identity, duplicate_occurrence),
                    "event": event_value,
                    "timestamp": timestamp,
                    "sex": sex_value,
                    "age_bucket": age_bucket,
                }
            )
        return {"rows": rows}
