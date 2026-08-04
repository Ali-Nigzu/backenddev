"""Production Event and demographic assembly stage."""

import hashlib
import json


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
        self, event_batch, demographics_batch
    ):
        demographics_by_track = {
            result["track_id"]: (result["age"], result["sex"])
            for result in demographics_batch["results"]
        }
        rows = []

        for event_index, event in enumerate(event_batch["events"]):
            age, sex = demographics_by_track[event["track_id"]]
            rows.append(
                {
                    "event_id": _create_event_id(event, event_index),
                    "event": event["event_type"],
                    "timestamp": event["timestamp"],
                    "sex": sex,
                    "age_bucket": _age_to_bucket(age),
                }
            )
        return {"rows": rows}
