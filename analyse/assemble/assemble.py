from datetime import timedelta

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

class Assemble:

    __slots__ = ()

    def __call__(
        self,
        event_batch: dict,
        demographics_batch: dict,
        source_origin,
        organisation_id: int,
        site_id: int,
        device_id: int,
    ) -> dict:
        demographics_by_track = {
            result["track_id"]: (result["age"], result["sex"])
            for result in demographics_batch["results"]
        }
        rows = []
        for event in event_batch["events"]:
            absolute_utc = source_origin + timedelta(seconds=float(event["timestamp"]))
            age, sex = demographics_by_track[event["track_id"]]
            timestamp = absolute_utc.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            rows.append(
                {
                    "organisation_id": organisation_id,
                    "site_id": site_id,
                    "device_id": device_id,
                    "event_id": event["event_id"],
                    "event": int(event["event_type"]),
                    "timestamp": timestamp,
                    "sex": int(sex),
                    "age_bucket": int(_age_to_bucket(age)),
                }
            )
        return {"rows": rows}
