"""Deterministic internal records used by the Snapshot engines."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SNAPSHOT_STATE_VERSION = 1


@dataclass(frozen=True)
class Event:
    destination: str
    site_id: int
    device_id: int
    event_id: int
    event: int
    timestamp: datetime
    sex: int
    age_bucket: int

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.destination, self.device_id, self.event_id

    @property
    def order_key(self) -> tuple:
        return (
            self.timestamp,
            -self.event,
            self.destination,
            self.site_id,
            self.device_id,
            self.event_id,
        )

    def json(self) -> dict[str, Any]:
        from .site_engine import stamp
        return {
            "destination": self.destination,
            "site_id": self.site_id,
            "device_id": self.device_id,
            "event_id": self.event_id,
            "event": self.event,
            "timestamp": stamp(self.timestamp),
            "sex": self.sex,
            "age_bucket": self.age_bucket,
        }


@dataclass(frozen=True)
class SourceRange:
    destination: str
    site_id: int
    device_id: int
    start: datetime
    end: datetime


@dataclass
class SnapshotCandidate:
    ts: datetime
    payload: dict[str, Any]
    state: dict[str, Any]
    changed: bool = True


@dataclass
class AttemptStats:
    bq_queries: int = 0
    events_fetched: int = 0
    events_normalized: int = 0
    changed_sites: int = 0
    classifications: dict[str, int] = field(default_factory=dict)
