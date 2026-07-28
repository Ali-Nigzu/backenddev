"""Typed dictionary contracts for Event outputs."""

from typing import TypedDict


class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class BestCrop(TypedDict):
    frame_id: str
    bbox: BoundingBox


class EventRecord(TypedDict):
    track_id: str
    timestamp: float
    event_type: int
    best_crop: BestCrop


class EventBatch(TypedDict):
    events: list[EventRecord]
