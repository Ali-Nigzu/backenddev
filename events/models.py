from typing import Literal, TypedDict


EventType = Literal["ENTRY", "EXIT"]
Direction = Literal["IN", "OUT"]


class RuntimeEventCandidate(TypedDict):
    event_id: str
    runtime_track_id: str
    timestamp: float
    event_type: EventType
    direction: Direction
    supporting_positions: list[list[float]]
