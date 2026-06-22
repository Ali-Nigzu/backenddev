from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RuntimeTrackV2:
    runtime_track_id: str
    state: str

    current_center: List[float]
    current_bbox: List[float]
    velocity: List[float]

    first_seen_timestamp: float
    last_seen_timestamp: float

    hit_count: int = 1
    miss_count: int = 0

    detection_history: List[str] = field(default_factory=list)
    center_history: List[List[float]] = field(default_factory=list)

    last_embedding: Optional[Any] = None
