"""Locked Track V2 contract helpers.

The public Track contract is dictionary-compatible because upstream V2 modules in this
repository currently exchange contract objects as dictionaries. These aliases document
that contract without introducing runtime-only tracking state.
"""

from typing import Any, Dict, List, MutableMapping, Sequence

BoundingBox = Dict[str, float]
Point2D = Dict[str, float]
FeatureVector = Dict[str, Any]
BestCrop = Dict[str, Any]
Point = Dict[str, Any]
TrackRecord = Dict[str, Any]
TrackingState = MutableMapping[str, List[TrackRecord]]
Observation = Dict[str, Any]
ObservationBatch = Dict[str, Any]

REQUIRED_TRACKING_STATE_FIELDS = ("tracks",)
REQUIRED_TRACK_FIELDS = ("track_id", "path", "best_crop", "best_crop_confidence")
REQUIRED_BEST_CROP_FIELDS = ("frame_id", "bbox", "embedding")
REQUIRED_POINT_FIELDS = ("timestamp", "center")
REQUIRED_OBSERVATION_BATCH_FIELDS = ("frame_id", "timestamp", "observations")
REQUIRED_OBSERVATION_FIELDS = (
    "detection_id",
    "bbox",
    "center",
    "embedding",
    "confidence",
)
REQUIRED_BBOX_FIELDS = ("x1", "y1", "x2", "y2")
REQUIRED_CENTER_FIELDS = ("x", "y")


def vector_values(vector: Any) -> Sequence[float]:
    """Return FeatureVector values from the locked dict form or a raw sequence."""

    if vector is None:
        return ()
    if isinstance(vector, dict):
        values = vector.get("values", ())
    else:
        values = vector
    if values is None:
        return ()
    return values
