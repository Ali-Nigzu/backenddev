import numpy as np

def build_observation(detection: dict, embedding: dict) -> dict:
    """
    Card 7: Observation Contract
    Pure deterministic mapping of detection + embedding → observation
    """

    x1, y1, x2, y2 = detection["bbox"]

    center = [
        (x1 + x2) / 2,
        (y1 + y2) / 2
    ]

    return {
        "detection_id": detection["detection_id"],
        "timestamp": float(detection["timestamp"]),
        "center": center,
        "bbox": detection["bbox"],
        "confidence": float(detection["confidence"]),
        "embedding": embedding["embedding"]
    }
