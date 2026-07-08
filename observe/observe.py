"""DetectionBatch + EmbeddingBatch -> ObservationBatch."""

__all__ = ["Observe"]


class Observe:
    __slots__ = ()

    def __call__(self, detection_batch, embedding_batch) -> dict:
        if detection_batch is None:
            raise ValueError("DetectionBatch is required")
        if embedding_batch is None:
            raise ValueError("EmbeddingBatch is required")

        for field in ("frame_id", "timestamp", "detections"):
            if field not in detection_batch:
                raise ValueError(f"Missing required DetectionBatch field: {field}")
        for field in ("frame_id", "embeddings"):
            if field not in embedding_batch:
                raise ValueError(f"Missing required EmbeddingBatch field: {field}")

        if detection_batch["frame_id"] != embedding_batch["frame_id"]:
            raise ValueError("DetectionBatch.frame_id must match EmbeddingBatch.frame_id")

        detections = detection_batch["detections"]
        embeddings = embedding_batch["embeddings"]
        if detections is None:
            raise ValueError("Missing required DetectionBatch detections")
        if embeddings is None:
            raise ValueError("Missing required EmbeddingBatch embeddings")

        detection_ids = set()
        for detection in detections:
            detection_id = detection["detection_id"]
            if detection_id in detection_ids:
                raise ValueError(f"Duplicate detection_id in DetectionBatch: {detection_id}")
            detection_ids.add(detection_id)

        embedding_map = {}
        for embedding in embeddings:
            detection_id = embedding["detection_id"]
            if detection_id in embedding_map:
                raise ValueError(f"Duplicate detection_id in EmbeddingBatch: {detection_id}")
            embedding_map[detection_id] = embedding

        extra_embedding_ids = set(embedding_map) - detection_ids
        if extra_embedding_ids:
            detection_id = next(iter(extra_embedding_ids))
            raise ValueError(f"Embedding has no matching detection_id: {detection_id}")

        observations = []
        for detection in detections:
            detection_id = detection["detection_id"]
            if detection_id not in embedding_map:
                raise ValueError(f"Missing embedding for detection_id: {detection_id}")

            bbox = detection["bbox"]
            embedding = embedding_map[detection_id]
            observations.append(
                {
                    "detection_id": detection_id,
                    "bbox": bbox,
                    "center": {
                        "x": (bbox["x1"] + bbox["x2"]) / 2,
                        "y": (bbox["y1"] + bbox["y2"]) / 2,
                    },
                    "embedding": embedding["vector"],
                    "confidence": detection["confidence"],
                }
            )

        return {
            "frame_id": detection_batch["frame_id"],
            "timestamp": detection_batch["timestamp"],
            "observations": observations,
        }
