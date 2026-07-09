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

        frame_id = detection_batch["frame_id"]
        timestamp = detection_batch["timestamp"]
        if frame_id != embedding_batch["frame_id"]:
            raise ValueError("DetectionBatch.frame_id must match EmbeddingBatch.frame_id")

        detections = detection_batch["detections"]
        embeddings = embedding_batch["embeddings"]
        if detections is None:
            raise ValueError("Missing required DetectionBatch detections")
        if embeddings is None:
            raise ValueError("Missing required EmbeddingBatch embeddings")

        embedding_map = {}
        for embedding in embeddings:
            detection_id = embedding["detection_id"]
            if detection_id in embedding_map:
                raise ValueError(f"Duplicate detection_id in EmbeddingBatch: {detection_id}")
            embedding_map[detection_id] = embedding

        seen_detection_ids = set()
        observations = [None] * len(detections)
        for index, detection in enumerate(detections):
            detection_id = detection["detection_id"]
            if detection_id in seen_detection_ids:
                raise ValueError(f"Duplicate detection_id in DetectionBatch: {detection_id}")
            seen_detection_ids.add(detection_id)

            embedding = embedding_map.pop(detection_id, None)
            if embedding is None:
                raise ValueError(f"Missing embedding for detection_id: {detection_id}")

            bbox = detection["bbox"]
            confidence = detection["confidence"]
            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]
            vector = embedding["vector"]
            observations[index] = {
                "detection_id": detection_id,
                "bbox": bbox,
                "center": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                },
                "embedding": vector,
                "confidence": confidence,
            }

        if embedding_map:
            detection_id = next(iter(embedding_map))
            raise ValueError(f"Embedding has no matching detection_id: {detection_id}")

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "observations": observations,
        }
