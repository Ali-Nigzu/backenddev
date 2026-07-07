"""DetectionBatch -> EmbeddingBatch."""

__all__ = ["Embed"]


class Embed:
    def __call__(self, detection_batch):
        return {
            "frame_id": detection_batch["frame_id"],
            "timestamp": float(detection_batch["timestamp"]),
            "embeddings": [
                {
                    "detection_id": detection["detection_id"],
                    "vector": {"dtype": "float32", "shape": [0], "values": []},
                }
                for detection in detection_batch["detections"]
            ],
        }
