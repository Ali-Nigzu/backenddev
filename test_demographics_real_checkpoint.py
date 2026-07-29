import hashlib
from pathlib import Path

import numpy as np
import pytest

from demographics import Demographic
from demographics.model import EXPECTED_SHA256


def test_real_checkpoint_cpu_smoke():
    checkpoint = Path("demographics/demographicweights.pth")
    if not checkpoint.exists():
        pytest.skip("MiVOLO checkpoint has not been assembled")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == EXPECTED_SHA256
    image = np.full((32, 24, 3), 127, dtype=np.uint8)
    event_batch = {
        "events": [
            {
                "track_id": "track_real",
                "timestamp": 1.25,
                "event_type": 1,
                "best_crop": {"frame_id": "frame-0", "bbox": {"x1": 0, "y1": 0, "x2": 24, "y2": 32}},
            }
        ]
    }
    frame_batch = {"frames": [{"frame_id": "frame-0", "timestamp": 0.0, "image": image}]}
    result = Demographic(checkpoint_path=checkpoint, device="cpu")(event_batch, frame_batch)
    assert result["results"][0]["track_id"] == "track_real"
    assert isinstance(result["results"][0]["age"], int)
    assert 1 <= result["results"][0]["age"] <= 95
    assert result["results"][0]["sex"] in (0, 1)
