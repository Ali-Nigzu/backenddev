import sys
import types

import numpy as np


class _ArrayLike:
    def __init__(self, value):
        self._value = np.array(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Boxes:
    def __init__(self):
        self.xyxy = _ArrayLike([[1.2, 2.3, 5.4, 7.8], [-1.0, 0.0, 0.0, 4.0]])
        self.conf = _ArrayLike([0.91, 0.42])

    def __len__(self):
        return 2


class _Result:
    boxes = _Boxes()


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [_Result()]


def _install_fake_ultralytics():
    module = types.ModuleType("ultralytics")
    module.YOLO = lambda _path: _FakeModel()
    sys.modules.setdefault("ultralytics", module)


def test_detect_returns_one_detection_batch_with_per_frame_detections(monkeypatch):
    _install_fake_ultralytics()
    from detect import Detect

    detect = Detect()
    frame = {
        "frame_id": "frame-0",
        "timestamp": 1.5,
        "image": np.zeros((10, 20, 3), dtype=np.uint8),
    }

    old_per_frame_output = detect._detect_frame(frame)
    new_output = detect({"frames": [frame]})

    assert new_output == {"detections": [old_per_frame_output]}
    assert detect._model.calls[0][1] == {
        "classes": (0,),
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 300,
        "device": "cpu",
        "verbose": False,
    }


def test_detect_preserves_input_order_and_empty_frames(monkeypatch):
    _install_fake_ultralytics()
    from detect import Detect

    detect = Detect()
    first = {"frame_id": "frame-2", "timestamp": 2.0, "image": np.zeros((10, 20, 3), dtype=np.uint8)}
    second = {"frame_id": "frame-1", "timestamp": 1.0, "image": np.zeros((10, 20, 3), dtype=np.uint8)}

    assert detect({"frames": []}) == {"detections": []}
    output = detect({"frames": [first, second]})
    assert [item["frame_id"] for item in output["detections"]] == ["frame-2", "frame-1"]
    assert [item["timestamp"] for item in output["detections"]] == [2.0, 1.0]
