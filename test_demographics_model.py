"""Standalone MobileNetV3 validation harness using existing person detection.

Edit IMAGE_PATH below, then run:

    python test_demographics_model.py

This script intentionally does not implement production demographics inference. It
only verifies that the vendored MobileNetV3-Small checkpoint can run on a crop
selected from the existing detection module output.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_PATH = "path/to/image.jpg"

REPO_ROOT = Path(__file__).resolve().parent
MOBILENETV3_DIR = REPO_ROOT / "mobilenetv3"
CHECKPOINT_PATH = MOBILENETV3_DIR / "450_act3_mobilenetv3_small.pth"
INPUT_SIZE = 224
RESIZE_SHORT_EDGE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TOP_K = 5


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        fail(f"IMAGE_PATH does not exist: {path}")
    if not path.is_file():
        fail(f"IMAGE_PATH is not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        fail(f"OpenCV could not read IMAGE_PATH as an image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        fail(f"Expected a 3-channel BGR image, got shape: {image.shape}")
    return image


def run_detection(image: np.ndarray) -> list[dict[str, Any]]:
    from detection.detection_engine import detect

    frame = {
        "frame_id": Path(IMAGE_PATH).name,
        "timestamp": 0.0,
        "image": image,
    }
    detections = detect(frame)
    if not isinstance(detections, list):
        fail(f"Detection module returned {type(detections).__name__}, expected list")
    if not detections:
        fail("Detection module returned no person detections")
    return detections


def validate_detection(detection: dict[str, Any], image_shape: tuple[int, ...]) -> None:
    required = ["detection_id", "bbox", "confidence"]
    missing = [field for field in required if field not in detection]
    if missing:
        fail(f"Detection is missing required field(s): {missing}")

    bbox = detection["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        fail(f"Detection bbox must be a 4-item list/tuple, got: {bbox!r}")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
        fail(f"Detection bbox contains non-finite numeric values: {bbox!r}")

    confidence = detection["confidence"]
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        fail(f"Detection confidence must be finite numeric, got: {confidence!r}")

    height, width = image_shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width=width, height=height)
    if x2 <= x1 or y2 <= y1:
        fail(f"Detection bbox is invalid after clamping: original={bbox!r}")


def select_highest_confidence_detection(
    detections: list[dict[str, Any]], image_shape: tuple[int, ...]
) -> dict[str, Any]:
    for detection in detections:
        validate_detection(detection, image_shape)

    return sorted(
        detections,
        key=lambda detection: (
            -float(detection["confidence"]),
            str(detection.get("detection_id", "")),
        ),
    )[0]


def clamp_bbox(
    bbox: list[float] | tuple[float, float, float, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1i = max(0, min(width, int(math.floor(float(x1)))))
    y1i = max(0, min(height, int(math.floor(float(y1)))))
    x2i = max(0, min(width, int(math.ceil(float(x2)))))
    y2i = max(0, min(height, int(math.ceil(float(y2)))))
    return x1i, y1i, x2i, y2i


def extract_person_crop(image: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(detection["bbox"], width=width, height=height)
    if x2 <= x1 or y2 <= y1:
        fail(f"Selected detection bbox is invalid after clamping: {detection['bbox']!r}")

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        fail(f"Selected detection produced an empty crop: bbox={detection['bbox']!r}")
    return crop.copy()


def resize_short_edge_bicubic(rgb_image: np.ndarray, short_edge: int) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    if height <= 0 or width <= 0:
        fail(f"Cannot resize image with invalid shape: {rgb_image.shape}")

    if height < width:
        new_height = short_edge
        new_width = int(round(width * short_edge / height))
    else:
        new_width = short_edge
        new_height = int(round(height * short_edge / width))

    return cv2.resize(
        rgb_image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )


def center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height < crop_size or width < crop_size:
        fail(
            f"Cannot center-crop {crop_size}x{crop_size} from resized crop shape "
            f"{image.shape}"
        )
    y1 = (height - crop_size) // 2
    x1 = (width - crop_size) // 2
    return image[y1 : y1 + crop_size, x1 : x1 + crop_size]


def preprocess_for_mobilenetv3(crop_bgr: np.ndarray):
    try:
        import torch
    except ModuleNotFoundError as exc:
        fail("PyTorch is required to run MobileNetV3 inference but is not installed")
        raise exc

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = resize_short_edge_bicubic(rgb, RESIZE_SHORT_EDGE)
    cropped = center_crop(resized, INPUT_SIZE)
    normalized = (cropped.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


def load_mobilenetv3_model(device: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        fail("PyTorch is required to load the MobileNetV3 checkpoint but is not installed")
        raise exc

    if not CHECKPOINT_PATH.exists():
        fail(f"MobileNetV3 checkpoint does not exist: {CHECKPOINT_PATH}")

    if str(MOBILENETV3_DIR) not in sys.path:
        sys.path.insert(0, str(MOBILENETV3_DIR))

    from mobilenetv3 import MobileNetV3_Small

    model = MobileNetV3_Small()
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, load_result


def print_detection_summary(detections: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    print("Detection summary")
    print("=================")
    print(f"detections: {len(detections)}")
    print(f"selected detection_id: {selected['detection_id']}")
    print(f"selected confidence: {float(selected['confidence']):.6f}")
    print(f"selected bbox: {[float(v) for v in selected['bbox']]}")
    print()


def print_inference_summary(logits) -> None:
    import torch

    probabilities = torch.softmax(logits, dim=1)
    top_values, top_indices = torch.topk(probabilities, k=min(TOP_K, probabilities.shape[1]), dim=1)

    print("MobileNetV3 inference output")
    print("============================")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"raw output type: {type(logits).__name__}")
    print(f"raw output shape: {tuple(logits.shape)}")
    print("top probabilities by ImageNet-style class index:")
    for rank, (index, probability) in enumerate(
        zip(top_indices[0].tolist(), top_values[0].tolist()), start=1
    ):
        print(f"  {rank}. class_index={index} probability={probability:.6f}")
    print()
    print("Note: this checkpoint naturally produces 1000 generic classification logits.")
    print("No sex/race/age mapping is applied by this validation harness.")


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        fail("PyTorch is required to run this harness but is not installed")

    image = load_image(IMAGE_PATH)
    detections = run_detection(image)
    selected_detection = select_highest_confidence_detection(detections, image.shape)
    print_detection_summary(detections, selected_detection)

    crop = extract_person_crop(image, selected_detection)
    input_tensor = preprocess_for_mobilenetv3(crop)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, load_result = load_mobilenetv3_model(device)
    print("Checkpoint load result")
    print("======================")
    print(load_result)
    print(f"device: {device}")
    print()

    with torch.inference_mode():
        logits = model(input_tensor.to(device))

    print_inference_summary(logits.detach().cpu())


if __name__ == "__main__":
    main()
