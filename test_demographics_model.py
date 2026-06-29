#!/usr/bin/env python3
"""Validate FairFace inference on crops returned by the detection pipeline.

This script is intentionally standalone and disposable. It loads one sample image,
runs the existing person detector, feeds every returned detection crop directly
into the FairFace 4-race checkpoint, and prints a compact table of predictions.
"""

from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from detection.detection_engine import detect


IMAGE_PATH = Path("data/samples/1000040807.jpg")
CHECKPOINT_PATH = Path("fairface_alldata_4race_20191111.pt")

RACE_LABELS = ["White", "Black", "Asian", "Indian"]
SEX_LABELS = ["M", "F"]
AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def build_fairface_transform():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_fairface_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FairFace checkpoint not found: {checkpoint_path}")

    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)

    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def predict_demographics(model, transform, crop, device: torch.device):
    if crop is None:
        raise ValueError("Detection crop is missing")

    image_tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor).squeeze(0)

    race_logits = outputs[:4]
    sex_logits = outputs[7:9]
    age_logits = outputs[9:18]

    race_index = int(torch.argmax(race_logits).item())
    sex_index = int(torch.argmax(sex_logits).item())
    age_index = int(torch.argmax(age_logits).item())

    return SEX_LABELS[sex_index], RACE_LABELS[race_index], AGE_LABELS[age_index]


def print_results(rows):
    print(f"Detections Found: {len(rows)}")
    print()
    print("---------------------------------------------------------")
    print("Detection | Confidence | Sex | Race  | Age")
    print("---------------------------------------------------------")
    for index, confidence, sex, race, age in rows:
        print(f"{index:<9} | {confidence:<10.2f} | {sex:<3} | {race:<5} | {age}")
    print("---------------------------------------------------------")


def main():
    image = load_image(IMAGE_PATH)
    frame = {
        "frame_id": IMAGE_PATH.stem,
        "timestamp": 0.0,
        "image": image,
    }

    detections = detect(frame)
    if not detections:
        raise RuntimeError("Detection returned no people")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_fairface_model(CHECKPOINT_PATH, device)
    transform = build_fairface_transform()

    rows = []
    for index, detection in enumerate(detections, start=1):
        sex, race, age = predict_demographics(model, transform, detection["image"], device)
        rows.append((index, float(detection["confidence"]), sex, race, age))

    print_results(rows)


if __name__ == "__main__":
    main()
