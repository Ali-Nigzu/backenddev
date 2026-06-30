from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


class DemographicsEngine:
    """FairFace demographics inference for detection-provided person images."""

    RACE_SLICE = slice(0, 4)
    SEX_SLICE = slice(7, 9)
    AGE_SLICE = slice(9, 18)

    CHECKPOINT_PATH = Path(__file__).parent / "fairface_alldata_4race_20191111.pt"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model = self._load_model()

    def _load_model(self):
        if not self.CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"FairFace checkpoint not found: {self.CHECKPOINT_PATH}"
            )

        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 18)

        state_dict = torch.load(str(self.CHECKPOINT_PATH), map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_image(self, person_image):
        if person_image is None:
            raise ValueError("person_image is required")
        if not isinstance(person_image, np.ndarray):
            raise TypeError("person_image must be a numpy array")
        if person_image.ndim != 3 or person_image.shape[2] != 3:
            raise ValueError("person_image must have shape HxWx3")

        return self.transform(person_image).unsqueeze(0).to(self.device)

    def predict(self, person_image):
        image_tensor = self._prepare_image(person_image)

        with torch.inference_mode():
            outputs = self.model(image_tensor).squeeze(0)

        sex = int(torch.argmax(outputs[self.SEX_SLICE]).item())
        race = int(torch.argmax(outputs[self.RACE_SLICE]).item())
        age = int(torch.argmax(outputs[self.AGE_SLICE]).item())

        return {
            "sex": sex,
            "race": race,
            "age": age,
        }
