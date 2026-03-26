from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ONNXPredictor:
    def __init__(self, onnx_path: str = "models/best_model.onnx", labels_path: str = "models/labels.json", preprocess_path: str = "models/preprocess.json"):
        self.onnx_path = self._resolve_path(onnx_path)
        self.labels_path = self._resolve_path(labels_path)
        self.preprocess_path = self._resolve_path(preprocess_path)

        labels_payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
        preprocess_payload = json.loads(self.preprocess_path.read_text(encoding="utf-8"))

        self.class_names = labels_payload["classes"]
        image_size = preprocess_payload["image_size"]
        mean = preprocess_payload["mean"]
        std = preprocess_payload["std"]

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def predict(self, image: Image.Image) -> tuple[str, dict[str, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).numpy().astype(np.float32)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        probs = self._softmax(logits[0])
        best_idx = int(np.argmax(probs))
        scores = {label: float(prob) for label, prob in zip(self.class_names, probs)}
        return self.class_names[best_idx], scores

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
