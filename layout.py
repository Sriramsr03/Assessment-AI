from __future__ import annotations

from typing import Dict, List

import numpy as np


class LayoutDetector:
    """
    Optional YOLOv8-based layout detector for:
    - question regions
    - option regions

    Expects model classes named exactly: "question", "option".
    """

    def __init__(self, model_path: str = "models/layout_yolov8.pt") -> None:
        self.model_path = model_path
        self.model = None

    def _load_model(self) -> None:
        if self.model is None:
            try:
                from ultralytics import YOLO
            except Exception as exc:
                raise RuntimeError(
                    "ultralytics is not installed. Install it to use layout detection."
                ) from exc
            self.model = YOLO(self.model_path)

    def detect_regions(self, image: np.ndarray) -> List[Dict[str, object]]:
        self._load_model()
        results = self.model.predict(image, verbose=False)
        regions: List[Dict[str, object]] = []

        if not results:
            return regions

        names = results[0].names
        boxes = results[0].boxes
        if boxes is None:
            return regions

        for box in boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf.item())
            regions.append(
                {
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                }
            )
        return regions

    @staticmethod
    def crop_regions(image: np.ndarray, regions: List[Dict[str, object]]) -> List[np.ndarray]:
        crops: List[np.ndarray] = []
        for reg in regions:
            x1, y1, x2, y2 = reg["bbox"]
            crops.append(image[y1:y2, x1:x2])
        return crops

