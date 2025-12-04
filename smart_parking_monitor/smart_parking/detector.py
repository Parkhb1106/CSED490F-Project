# smart_parking/detector.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    cls_id: int

class VehicleDetector:
    def __init__(self, use_yolo: bool = False, model_path: str = "yolov8n.pt"):
        self.use_yolo = use_yolo
        self.model = None
        self.cls_ids_of_interest = {2, 3, 5, 7}  # car, motorbike, bus, truck 등

        if use_yolo:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print("[Detector] YOLO model loaded")
            except Exception as e:
                print(f"[Detector] YOLO load failed, fallback to dummy: {e}")
                self.use_yolo = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.use_yolo or self.model is None:
            # Dummy detection: 중앙에 박스 하나
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2
            size = min(w, h) // 10
            bbox = (cx - size, cy - size, cx + size, cy + size)
            return [Detection(bbox=bbox, score=0.9, cls_id=2)]

        results = self.model(frame, verbose=False)[0]
        detections: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.cls_ids_of_interest:
                continue
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=conf,
                    cls_id=cls_id,
                )
            )
        return detections