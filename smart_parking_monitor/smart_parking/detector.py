# smart_parking/detector.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    cls_id: int

class VehicleDetector:
    def __init__(self,
                 use_yolo: bool = False,
                 model_path: str = "yolov8n.pt",
                 img_size: int = 960,
                 conf_threshold: float = 0.25,
                 enable_multiscale: bool = True,
                 extra_scale: float = 1.6):
        self.use_yolo = use_yolo
        self.model = None
        self.cls_ids_of_interest = {2, 3, 5, 7}  # car, motorbike, bus, truck 등
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.scales: List[float] = [1.0]
        if enable_multiscale and extra_scale > 1.0:
            self.scales.append(extra_scale)

        if use_yolo:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print("[Detector] YOLO model loaded")
            except Exception as e:
                print(f"[Detector] YOLO load failed, fallback to dummy: {e}")
                self.use_yolo = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []  # Or handle as needed (e.g., skip frame)
        
        # Ensure frame is a valid numpy array
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        
        # Convert grayscale to BGR if necessary (common for PKLot videos)
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if not self.use_yolo or self.model is None:
            return self._dummy_detection(frame)

        detections: List[Detection] = []
        for scale in self.scales:
            scaled_frame = frame if scale == 1.0 else cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            result = self.model(
                scaled_frame,
                imgsz=self.img_size,
                verbose=False
            )[0]
            detections.extend(self._extract_detections(result, scale))

        if len(detections) > 1 and len(self.scales) > 1:
            detections = self._nms(detections)
        return detections

    def _dummy_detection(self, frame: np.ndarray) -> List[Detection]:
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        size = min(w, h) // 10
        bbox = (cx - size, cy - size, cx + size, cy + size)
        return [Detection(bbox=bbox, score=0.9, cls_id=2)]

    def _extract_detections(self, result, scale: float) -> List[Detection]:
        detections: List[Detection] = []
        inv_scale = 1.0 / scale
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.cls_ids_of_interest:
                continue
            conf = float(box.conf[0].item())
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    bbox=(
                        int(x1 * inv_scale),
                        int(y1 * inv_scale),
                        int(x2 * inv_scale),
                        int(y2 * inv_scale),
                    ),
                    score=conf,
                    cls_id=cls_id,
                )
            )
        return detections

    def _nms(self, detections: List[Detection], iou_thr: float = 0.5) -> List[Detection]:
        ordered = sorted(detections, key=lambda d: d.score, reverse=True)
        kept: List[Detection] = []
        for det in ordered:
            if all(self._iou(det.bbox, k.bbox) < iou_thr for k in kept):
                kept.append(det)
        return kept

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union
