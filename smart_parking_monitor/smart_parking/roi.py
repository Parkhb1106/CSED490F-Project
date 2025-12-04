# smart_parking/roi.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ParkingSlot:
    slot_id: int
    polygon: np.ndarray       # (N, 2)
    is_no_parking_zone: bool = False

class ParkingSlotDetector:
    def __init__(self):
        self.slots: List[ParkingSlot] = []
        self.initialized = False

    def manual_init_example(self, frame):
        h, w, _ = frame.shape
        poly1 = np.array([
            [w * 0.2, h * 0.3],
            [w * 0.4, h * 0.3],
            [w * 0.4, h * 0.7],
            [w * 0.2, h * 0.7],
        ], dtype=np.int32)
        poly2 = np.array([
            [w * 0.6, h * 0.3],
            [w * 0.8, h * 0.3],
            [w * 0.8, h * 0.7],
            [w * 0.6, h * 0.7],
        ], dtype=np.int32)
        self.slots = [
            ParkingSlot(slot_id=1, polygon=poly1, is_no_parking_zone=False),
            ParkingSlot(slot_id=2, polygon=poly2, is_no_parking_zone=False),
        ]
        self.initialized = True
        print("[ROI] Manual parking slots initialized")

    def ensure_initialized(self, frame):
        if not self.initialized:
            # TODO: Auto-ROI 알고리즘으로 대체
            self.manual_init_example(frame)

    def get_slots(self) -> List[ParkingSlot]:
        return self.slots

    def point_in_slot(self, x: int, y: int) -> Optional[ParkingSlot]:
        pt = (x, y)
        for slot in self.slots:
            inside = cv2.pointPolygonTest(slot.polygon, pt, False) >= 0
            if inside:
                return slot
        return None