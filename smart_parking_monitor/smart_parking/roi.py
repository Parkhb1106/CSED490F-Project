import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .detector import Detection


@dataclass
class ParkingSlot:
    slot_id: int
    polygon: np.ndarray       # (N, 2)
    is_no_parking_zone: bool = False
    support: int = 0


class ParkingSlotDetector:
    """
    Slot layout is bootstrapped form early detections only, then frozen:
      1) Collect detection bounding boxes for a short window.
      2) Cluster the boxes by IoU so that the same parked car contributes to one slot.
      3) Turn the averaged boxes into slot polygons and keep them fixed afterwards.

    This keeps the total slot count low and provides a stable ROI that
    subsequent vehicle detections can be compared against.
    """

    def __init__(self,
                 min_slot_area: int = 400,
                 min_collect_frames: int = 80,
                 min_samples: int = 25,
                 min_support: int = 3,
                 cluster_iou: float = 0.55,
                 max_samples: int = 1500):
        self.slots: List[ParkingSlot] = []
        self.initialized = False

        self._frame_shape: Optional[Tuple[int, int]] = None
        self._frame_index = 0
        self._next_slot_id = 1

        self._sample_boxes: List[Tuple[int, int, int, int]] = []
        self._max_samples = max_samples

        self._min_slot_area = min_slot_area
        self._min_collect_frames = min_collect_frames
        self._min_samples = min_samples
        self._min_support = min_support
        self._cluster_iou = cluster_iou

    # ---- public API -----------------------------------------------------
    def is_initialized(self) -> bool:
        return self.initialized

    def update_auto(self, frame, detections: List[Detection]):
        if frame is None:
            return

        if self.initialized:
            # Slots remain fixed once initialized so only comparisons happen after this.
            return

        h, w = frame.shape[:2]
        if self._frame_shape is None:
            self._frame_shape = (h, w)

        self._frame_index += 1

        for det in detections:
            bbox = self._clip_bbox(det.bbox, w, h)
            if bbox is None:
                continue

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self._min_slot_area:
                continue

            self._append_sample(bbox)

        if (self._frame_index >= self._min_collect_frames and
                len(self._sample_boxes) >= self._min_samples):
            self._finalize_slots()

    def get_slots(self) -> List[ParkingSlot]:
        return self.slots

    def point_in_slot(self, x: int, y: int) -> Optional[ParkingSlot]:
        if not self.initialized:
            return None

        pt = (x, y)
        for slot in self.slots:
            if cv2.pointPolygonTest(slot.polygon, pt, False) >= 0:
                return slot
        return None

    # ---- sample collection & clustering ---------------------------------
    def _append_sample(self, bbox: Tuple[int, int, int, int]):
        self._sample_boxes.append(bbox)
        if len(self._sample_boxes) > self._max_samples:
            self._sample_boxes.pop(0)

    def _finalize_slots(self):
        clusters = self._cluster_samples()
        slots: List[ParkingSlot] = []
        next_id = 1
        for cluster in clusters:
            if cluster["count"] < self._min_support:
                continue
            bbox = tuple(int(v) for v in cluster["bbox"])
            polygon = self._bbox_to_polygon(bbox)
            slots.append(ParkingSlot(
                slot_id=next_id,
                polygon=polygon,
                support=cluster["count"]
            ))
            next_id += 1

        if slots:
            self.slots = slots
            self._next_slot_id = next_id
            self.initialized = True
            print(f"[ROI] Auto-ROI locked with {len(self.slots)} slots")
        else:
            print("[ROI] Not enough consistent detections to lock slots yet")

    def _cluster_samples(self) -> List[dict]:
        clusters: List[dict] = []

        for box in self._sample_boxes:
            matched = None
            best_iou = 0.0
            for cluster in clusters:
                iou = self._bbox_iou(box, cluster["bbox"])
                if iou > self._cluster_iou and iou > best_iou:
                    best_iou = iou
                    matched = cluster

            if matched is not None:
                count = matched["count"]
                blended = (
                    (matched["bbox"][0] * count + box[0]) / (count + 1),
                    (matched["bbox"][1] * count + box[1]) / (count + 1),
                    (matched["bbox"][2] * count + box[2]) / (count + 1),
                    (matched["bbox"][3] * count + box[3]) / (count + 1),
                )
                matched["bbox"] = blended
                matched["count"] = count + 1
            else:
                clusters.append({
                    "bbox": tuple(float(v) for v in box),
                    "count": 1
                })

        # Suppress any clusters that still overlap too much by keeping the strongest ones first.
        clusters.sort(key=lambda c: c["count"], reverse=True)
        filtered: List[dict] = []
        for cluster in clusters:
            overlapped = any(
                self._bbox_iou(cluster["bbox"], keep["bbox"]) > self._cluster_iou
                for keep in filtered
            )
            if not overlapped:
                filtered.append(cluster)
        return filtered

    # ---- geometry helpers -----------------------------------------------
    def _bbox_to_polygon(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.int32)

    def _clip_bbox(self, bbox: Tuple[int, int, int, int],
                   width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _bbox_iou(self, a: Tuple[int, int, int, int],
                  b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return inter / denom
