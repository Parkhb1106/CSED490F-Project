# # smart_parking/roi.py
# import cv2
# import numpy as np
# from dataclasses import dataclass
# from typing import List, Optional

# @dataclass
# class ParkingSlot:
#     slot_id: int
#     polygon: np.ndarray       # (N, 2)
#     is_no_parking_zone: bool = False

# class ParkingSlotDetector:
#     def __init__(self):
#         self.slots: List[ParkingSlot] = []
#         self.initialized = False

#     def manual_init_example(self, frame):
#         h, w, _ = frame.shape
#         poly1 = np.array([
#             [w * 0.2, h * 0.3],
#             [w * 0.4, h * 0.3],
#             [w * 0.4, h * 0.7],
#             [w * 0.2, h * 0.7],
#         ], dtype=np.int32)
#         poly2 = np.array([
#             [w * 0.6, h * 0.3],
#             [w * 0.8, h * 0.3],
#             [w * 0.8, h * 0.7],
#             [w * 0.6, h * 0.7],
#         ], dtype=np.int32)
#         self.slots = [
#             ParkingSlot(slot_id=1, polygon=poly1, is_no_parking_zone=False),
#             ParkingSlot(slot_id=2, polygon=poly2, is_no_parking_zone=False),
#         ]
#         self.initialized = True
#         print("[ROI] Manual parking slots initialized")

#     def ensure_initialized(self, frame):
#         if not self.initialized:
#             # TODO: Auto-ROI 알고리즘으로 대체
#             self.manual_init_example(frame)

#     def get_slots(self) -> List[ParkingSlot]:
#         return self.slots

#     def point_in_slot(self, x: int, y: int) -> Optional[ParkingSlot]:
#         pt = (x, y)
#         for slot in self.slots:
#             inside = cv2.pointPolygonTest(slot.polygon, pt, False) >= 0
#             if inside:
#                 return slot
#         return None


# smart_parking/roi.py
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


class ParkingSlotDetector:
    """
    Auto-ROI:
      - 초기 몇 프레임 동안 YOLO detection 결과의 bbox center와 크기를 모은 뒤
      - y 기준으로 row를 나누고, 각 row에서 x 간격과 bbox height를 이용해
        사각형 슬롯 polygon들을 자동으로 생성한다.
    """

    def __init__(self, min_collect_frames: int = 60):
        self.slots: List[ParkingSlot] = []
        self.initialized = False

        # auto-ROI를 위한 샘플
        self._frame_shape: Optional[Tuple[int, int]] = None  # (h, w)
        self._centers: List[Tuple[float, float]] = []
        self._heights: List[float] = []
        self._collect_frames = 0
        self._min_collect_frames = min_collect_frames

    # ---- public API ----
    def is_initialized(self) -> bool:
        return self.initialized

    def update_auto(self, frame, detections: List[Detection]):
        """각 프레임마다 호출: detections를 이용해 auto-ROI 학습."""
        if self.initialized:
            return

        h, w = frame.shape[:2]
        if self._frame_shape is None:
            self._frame_shape = (h, w)

        self._collect_frames += 1

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self._centers.append((cx, cy))
            self._heights.append(y2 - y1)

        # 충분히 모였으면 slot 생성
        if self._collect_frames >= self._min_collect_frames and len(self._centers) > 0:
            self._build_slots_from_samples()

    def get_slots(self) -> List[ParkingSlot]:
        return self.slots

    def point_in_slot(self, x: int, y: int) -> Optional[ParkingSlot]:
        if not self.initialized:
            return None
        pt = (x, y)
        for slot in self.slots:
            inside = cv2.pointPolygonTest(slot.polygon, pt, False) >= 0
            if inside:
                return slot
        return None

    # ---- internal helpers ----
    def _build_slots_from_samples(self):
        """모아 둔 center들을 row / column 으로 나누어 자동으로 슬롯 polygon 생성."""
        h, w = self._frame_shape
        pts = np.array(self._centers)  # (N, 2)
        heights = np.array(self._heights)

        # 1) y 기준으로 정렬 후 row 분리
        order = np.argsort(pts[:, 1])
        pts_sorted = pts[order]

        rows: List[np.ndarray] = []
        current = [pts_sorted[0]]
        # y 차이가 큰 부분에서 row 분리 (자동 threshold)
        dy = np.diff(pts_sorted[:, 1])
        # 작은 간격들(같은 row 내)과 큰 간격들(row 간) 사이에 중간값 사용
        if len(dy) > 0:
            median_dy = np.median(dy)
            row_gap_thr = max(25.0, median_dy * 2.5)  # 경험적 값
        else:
            row_gap_thr = 40.0

        for prev, cur in zip(pts_sorted[:-1], pts_sorted[1:]):
            if abs(cur[1] - prev[1]) < row_gap_thr:
                current.append(cur)
            else:
                rows.append(np.array(current))
                current = [cur]
        rows.append(np.array(current))

        slots: List[ParkingSlot] = []
        global_med_height = float(np.median(heights)) if len(heights) > 0 else 40.0
        slot_id = 1

        for row_pts in rows:
            if len(row_pts) < 2:
                continue
            # x 정렬 후 간격으로 width 추정
            xs = np.sort(row_pts[:, 0])
            dx = np.diff(xs)
            med_dx = float(np.median(dx))
            if med_dx < 10:  # 이상하게 작으면 skip
                continue
            slot_width = med_dx * 0.9
            slot_height = global_med_height * 1.2

            for cx, cy in row_pts:
                half_w = slot_width / 2.0
                half_h = slot_height / 2.0
                x1 = int(max(0, cx - half_w))
                x2 = int(min(w - 1, cx + half_w))
                y1 = int(max(0, cy - half_h))
                y2 = int(min(h - 1, cy + half_h))
                poly = np.array([[x1, y1],
                                 [x2, y1],
                                 [x2, y2],
                                 [x1, y2]], dtype=np.int32)
                slots.append(ParkingSlot(slot_id=slot_id, polygon=poly))
                slot_id += 1

        if len(slots) == 0:
            print("[ROI] Auto-ROI failed, no slots generated.")
            return

        self.slots = slots
        self.initialized = True
        print(f"[ROI] Auto-ROI initialized with {len(self.slots)} slots")