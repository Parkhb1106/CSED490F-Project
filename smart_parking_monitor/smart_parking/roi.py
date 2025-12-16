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
        # detection 샘플이 과도하게 누적되면 같은 슬롯이 여러 번 기록되어
        # row 분할과 width 추정이 크게 틀어질 수 있으므로 최소 간격 제어
        self._min_slot_gap_px = 12.0

    # ---- public API ----
    def is_initialized(self) -> bool:
        return self.initialized

    def update_auto(self, frame, detections: List[Detection]):
        """각 프레임마다 호출: detections를 이용해 auto-ROI 학습."""
        h, w = frame.shape[:2]
        if self._frame_shape is None:
            self._frame_shape = (h, w)

        new_sample = False
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 이미 생성된 슬롯 내부에 있는 detection은 샘플에서 제외한다.
            if self.slots and self._point_in_existing_slots(int(cx), int(cy)):
                continue

            self._centers.append((cx, cy))
            self._heights.append(y2 - y1)
            new_sample = True

        if not new_sample:
            return

        self._collect_frames += 1
        # 초기 학습 이후에는 더 짧은 구간 동안 샘플을 모아서 신규 슬롯을 추적한다.
        min_frames = self._min_collect_frames if not self.initialized else max(15, self._min_collect_frames // 3)

        # 충분히 모였으면 slot 생성/갱신
        if self._collect_frames >= min_frames and len(self._centers) > 0:
            start_slot_id = len(self.slots) + 1
            new_slots = self._build_slots_from_samples(start_slot_id=start_slot_id)
            self._collect_frames = 0
            self._centers.clear()
            self._heights.clear()

            if len(new_slots) == 0:
                if not self.initialized:
                    print("[ROI] Auto-ROI failed, no slots generated.")
                return

            if not self.initialized:
                self.slots = new_slots
                self.initialized = True
                print(f"[ROI] Auto-ROI initialized with {len(self.slots)} slots")
            else:
                added = self._append_new_slots(new_slots)
                if added:
                    print(f"[ROI] Auto-ROI added {added} new slots (total: {len(self.slots)})")

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
    def _build_slots_from_samples(self, start_slot_id: int = 1) -> List[ParkingSlot]:
        """모아 둔 center들을 row / column 으로 나누어 자동으로 슬롯 polygon 생성."""
        if len(self._centers) == 0:
            return []

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
        slot_id = start_slot_id

        for row_pts in rows:
            if len(row_pts) < 2:
                continue
            # x 정렬 후 간격으로 width 추정 (동일 슬롯에 대한 반복 샘플 제거)
            xs = np.sort(row_pts[:, 0])
            dx = np.diff(xs)
            # 동일 슬롯에서 나온 매우 작은 간격은 제외
            valid_dx = dx[dx > self._min_slot_gap_px]
            if len(valid_dx) == 0:
                continue
            med_dx = float(np.median(valid_dx))
            if med_dx < 10:  # 이상하게 작으면 skip
                continue
            slot_width = med_dx * 0.9
            slot_height = global_med_height * 1.2

            # 가까운 center 들은 하나의 슬롯으로 병합하여 중복 생성을 방지
            merged_pts = self._merge_close_points(
                row_pts,
                min_gap=max(self._min_slot_gap_px, slot_width * 0.5))

            for cx, cy in merged_pts:
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

        return slots

    def _merge_close_points(self, row_pts: np.ndarray, min_gap: float) -> np.ndarray:
        """X축 기준으로 가까운 center들을 하나로 병합하여 중복 슬롯 생성을 방지"""
        if len(row_pts) == 0:
            return row_pts
        order = np.argsort(row_pts[:, 0])
        sorted_pts = row_pts[order]
        merged = [sorted_pts[0].astype(float)]
        for pt in sorted_pts[1:]:
            if pt[0] - merged[-1][0] < min_gap:
                merged[-1] = (merged[-1] + pt) / 2.0
            else:
                merged.append(pt.astype(float))
        return np.array(merged)

    def _point_in_existing_slots(self, x: int, y: int) -> bool:
        pt = (x, y)
        for slot in self.slots:
            if cv2.pointPolygonTest(slot.polygon, pt, False) >= 0:
                return True
        return False

    def _append_new_slots(self, candidate_slots: List[ParkingSlot]) -> int:
        added = 0
        for slot in candidate_slots:
            if self._overlaps_existing(slot):
                continue
            slot.slot_id = len(self.slots) + 1
            self.slots.append(slot)
            added += 1
        return added

    def _overlaps_existing(self, new_slot: ParkingSlot, iou_thr: float = 0.35) -> bool:
        nx1, ny1, nx2, ny2 = self._polygon_bbox(new_slot.polygon)
        new_area = max(0, nx2 - nx1) * max(0, ny2 - ny1)
        if new_area <= 0:
            return True

        for slot in self.slots:
            sx1, sy1, sx2, sy2 = self._polygon_bbox(slot.polygon)
            inter_w = max(0, min(nx2, sx2) - max(nx1, sx1))
            inter_h = max(0, min(ny2, sy2) - max(ny1, sy1))
            inter_area = inter_w * inter_h
            if inter_area == 0:
                continue
            slot_area = max(0, sx2 - sx1) * max(0, sy2 - sy1)
            if slot_area <= 0:
                continue
            union = new_area + slot_area - inter_area
            if union <= 0:
                continue
            iou = inter_area / union
            if iou >= iou_thr:
                return True
        return False

    def _polygon_bbox(self, poly: np.ndarray) -> Tuple[int, int, int, int]:
        x_coords = poly[:, 0]
        y_coords = poly[:, 1]
        return int(np.min(x_coords)), int(np.min(y_coords)), int(np.max(x_coords)), int(np.max(y_coords))
