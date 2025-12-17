# smart_parking/pipeline.py
import cv2
import time
from pathlib import Path

from .detector import VehicleDetector
from .tracker import SimpleTracker
from .roi import ParkingSlotDetector
from .anomaly import AnomalyDetector
from .vlm import VLMReporter
from .tracker import bbox_center

class SmartParkingMonitor:
    def __init__(self,
                 use_yolo: bool = False,
                 video_source: int | str = 0,
                 frame_interval_minutes: float | None = 30.0):
        self.detector = VehicleDetector(use_yolo=use_yolo)
        self.tracker = SimpleTracker()
        self.slot_detector = ParkingSlotDetector()
        self.frame_interval_minutes = (
            frame_interval_minutes if frame_interval_minutes and frame_interval_minutes > 0
            else None
        )
        self._frame_interval_seconds = (
            self.frame_interval_minutes * 60.0 if self.frame_interval_minutes else None
        )
        self._virtual_time = 0.0
        self.anomaly_detector = AnomalyDetector(
            max_outside_time=10.0,
            long_parking_time=24 * 3600.0  # 24시간
        )
        self.vlm_reporter = VLMReporter(
            frame_interval_minutes=self.frame_interval_minutes or 0.0
        )
        self.video_source = self._resolve_video_source(video_source)

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("[Error] Cannot open video source")
            return

        print("[Info] Smart Parking Monitor started")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracker_now = time.time()

            #self.slot_detector.ensure_initialized(frame)

            detections = self.detector.detect(frame)
            self.slot_detector.update_auto(frame, detections)
            tracks = self.tracker.update(detections, tracker_now)

            if self._frame_interval_seconds:
                anomaly_now = self._virtual_time
                self._virtual_time += self._frame_interval_seconds
            else:
                anomaly_now = tracker_now

            slots = self.slot_detector.get_slots()
            events = self.anomaly_detector.update_and_detect(
                tracks=tracks,
                slots=slots,
                slot_detector=self.slot_detector,
                now=anomaly_now
            )

            for ev in events:
                t = next((t for t in tracks if t.track_id == ev.track_id), None)
                if t is None:
                    continue
                msg = self.vlm_reporter.describe_event(frame, t, ev)
                print("[EVENT]", msg)

            vis_frame = frame.copy()
            # 슬롯
            for slot in slots:
                color = (0, 255, 0) if not slot.is_no_parking_zone else (0, 0, 255)
                cv2.polylines(vis_frame, [slot.polygon], True, color, 2)
                cx = int(slot.polygon[:, 0].mean())
                cy = int(slot.polygon[:, 1].mean())
                cv2.putText(vis_frame, f"S{slot.slot_id}", (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 트랙
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                cx, cy = bbox_center(track.bbox)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(vis_frame, f"ID {track.track_id}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)

            cv2.imshow("Smart Parking Monitor", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[Info] Stopped")

    def _resolve_video_source(self, source: int | str):
        """Allow relative paths regardless of the current working directory."""
        if isinstance(source, int):
            return source

        path = Path(source)
        if path.is_file():
            return str(path)

        # Try resolving relative to project root (smart_parking_monitor)
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / path
        if candidate.is_file():
            return str(candidate)

        print(f"[Warn] Video source not found: {source}")
        return str(path)
