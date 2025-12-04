# smart_parking/pipeline.py
import cv2
import time

from .detector import VehicleDetector
from .tracker import SimpleTracker
from .roi import ParkingSlotDetector
from .anomaly import AnomalyDetector
from .vlm import VLMReporter
from .tracker import bbox_center

class SmartParkingMonitor:
    def __init__(self,
                 use_yolo: bool = False,
                 video_source: int | str = 0):
        self.detector = VehicleDetector(use_yolo=use_yolo)
        self.tracker = SimpleTracker()
        self.slot_detector = ParkingSlotDetector()
        self.anomaly_detector = AnomalyDetector(
            max_outside_time=10.0,
            long_parking_time=60.0
        )
        self.vlm_reporter = VLMReporter()
        self.video_source = video_source

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

            now = time.time()

            self.slot_detector.ensure_initialized(frame)

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, now)

            slots = self.slot_detector.get_slots()
            events = self.anomaly_detector.update_and_detect(
                tracks=tracks,
                slots=slots,
                slot_detector=self.slot_detector,
                now=now
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