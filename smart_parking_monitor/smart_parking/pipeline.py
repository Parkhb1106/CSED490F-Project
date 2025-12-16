# smart_parking/pipeline.py
import os
import cv2
import time

from .detector import VehicleDetector
from .tracker import SimpleTracker, bbox_center
from .roi import ParkingSlotDetector
from .anomaly import AnomalyDetector
from .vlm import VLMReporter

from .web_reporter import WebReporter, install_print_hook


class SmartParkingMonitor:
    def __init__(self, use_yolo: bool = False, video_source: int | str = 0):
        self.detector = VehicleDetector(use_yolo=use_yolo)
        self.tracker = SimpleTracker()
        self.slot_detector = ParkingSlotDetector()
        self.anomaly_detector = AnomalyDetector(
            max_outside_time=10.0,
            long_parking_time=60.0,
        )
        self.vlm_reporter = VLMReporter()
        self.video_source = video_source

        # --- Web monitor (env 기반) ---
        self.web_enabled = os.getenv("SPM_WEB_ENABLE", "0") == "1"
        self.web_url = os.getenv("SPM_WEB_URL", "http://127.0.0.1:8000")
        self.web_token = os.getenv("SPM_WEB_TOKEN")
        self.web_fps = float(os.getenv("SPM_WEB_FPS", "8"))
        self.web_width = int(os.getenv("SPM_WEB_WIDTH", "960"))
        self.web_quality = int(os.getenv("SPM_WEB_JPEG_QUALITY", "70"))

        self._web: WebReporter | None = None
        self._restore_print = None

    def _init_web(self):
        if not self.web_enabled:
            return
        self._web = WebReporter(
            self.web_url,
            token=self.web_token,
            frame_fps=self.web_fps,
            resize_width=self.web_width,
            jpeg_quality=self.web_quality,
            enabled=True,
        )
        self._restore_print = install_print_hook(self._web)
        print(f"[Web] enabled: url={self.web_url}")

    def _close_web(self):
        try:
            if self._restore_print:
                self._restore_print()
        except Exception:
            pass
        try:
            if self._web:
                self._web.close()
        except Exception:
            pass

    def run(self):
        self._init_web()

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("[Error] Cannot open video source")
            self._close_web()
            return

        print("[Info] Smart Parking Monitor started")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()

                detections = self.detector.detect(frame)
                self.slot_detector.update_auto(frame, detections)

                tracks = self.tracker.update(detections, now)
                slots = self.slot_detector.get_slots()

                events = self.anomaly_detector.update_and_detect(
                    tracks=tracks,
                    slots=slots,
                    slot_detector=self.slot_detector,
                    now=now,
                )

                # 이벤트 처리: 레포 이벤트 구조는 (event_type, track_id, timestamp, extra_info) :contentReference[oaicite:6]{index=6}
                for ev in events:
                    t = next((t for t in tracks if t.track_id == ev.track_id), None)
                    if t is None:
                        continue

                    msg = self.vlm_reporter.describe_event(frame, t, ev)
                    print("[EVENT]", msg)

                    if self._web:
                        self._web.event(
                            msg,
                            meta={
                                "event_type": ev.event_type,
                                "track_id": ev.track_id,
                                "timestamp": ev.timestamp,
                                "extra_info": ev.extra_info,
                            },
                        )

                # 시각화 프레임 만들기(기존 코드 유지) :contentReference[oaicite:7]{index=7}
                vis_frame = frame.copy()

                for slot in slots:
                    color = (0, 255, 0) if not slot.is_no_parking_zone else (0, 0, 255)
                    cv2.polylines(vis_frame, [slot.polygon], True, color, 2)
                    cx = int(slot.polygon[:, 0].mean())
                    cy = int(slot.polygon[:, 1].mean())
                    cv2.putText(
                        vis_frame,
                        f"S{slot.slot_id}",
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                for track in tracks:
                    x1, y1, x2, y2 = track.bbox
                    cx, cy = bbox_center(track.bbox)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 255), -1)
                    cv2.putText(
                        vis_frame,
                        f"ID {track.track_id}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                # --- 웹으로 영상 전송 ---
                if self._web:
                    self._web.frame(vis_frame)

                cv2.imshow("Smart Parking Monitor", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[Info] Stopped")
            self._close_web()
