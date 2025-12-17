# smart_parking/pipeline.py
import cv2
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .detector import VehicleDetector
from .tracker import SimpleTracker
from .roi import ParkingSlotDetector
from .anomaly import AnomalyDetector
from .vlm import VLMReporter
from .tracker import bbox_center


@dataclass
class FramePacket:
    frame: Any
    tracker_now: float

class SmartParkingMonitor:
    def __init__(self,
                 use_yolo: bool = False,
                 video_source: int | str = 0,
                 frame_interval_minutes: float | None = 30.0,
                 interactive_no_parking: bool = False,
                 manual_no_parking_slots: List[int] | None = None,
                 vlm_endpoint: str | None = None,
                 vlm_api_key: str | None = None,
                 vlm_timeout: float = 12.0,
                 detector_device: str | None = None,
                 detector_precision: str | None = None,
                 async_detection: bool = True,
                 async_vlm: bool = True):
        self.detector = VehicleDetector(
            use_yolo=use_yolo,
            device=detector_device or "auto",
            precision=detector_precision,
        )
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
        self.interactive_no_parking = interactive_no_parking
        self.manual_no_parking_slots: List[int] = list(manual_no_parking_slots or [])
        self._no_parking_configured = not (self.interactive_no_parking or self.manual_no_parking_slots)
        self.anomaly_detector = AnomalyDetector(
            max_outside_time=60 * 60.0,  # 1시간
            long_parking_time=24 * 3600.0  # 24시간
        )
        self.vlm_reporter = VLMReporter(
            frame_interval_minutes=self.frame_interval_minutes or 0.0,
            remote_endpoint=vlm_endpoint,
            remote_api_key=vlm_api_key,
            remote_timeout=vlm_timeout,
            async_mode=async_vlm,
        )
        self.video_source = self._resolve_video_source(video_source)
        self._waiting_for_slots_logged = False
        self.async_detection = async_detection
        self._stop_requested = False

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("[Error] Cannot open video source")
            return

        print("[Info] Smart Parking Monitor started")

        detector_executor: Optional[ThreadPoolExecutor] = None
        future: Optional[Future] = None
        pending_packet: Optional[FramePacket] = None

        if self.async_detection:
            detector_executor = ThreadPoolExecutor(max_workers=1)

        try:
            if detector_executor:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    packet = FramePacket(frame=frame, tracker_now=time.time())
                    if future is None:
                        future = detector_executor.submit(self.detector.detect, frame)
                        pending_packet = packet
                        continue

                    detections = future.result()
                    if pending_packet and not self._process_frame(
                        pending_packet.frame, detections, pending_packet.tracker_now
                    ):
                        self._stop_requested = True
                        break

                    future = detector_executor.submit(self.detector.detect, frame)
                    pending_packet = packet

                if (
                    not self._stop_requested
                    and future is not None
                    and pending_packet is not None
                ):
                    detections = future.result()
                    self._process_frame(pending_packet.frame, detections, pending_packet.tracker_now)
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tracker_now = time.time()
                    detections = self.detector.detect(frame)
                    if not self._process_frame(frame, detections, tracker_now):
                        self._stop_requested = True
                        break
        finally:
            if detector_executor:
                detector_executor.shutdown(wait=True)
            cap.release()
            cv2.destroyAllWindows()
            self.vlm_reporter.shutdown()
            print("[Info] Stopped")

    def _process_frame(self, frame, detections, tracker_now: float) -> bool:
        self.slot_detector.update_auto(frame, detections)
        tracks = self.tracker.update(detections, tracker_now)

        if self._frame_interval_seconds:
            anomaly_now = self._virtual_time
            self._virtual_time += self._frame_interval_seconds
        else:
            anomaly_now = tracker_now

        slots = self.slot_detector.get_slots()
        events = []
        if self.slot_detector.is_initialized():
            if not self._no_parking_configured:
                if self.manual_no_parking_slots:
                    self._apply_no_parking_slots(self.manual_no_parking_slots)
                    self.manual_no_parking_slots = []
                if self.interactive_no_parking:
                    self._prompt_no_parking_slots(frame, slots)
                self._no_parking_configured = True

            events = self.anomaly_detector.update_and_detect(
                tracks=tracks,
                slots=slots,
                slot_detector=self.slot_detector,
                now=anomaly_now
            )
            active_event_tracks = self.anomaly_detector.get_active_violation_track_ids()
        else:
            if not self._waiting_for_slots_logged:
                print("[Info] Waiting for slot detection before processing events...")
                self._waiting_for_slots_logged = True
            active_event_tracks = set()

        for ev in events:
            t = next((t for t in tracks if t.track_id == ev.track_id), None)
            if t is None:
                continue
            self.vlm_reporter.report_event(
                frame,
                t,
                ev,
                callback=lambda msg: print("[EVENT]", msg)
            )

        vis_frame = frame.copy()
        for slot in slots:
            color = (0, 255, 0) if not slot.is_no_parking_zone else (0, 0, 255)
            cv2.polylines(vis_frame, [slot.polygon], True, color, 2)
            cx = int(slot.polygon[:, 0].mean())
            cy = int(slot.polygon[:, 1].mean())
            cv2.putText(vis_frame, f"S{slot.slot_id}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cx, cy = bbox_center(track.bbox)
            color = (0, 165, 255) if track.track_id in active_event_tracks else (255, 255, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(vis_frame, f"ID {track.track_id}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        cv2.imshow("Smart Parking Monitor", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        return True

    def _apply_no_parking_slots(self, slot_ids: Iterable[int]):
        slots = self.slot_detector.get_slots()
        slot_dict = {slot.slot_id: slot for slot in slots}
        applied = []
        missing = []
        for sid in slot_ids:
            slot = slot_dict.get(sid)
            if slot is None:
                missing.append(sid)
                continue
            slot.is_no_parking_zone = True
            applied.append(sid)

        if applied:
            print(f"[Config] Marked slots as no-parking: {applied}")
        if missing:
            print(f"[Warn] Slot IDs not found (cannot mark no-parking): {missing}")

    def _prompt_no_parking_slots(self, frame, slots: List):
        if not slots:
            print("[Warn] No slots available for manual no-parking selection.")
            return

        vis = frame.copy()
        for slot in slots:
            color = (0, 0, 255) if slot.is_no_parking_zone else (0, 255, 0)
            cv2.polylines(vis, [slot.polygon], True, color, 2)
            cx = int(slot.polygon[:, 0].mean())
            cy = int(slot.polygon[:, 1].mean())
            label = f"S{slot.slot_id}"
            if slot.is_no_parking_zone:
                label += " (NP)"
            cv2.putText(vis, label, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        window_name = "Select No-Parking Slots"
        cv2.imshow(window_name, vis)
        print("\n[Config] 주차 금지 구역으로 지정할 슬롯을 선택하세요.")
        print("창에서 슬롯 번호를 확인한 뒤 아무 키나 눌러 창을 닫으면 됩니다.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        print("현재 감지된 슬롯 ID:", ", ".join(str(slot.slot_id) for slot in slots))
        while True:
            raw = input("주차 금지 슬롯 ID를 콤마로 구분해 입력 (예: 1,3,5, 없으면 Enter): ").strip()
            if not raw:
                print("[Config] 주차 금지 구역을 추가로 지정하지 않습니다.")
                break
            try:
                ids = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
            except ValueError:
                print("[Warn] 숫자만 입력해주세요.")
                continue
            if not ids:
                print("[Config] 입력된 숫자가 없습니다. 다시 입력해주세요.")
                continue
            self._apply_no_parking_slots(ids)
            break


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
