# smart_parking/vlm.py
from .tracker import Track, bbox_center
from .anomaly import Event
import numpy as np


def _format_minutes(minutes: float) -> str:
    """
    Convert minutes into a short human-readable text such as '1시간 30.0분' or '15.0분'.
    """
    minutes = max(minutes, 0.0)
    if minutes >= 60.0:
        hours = int(minutes // 60)
        remain = minutes - (hours * 60)
        if remain < 1e-2:
            return f"{hours}시간"
        return f"{hours}시간 {remain:.1f}분"
    return f"{minutes:.1f}분"


class VLMReporter:
    """
    실제 VLM 대신, 현재는 event + bbox 정보를 바탕으로 문장 생성만.
    나중에 LLaVA / Qwen-VL 같은 모델 붙이면 됨.
    """
    def __init__(self, frame_interval_minutes: float = 30.0):
        # frame_interval_minutes는 각 프레임이 몇 분 간격으로 촬영되었는지 나타낸다.
        # 현재 데이터셋은 30분 간격 이미지 기반 영상이므로 기본값을 30으로 사용한다.
        self.frame_interval_minutes = frame_interval_minutes

    def _duration_minutes(self, duration_seconds: float) -> float:
        return max(duration_seconds, 0.0) / 60.0

    def _duration_text(self, duration_seconds: float) -> str:
        minutes = self._duration_minutes(duration_seconds)
        return _format_minutes(minutes)

    def _interval_note(self) -> str:
        if self.frame_interval_minutes > 0:
            return f"프레임 간격 {self.frame_interval_minutes:g}분 기준"
        return "실시간 경과 기준"

    def describe_event(self, frame: np.ndarray, track: Track, event: Event) -> str:
        cx, cy = bbox_center(track.bbox)

        if event.event_type == "OUTSIDE_SLOT_PARKING":
            dur_seconds = event.extra_info.get("duration", 0)
            dur_text = self._duration_text(dur_seconds)
            msg = (
                f"차량 ID {track.track_id}가 주차 구역 밖에 "
                f"{dur_text} 이상 정차 중입니다. "
                f"({self._interval_note()}, 위치: ({cx}, {cy}))"
            )
        elif event.event_type == "LONG_PARKING":
            slot_id = event.extra_info.get("slot_id", -1)
            dur_seconds = event.extra_info.get("duration", 0)
            dur_text = self._duration_text(dur_seconds)
            msg = (
                f"차량 ID {track.track_id}가 슬롯 {slot_id}에 "
                f"{dur_text} 이상 장기 주차 중입니다. "
                f"({self._interval_note()})"
            )
        elif event.event_type == "NO_PARKING_ZONE":
            slot_id = event.extra_info.get("slot_id", -1)
            dur_seconds = event.extra_info.get("duration", 0)
            dur_text = self._duration_text(dur_seconds)
            msg = (
                f"차량 ID {track.track_id}가 주차 금지 구역 슬롯 {slot_id}에 "
                f"{dur_text} 동안 머물러 있습니다. "
                f"({self._interval_note()}, 위치: ({cx}, {cy}))"
            )
        else:
            msg = f"차량 ID {track.track_id} 관련 이벤트: {event.event_type}"

        return msg
