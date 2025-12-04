# smart_parking/vlm.py
from .tracker import Track, bbox_center
from .anomaly import Event
import numpy as np

class VLMReporter:
    """
    실제 VLM 대신, 현재는 event + bbox 정보를 바탕으로 문장 생성만.
    나중에 LLaVA / Qwen-VL 같은 모델 붙이면 됨.
    """
    def describe_event(self, frame: np.ndarray, track: Track, event: Event) -> str:
        cx, cy = bbox_center(track.bbox)

        if event.event_type == "OUTSIDE_SLOT_PARKING":
            dur = event.extra_info.get("duration", 0)
            msg = (
                f"차량 ID {track.track_id}가 주차 구역 밖에 "
                f"{dur:.1f}초 이상 정차 중입니다. (위치: ({cx}, {cy}))"
            )
        elif event.event_type == "LONG_PARKING":
            slot_id = event.extra_info.get("slot_id", -1)
            dur = event.extra_info.get("duration", 0)
            msg = (
                f"차량 ID {track.track_id}가 슬롯 {slot_id}에 "
                f"{dur:.1f}초 이상 장기 주차 중입니다."
            )
        else:
            msg = f"차량 ID {track.track_id} 관련 이벤트: {event.event_type}"

        return msg