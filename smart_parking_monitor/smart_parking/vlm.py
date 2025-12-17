# smart_parking/vlm.py
"""
Generate event descriptions by optionally leveraging a remote Visual Language Model.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests

from .anomaly import Event
from .tracker import Track, bbox_center


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


@dataclass
class RemoteVLMConfig:
    endpoint: str
    api_key: Optional[str] = None
    timeout: float = 12.0


class VLMReporter:
    """
    실제 Vision-Language 모델과의 통합을 위한 헬퍼.
    - remote_config가 설정되어 있으면 해당 엔드포인트로 이미지/메타데이터를 전달해
      자연어 설명을 받아온다.
    - 설정되지 않거나 요청이 실패하면 템플릿 기반 메시지를 사용한다.
    """

    def __init__(self,
                 frame_interval_minutes: float = 30.0,
                 remote_endpoint: Optional[str] = None,
                 remote_api_key: Optional[str] = None,
                 remote_timeout: float = 12.0):
        self.frame_interval_minutes = frame_interval_minutes
        endpoint = remote_endpoint or os.getenv("SMART_PARKING_VLM_ENDPOINT")
        api = remote_api_key or os.getenv("SMART_PARKING_VLM_API_KEY")
        if endpoint:
            self.remote_config = RemoteVLMConfig(endpoint, api_key=api, timeout=remote_timeout)
        else:
            self.remote_config = None

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
        template = self._template_message(track, event)
        if not self.remote_config:
            return template

        remote_msg = self._call_remote_vlm(frame, track, event, template)
        return remote_msg or template

    # ------------------------------------------------------------------
    def _template_message(self, track: Track, event: Event) -> str:
        cx, cy = bbox_center(track.bbox)
        dur_seconds = event.extra_info.get("duration", 0)
        dur_text = self._duration_text(dur_seconds)

        if event.event_type == "OUTSIDE_SLOT_PARKING":
            return (
                f"차량 ID {track.track_id}가 주차 구역 밖에 "
                f"{dur_text} 이상 정차 중입니다. "
                f"({self._interval_note()}, 위치: ({cx}, {cy}))"
            )
        if event.event_type == "LONG_PARKING":
            slot_id = event.extra_info.get("slot_id", -1)
            return (
                f"차량 ID {track.track_id}가 슬롯 {slot_id}에 "
                f"{dur_text} 이상 장기 주차 중입니다. "
                f"({self._interval_note()})"
            )
        if event.event_type == "NO_PARKING_ZONE":
            slot_id = event.extra_info.get("slot_id", -1)
            return (
                f"차량 ID {track.track_id}가 주차 금지 구역 슬롯 {slot_id}에 "
                f"{dur_text} 동안 머물러 있습니다. "
                f"({self._interval_note()}, 위치: ({cx}, {cy}))"
            )

        return f"차량 ID {track.track_id} 관련 이벤트: {event.event_type}"

    def _call_remote_vlm(self,
                         frame: np.ndarray,
                         track: Track,
                         event: Event,
                         fallback: str) -> Optional[str]:
        if not self.remote_config:
            return None

        payload = {
            "prompt": self._build_prompt(track, event, fallback),
            "event_type": event.event_type,
            "track_id": track.track_id,
            "duration_seconds": event.extra_info.get("duration", 0),
            "frame_interval_minutes": self.frame_interval_minutes,
        }
        image_b64 = self._encode_crop(frame, track.bbox)
        if image_b64:
            payload["image_base64"] = image_b64

        headers = {"Content-Type": "application/json"}
        if self.remote_config.api_key:
            headers["Authorization"] = f"Bearer {self.remote_config.api_key}"

        try:
            response = requests.post(
                self.remote_config.endpoint,
                json=payload,
                headers=headers,
                timeout=self.remote_config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message") or data.get("description")
        except Exception as exc:  # noqa: BLE001
            print(f"[VLM] Remote call failed ({exc}); falling back to template output.")
            return None

    def _build_prompt(self, track: Track, event: Event, fallback: str) -> str:
        cx, cy = bbox_center(track.bbox)
        description_hint = (
            "현재 이미지는 스마트 주차 감시 카메라에서 추출한 차량 주변 장면입니다. "
            "이미지를 보고 상황을 한글로 설명하고, 왜 문제가 되는지 짧게 언급해주세요."
        )
        meta = (
            f"이벤트 유형: {event.event_type}\n"
            f"차량 ID: {track.track_id}\n"
            f"위치 중심 좌표: ({cx}, {cy})\n"
            f"프레임 간 간격: {self.frame_interval_minutes}분\n"
            f"기본 설명(백업): {fallback}"
        )
        return description_hint + "\n" + meta

    def _encode_crop(self, frame: np.ndarray, bbox) -> Optional[str]:
        if frame is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            crop = frame
        else:
            crop = frame[y1:y2, x1:x2]

        success, buffer = cv2.imencode(".jpg", crop)
        if not success:
            return None
        return base64.b64encode(buffer).decode("utf-8")
