# scripts/local_vlm_server.py
"""Minimal local VLM-like server for development/testing.

This HTTP service mimics the API contract expected by VLMReporter.
It does NOT run a real multimodal model, but it inspects the provided metadata
and generates a contextual Korean description so the pipeline can exercise the
remote-path code without external dependencies.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import logging
from typing import Any, Dict

from flask import Flask, jsonify, request

app = Flask(__name__)
LOGGER = logging.getLogger("local_vlm")


def _build_message(payload: Dict[str, Any]) -> str:
    event_type = payload.get("event_type", "UNKNOWN")
    track_id = payload.get("track_id", "N/A")
    duration = payload.get("duration_seconds", 0.0)
    interval = payload.get("frame_interval_minutes", 0.0)

    # Check whether an image was attached (just to mention it in the description)
    image_info = "이미지 없음"
    image_b64 = payload.get("image_base64")
    if image_b64:
        try:
            decoded = base64.b64decode(image_b64, validate=True)
            image_info = f"이미지 {len(decoded)}바이트 수신"
        except base64.binascii.Error:
            image_info = "이미지 디코딩 실패"

    event_desc = {
        "OUTSIDE_SLOT_PARKING": "주차 구역 밖에서 장시간 정차",
        "LONG_PARKING": "슬롯에서 장기 주차",
        "NO_PARKING_ZONE": "주차 금지 구역 침범",
    }.get(event_type, "알 수 없는 상태")

    minutes = duration / 60.0 if duration else 0.0
    now = dt.datetime.now().strftime("%H:%M:%S")
    return (
        f"[VLM {now}] 차량 ID {track_id} 감지: "
        f"{event_desc} (약 {minutes:.1f}분 경과, 프레임 간격 {interval}분). "
        f"{image_info}."
    )


@app.post("/vlm")
def vlm_endpoint():
    payload = request.get_json(force=True, silent=True) or {}
    LOGGER.info("Received VLM request from track %s", payload.get("track_id"))
    message = _build_message(payload)
    return jsonify({"message": message})


def main():
    parser = argparse.ArgumentParser(description="Run a dummy local VLM HTTP server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on (default: 5001)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode for easier iteration.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print(f"[VLM] Serving on http://{args.host}:{args.port}/vlm")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
