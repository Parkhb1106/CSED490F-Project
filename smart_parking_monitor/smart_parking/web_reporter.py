# smart_parking/web_reporter.py
from __future__ import annotations

import builtins
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class _Msg:
    kind: str  # "log" | "event" | "frame"
    payload: Any
    ts: float


class WebReporter:
    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        frame_fps: float = 8.0,
        jpeg_quality: int = 70,
        resize_width: Optional[int] = 960,
        queue_size: int = 200,
        enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.enabled = enabled

        self.frame_interval = 1.0 / max(frame_fps, 0.1)
        self.jpeg_quality = int(jpeg_quality)
        self.resize_width = resize_width

        self._last_frame_ts = 0.0
        self._q: queue.Queue[_Msg] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)

        if self.enabled:
            self._t.start()

    def close(self):
        if not self.enabled:
            return
        self._stop.set()
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass

    def _headers(self) -> dict:
        h = {}
        if self.token:
            h["X-Auth-Token"] = self.token
        return h

    def log(self, line: str):
        if not self.enabled:
            return
        try:
            self._q.put_nowait(_Msg("log", {"line": line}, time.time()))
        except queue.Full:
            pass

    def event(self, text: str, meta: Optional[dict] = None):
        if not self.enabled:
            return
        try:
            self._q.put_nowait(_Msg("event", {"text": text, "meta": meta or {}}, time.time()))
        except queue.Full:
            pass

    def frame(self, bgr_frame):
        if not self.enabled or cv2 is None:
            return
        now = time.time()
        if now - self._last_frame_ts < self.frame_interval:
            return
        self._last_frame_ts = now

        # 프레임은 큐가 꽉 차면 드롭(실시간성 유지)
        try:
            self._q.put_nowait(_Msg("frame", {"frame": bgr_frame}, now))
        except queue.Full:
            pass

    def _worker(self):
        sess = requests.Session()
        while not self._stop.is_set():
            try:
                msg = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if msg.kind == "log":
                    sess.post(
                        f"{self.base_url}/api/log",
                        json=msg.payload,
                        headers=self._headers(),
                        timeout=0.5,
                    )

                elif msg.kind == "event":
                    sess.post(
                        f"{self.base_url}/api/event",
                        json=msg.payload,
                        headers=self._headers(),
                        timeout=1.0,
                    )

                elif msg.kind == "frame":
                    frame = msg.payload["frame"]

                    if self.resize_width is not None:
                        h, w = frame.shape[:2]
                        if w > 0 and w != self.resize_width:
                            nh = int(h * (self.resize_width / w))
                            frame = cv2.resize(frame, (self.resize_width, nh))

                    ok, enc = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if not ok:
                        continue

                    sess.post(
                        f"{self.base_url}/api/frame",
                        data=enc.tobytes(),
                        headers={**self._headers(), "Content-Type": "image/jpeg"},
                        timeout=0.5,
                    )
            except Exception:
                # 네트워크 문제로 모니터링이 멈추면 안 되므로 조용히 무시
                pass


def install_print_hook(reporter: WebReporter):
    """
    print(...)를 그대로 출력하면서, 동시에 reporter.log로도 전송.
    반환값: restore() 함수
    """
    orig_print = builtins.print

    def hooked_print(*args, **kwargs):
        orig_print(*args, **kwargs)
        try:
            sep = kwargs.get("sep", " ")
            msg = sep.join(str(a) for a in args)
            reporter.log(msg)
        except Exception:
            pass

    builtins.print = hooked_print

    def restore():
        builtins.print = orig_print

    return restore
