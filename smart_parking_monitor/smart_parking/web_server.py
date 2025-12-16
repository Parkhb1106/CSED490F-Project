# smart_parking/web_server.py
from __future__ import annotations

import asyncio
import time
from typing import Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# 토큰 사용(선택): 환경변수 SPM_WEB_TOKEN이 설정되면 요청 헤더 X-Auth-Token 검사
AUTH_TOKEN: Optional[str] = None


def _check_token(req: Request):
    if AUTH_TOKEN is None:
        return
    token = req.headers.get("X-Auth-Token")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----- WS clients -----
ws_logs: Set[WebSocket] = set()
ws_events: Set[WebSocket] = set()

# ----- latest JPEG frame -----
latest_jpeg: Optional[bytes] = None
latest_lock = asyncio.Lock()

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Smart Parking Remote Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: sans-serif; margin: 16px; display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
    img { width: 100%; border-radius: 12px; background:#000; }
    #log, #events { white-space: pre-wrap; padding: 12px; height: 42vh; overflow:auto; border-radius: 12px; }
    #log { background:#111; color:#0f0; }
    #events { background:#f4f4f4; }
    .small { color:#666; font-size: 12px; }
  </style>
</head>
<body>
  <div>
    <h2>Live Video</h2>
    <div class="small">MJPEG: /stream.mjpg</div>
    <img src="/stream.mjpg" />
  </div>

  <div>
    <h2>Events</h2>
    <div id="events"></div>

    <h2>Logs</h2>
    <div id="log"></div>
  </div>

<script>
  function append(div, msg) {
    div.textContent += msg + "\\n";
    div.scrollTop = div.scrollHeight;
  }

  const logDiv = document.getElementById("log");
  const evDiv  = document.getElementById("events");

  const wsProto = (location.protocol === "https:") ? "wss" : "ws";

  const wsLog = new WebSocket(`${wsProto}://${location.host}/ws/logs`);
  wsLog.onmessage = (e) => append(logDiv, e.data);

  const wsEv = new WebSocket(`${wsProto}://${location.host}/ws/events`);
  wsEv.onmessage = (e) => append(evDiv, e.data);
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)


async def _broadcast(sockets: Set[WebSocket], text: str):
    dead = []
    for ws in list(sockets):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        sockets.discard(ws)


@app.websocket("/ws/logs")
async def ws_logs_endpoint(ws: WebSocket):
    await ws.accept()
    ws_logs.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        ws_logs.discard(ws)


@app.websocket("/ws/events")
async def ws_events_endpoint(ws: WebSocket):
    await ws.accept()
    ws_events.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        ws_events.discard(ws)


@app.post("/api/log")
async def api_log(req: Request):
    _check_token(req)
    data = await req.json()
    line = str(data.get("line", ""))
    await _broadcast(ws_logs, line)
    return {"ok": True}


@app.post("/api/event")
async def api_event(req: Request):
    _check_token(req)
    data = await req.json()
    text = str(data.get("text", ""))
    meta = data.get("meta", {})
    # meta는 화면 표시용으로 한 줄에 같이 뿌림
    if meta:
        await _broadcast(ws_events, f"{text} | meta={meta}")
    else:
        await _broadcast(ws_events, text)
    return {"ok": True}


@app.post("/api/frame")
async def api_frame(req: Request):
    _check_token(req)
    body = await req.body()
    async with latest_lock:
        global latest_jpeg
        latest_jpeg = body
    return {"ok": True, "bytes": len(body)}


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        time.sleep(0.05)  # 최대 20fps 정도로 제한
        if latest_jpeg is None:
            continue
        frame = latest_jpeg
        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(frame)}\r\n\r\n".encode()
        yield frame + b"\r\n"


@app.get("/stream.mjpg")
def stream_mjpg():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
