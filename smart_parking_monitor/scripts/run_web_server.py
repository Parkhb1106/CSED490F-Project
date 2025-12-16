# scripts/run_web_server.py
import os
import sys

# 패키지 임포트 위해 부모 디렉토리 추가 (run_monitor.py와 동일 패턴) :contentReference[oaicite:4]{index=4}
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from smart_parking.web_server import app
import smart_parking.web_server as ws_mod

def main():
    host = os.getenv("SPM_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("SPM_WEB_PORT", "8000"))
    ws_mod.AUTH_TOKEN = os.getenv("SPM_WEB_TOKEN")  # 없으면 인증 없이 동작

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
