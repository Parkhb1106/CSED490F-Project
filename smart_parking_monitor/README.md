# 가상환경 만들기

cd smart_parking_monitor

# 파이썬 3.10 환경 생성
conda create -n spm python=3.10

# 활성화
conda activate spm


pip install --upgrade pip
pip install -r requirements.txt

# 웹서버 실행
cd smart_parking_monitor
// optional - export SPM_WEB_TOKEN="mysecret"
python scripts/run_web_server.py

# 모니터 실행
cd smart_parking_monitor
export SPM_WEB_ENABLE=1
export SPM_WEB_URL="http://127.0.0.1:8000"
// optional - export SPM_WEB_TOKEN="mysecret"
python scripts/run_monitor.py

# 리모트 기기에서 접속 (같은 와이파이)
http://<run_monitor 돌리는 기기 IP>:8000