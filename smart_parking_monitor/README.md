# 가상환경 만들기

cd smart_parking_monitor

# 파이썬 3.10 환경 생성
conda create -n spm python=3.10

# 활성화
conda activate spm

pip install --upgrade pip
pip install -r requirements.txt

# 데이터셋 로드
python data/loadPKLot.py

# 웹서버 실행
// optional - export SPM_WEB_TOKEN="mysecret"
export SPM_WEB_HOST=127.0.0.1
export SPM_WEB_PORT=8000
python scripts/run_web_server.py

# 모니터 실행
// optional - export SPM_WEB_TOKEN="mysecret"
export SPM_WEB_ENABLE=1
export SPM_WEB_URL="http://127.0.0.1:8000"
python scripts/run_monitor.py

# 리모트 기기에서 접속 (vast ai 서버 이용 시)
pip install -r requirements.txt

# 리모트 기기에서 접속 (같은 와이파이)
http://<run_monitor 돌리는 기기 IP>:8000