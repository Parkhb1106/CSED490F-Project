# 가상환경 만들기

cd smart_parking_monitor

# 파이썬 3.10 환경 생성
conda create -n spm python=3.10

# 활성화
conda activate spm


pip install --upgrade pip
pip install -r requirements.txt

# 실행하려면
cd smart_parking_monitor
python scripts/run_monitor.py