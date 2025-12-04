# scripts/run_monitor.py
import sys
import os

# 패키지 임포트 위해 부모 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from smart_parking.pipeline import SmartParkingMonitor

def main():
    monitor = SmartParkingMonitor(
        use_yolo=False,          # YOLO 쓰려면 True 로 + 모델 준비
        video_source=0           # or "data/videos/parking_sample.mp4"
    )
    monitor.run()

if __name__ == "__main__":
    main()