# 가상환경 만들기

cd smart_parking_monitor

# 파이썬 3.10 환경 생성
conda create -n spm python=3.10

# 활성화
conda activate spm

# python library 설치
pip install --upgrade pip
pip install -r requirements.txt

# 데이터셋 준비

# 실행
cd smart_parking_monitor

python scripts/run_monitor.py --dataset pklot
python scripts/run_monitor.py --dataset cnr --video videos/camera4.mp4
python scripts/run_monitor.py --dataset cnr --video videos/camera5.mp4
# 이미지 프레임 간 간격 설정 (예시: 30분마다 촬영된 영상)
python smart_parking_monitor/scripts/run_monitor.py --dataset cnr --video videos/camera4.mp4 --frame-interval-minutes 30
# 주차 금지 구역을 직접 지정하고 싶다면:
python smart_parking_monitor/scripts/run_monitor.py --dataset cnr --video videos/camera4.mp4 --interactive-no-parking
# 이미 알고 있는 슬롯 번호가 있다면 (예시): --no-parking-slots 1,4
# 실제 VLM API를 사용하려면 --vlm-endpoint https://your-server/vlm (--vlm-api-key KEY) 를 추가하세요.
#    또는 SMART_PARKING_VLM_ENDPOINT / SMART_PARKING_VLM_API_KEY 환경 변수로도 설정 가능합니다.

## CNR 이미지 → 동영상 만들기
CNR 데이터셋은 기본적으로 이미지 프레임만 제공하므로, `scripts/create_video_dataset.py`를 이용해 원하는 구간을 MP4로 변환할 수 있습니다.

```bash
# 예시: OVERCAST/2015-12-19/camera1 폴더 이미지를 5fps, 최대 300프레임짜리 영상으로 생성
python3 scripts/create_video_dataset.py \
  --dataset cnr \
  --image-root CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-12-19/camera1 \
  --output-dir videos \
  --fps 5 \
  --max-frames 300 \
  --max-videos 1
```

- `--dataset cnr` : `datasets/CNR` 경로 별칭을 사용합니다.
- `--image-root`  : 이미지가 들어있는 서브폴더를 지정합니다. `FULL_IMAGE_1000x750` 전체를 주면 모든 `camera*` 폴더가 순회됩니다.
- `--output-dir`  : 생성된 영상을 저장할 폴더(기본값은 `<dataset>/videos`).
- `--max-frames`, `--max-videos` : 너무 큰 영상을 만들고 싶지 않을 때 제한을 둘 수 있습니다.
- 이미 동일한 MP4가 있다면 기본적으로 건너뛰며, 덮어쓰려면 `--overwrite`를 지정합니다.

데이터셋 전체를 PKLot처럼 한 개의 영상으로 묶고 싶다면 `--single-video` 플래그를 사용합니다.

```bash
python3 scripts/create_video_dataset.py \
  --dataset cnr \
  --image-root CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750 \
  --output-dir videos \
  --single-video \
  --single-video-name cnr_full.mp4 \
  --fps 5
```

- `--single-video` : 모든 카메라/날짜 폴더를 순서대로 이어붙여 하나의 MP4를 만듭니다.
- `--single-video-name` : 결과 파일명을 지정합니다(기본 `dataset_full.mp4`, `--output-dir` 하위 경로로 저장).

카메라별로 날짜 전체를 모아 각각 하나의 영상으로 만들고 싶다면 `--group-by-camera` 옵션을 사용합니다.

```bash
python3 scripts/create_video_dataset.py \
  --dataset cnr \
  --image-root CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750 \
  --output-dir videos \
  --group-by-camera \
  --fps 5
```

`camera1`, `camera2` ... 와 같이 폴더명이 붙은 모든 이미지들이 각각 하나의 MP4로 저장되며, `--max-videos` 옵션을 사용하면 생성할 카메라 수를 제한할 수 있습니다.

영상이 생성되면 `python scripts/run_monitor.py --dataset cnr --video datasets/CNR/videos/OVERCAST/.../camera1.mp4` 처럼 실행하면 됩니다.
