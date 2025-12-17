# scripts/run_monitor.py
"""CLI entry point for running the smart parking monitor with configurable datasets."""

import argparse
import os
import sys
from pathlib import Path

# 패키지 임포트 위해 부모 디렉토리 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from smart_parking.pipeline import SmartParkingMonitor
from smart_parking.datasets.utils import DATASET_ALIASES, resolve_dataset_path

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "videos" / "pklot_train_5fps.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Parking Monitor")
    parser.add_argument(
        "--dataset",
        default=None,
        help=("Dataset alias or path. Aliases: "
              + ", ".join(sorted(DATASET_ALIASES.keys())) + "."),
    )
    parser.add_argument(
        "--video",
        default=None,
        help=("Video file to use. If relative and --dataset is set, the path is resolved "
              "inside that dataset directory."),
    )
    parser.add_argument(
        "--use-yolo",
        dest="use_yolo",
        action="store_true",
        default=True,
        help="Use YOLO detector (default: True).",
    )
    parser.add_argument(
        "--no-yolo",
        dest="use_yolo",
        action="store_false",
        help="Disable YOLO and use the dummy detector.",
    )
    return parser.parse_args()


def resolve_video_path(video: str | None, dataset_path: Path | None) -> str:
    if video:
        path = Path(video)
        if not path.is_absolute():
            base = dataset_path if dataset_path else PROJECT_ROOT
            path = (base / video).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file '{path}' not found.")
        return str(path)

    if dataset_path:
        mp4s = sorted(dataset_path.rglob("*.mp4"))
        if mp4s:
            print(f"[Run] Using video '{mp4s[0]}' from dataset '{dataset_path}'.")
            return str(mp4s[0])
        raise FileNotFoundError(
            f"No .mp4 files found under dataset directory '{dataset_path}'. Specify --video explicitly.")

    return str(DEFAULT_VIDEO)


def main():
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    video_source = resolve_video_path(args.video, dataset_path)

    monitor = SmartParkingMonitor(
        use_yolo=args.use_yolo,
        video_source=video_source,
    )
    monitor.run()


if __name__ == "__main__":
    main()
