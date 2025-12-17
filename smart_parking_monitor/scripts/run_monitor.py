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
    parser.add_argument(
        "--frame-interval-minutes",
        type=float,
        default=30.0,
        help=("Minutes between captured frames in the source images. "
              "Use 0 to rely on real-time playback timing."),
    )
    parser.add_argument(
        "--interactive-no-parking",
        action="store_true",
        help="Show detected slots and let you choose which ones are no-parking zones.",
    )
    parser.add_argument(
        "--no-parking-slots",
        type=str,
        default=None,
        help="Comma-separated slot IDs to always treat as no-parking zones (e.g., '1,3,4').",
    )
    parser.add_argument(
        "--vlm-endpoint",
        type=str,
        default=None,
        help="HTTP endpoint of a remote VLM service (e.g., OpenAI, custom server).",
    )
    parser.add_argument(
        "--vlm-api-key",
        type=str,
        default=None,
        help="API key used for the VLM endpoint (optional).",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=12.0,
        help="Timeout (seconds) for remote VLM requests.",
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

    manual_no_parking_slots = None
    if args.no_parking_slots:
        try:
            manual_no_parking_slots = [
                int(part.strip())
                for part in args.no_parking_slots.split(",")
                if part.strip()
            ]
        except ValueError:
            raise SystemExit("Invalid --no-parking-slots value. Use comma-separated integers.")

    monitor = SmartParkingMonitor(
        use_yolo=args.use_yolo,
        video_source=video_source,
        frame_interval_minutes=args.frame_interval_minutes,
        interactive_no_parking=args.interactive_no_parking,
        manual_no_parking_slots=manual_no_parking_slots,
        vlm_endpoint=args.vlm_endpoint,
        vlm_api_key=args.vlm_api_key,
        vlm_timeout=args.vlm_timeout,
    )
    monitor.run()


if __name__ == "__main__":
    main()
