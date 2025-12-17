# scripts/create_video_dataset.py
"""Utility to convert image-only datasets (e.g., CNR) into MP4 videos."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2

# Allow importing project modules when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from smart_parking.datasets.utils import resolve_dataset_path  # noqa:E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
CAMERA_PATTERN = re.compile(r"(camera\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MP4 video files from image sequences inside a dataset directory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset alias or path. Example aliases: cnr, pklot.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help=("Relative path inside the dataset to start scanning for images. "
              "Defaults to the dataset root."),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=("Where to store generated videos. Relative paths are resolved inside the dataset "
              "directory. Defaults to '<dataset>/videos'."),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for the generated videos (default: 5).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames per video.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on the number of videos to generate.",
    )
    parser.add_argument(
        "--single-video",
        action="store_true",
        help="Combine every discovered image sequence into a single MP4.",
    )
    parser.add_argument(
        "--single-video-name",
        default="dataset_full.mp4",
        help=("Filename (or relative path) to use when --single-video is enabled. "
              "Defaults to 'dataset_full.mp4' inside the output directory."),
    )
    parser.add_argument(
        "--group-by-camera",
        action="store_true",
        help=("Combine all folders belonging to the same camera (e.g., camera1) "
              "into one MP4 per camera."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 files instead of skipping them.",
    )
    return parser.parse_args()


def resolve_inside_dataset(dataset_path: Path, value: str | None, default: Path) -> Path:
    if not value:
        return default
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (dataset_path / candidate).resolve()
    return candidate


def resolve_single_video_path(dataset_path: Path, output_dir: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate
    return (output_dir / candidate).resolve()


def extract_camera_key(seq_dir: Path, search_root: Path) -> str:
    """Return a consistent key for the camera folder of this sequence."""
    rel_parts = seq_dir.resolve().relative_to(search_root).parts
    for part in reversed(rel_parts):
        match = CAMERA_PATTERN.fullmatch(part)
        if match:
            return match.group(1).lower()
    return rel_parts[-1].lower()


def group_sequences_by_camera(seqs: Sequence[Path], search_root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for seq_dir in seqs:
        key = extract_camera_key(seq_dir, search_root)
        groups.setdefault(key, []).append(seq_dir)
    for key in groups:
        groups[key].sort(key=lambda p: p.resolve().relative_to(search_root).as_posix())
    return groups


def find_image_dirs(root: Path, output_dir: Path) -> Iterable[Path]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath).resolve()
        # Avoid descending into the output directory.
        dirnames[:] = [
            d for d in dirnames
            if not (Path(dirpath) / d).resolve().is_relative_to(output_dir)
        ]
        if current == output_dir or current.is_relative_to(output_dir):
            continue
        if any(Path(name).suffix.lower() in IMAGE_EXTS for name in filenames):
            yield current


def gather_images(image_dir: Path) -> list[Path]:
    return [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def build_video_from_dir(image_dir: Path,
                         output_path: Path,
                         fps: float,
                         max_frames: int | None) -> Tuple[bool, int]:
    images = gather_images(image_dir)
    if not images:
        return False, 0

    max_frames = max_frames if max_frames and max_frames > 0 else None

    video_writer = None
    frame_size = None
    frames_written = 0
    try:
        for img_path in images:
            if max_frames and frames_written >= max_frames:
                break

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if frame_size is None:
                frame_size = (frame.shape[1], frame.shape[0])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(str(output_path), FOURCC, fps, frame_size)
                if not video_writer.isOpened():
                    return False, 0
            elif (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size)

            video_writer.write(frame)
            frames_written += 1
    finally:
        if video_writer is not None:
            video_writer.release()

    if frames_written == 0:
        if output_path.exists():
            output_path.unlink()
        return False, 0

    return True, frames_written


def build_single_video(sequences: Sequence[Path],
                       output_path: Path,
                       fps: float,
                       max_frames: int | None) -> Tuple[bool, int, int]:
    """Combine all frames from the provided directories into one MP4."""
    video_writer = None
    frame_size = None
    frames_written = 0
    dirs_used = 0

    max_frames = max_frames if max_frames and max_frames > 0 else None

    try:
        for seq_dir in sequences:
            images = gather_images(seq_dir)
            if not images:
                continue
            dirs_used += 1
            for img_path in images:
                if max_frames and frames_written >= max_frames:
                    break

                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                if frame_size is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = cv2.VideoWriter(str(output_path), FOURCC, fps, frame_size)
                    if not video_writer.isOpened():
                        return False, 0, 0
                elif (frame.shape[1], frame.shape[0]) != frame_size:
                    frame = cv2.resize(frame, frame_size)

                video_writer.write(frame)
                frames_written += 1

            if max_frames and frames_written >= max_frames:
                break
    finally:
        if video_writer is not None:
            video_writer.release()

    if frames_written == 0:
        if output_path.exists():
            output_path.unlink()
        return False, 0, 0

    return True, frames_written, dirs_used


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("FPS must be a positive number.")
    if args.single_video and args.group_by_camera:
        raise ValueError("Choose either --single-video or --group-by-camera, not both.")
    dataset_path = resolve_dataset_path(args.dataset)
    if dataset_path is None:
        raise ValueError("Dataset path could not be resolved.")

    search_root = resolve_inside_dataset(dataset_path, args.image_root, dataset_path)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Image root '{search_root}' does not exist.")

    default_output = dataset_path / "videos"
    output_dir = resolve_inside_dataset(dataset_path, args.output_dir, default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(
        find_image_dirs(search_root, output_dir),
        key=lambda p: p.resolve().relative_to(search_root).as_posix()
    )
    if not sequences:
        print(f"[Warn] No image directories found under '{search_root}'.")
        return

    if args.group_by_camera:
        groups = group_sequences_by_camera(sequences, search_root)
        max_videos = args.max_videos if args.max_videos and args.max_videos > 0 else None
        generated = 0
        skipped = 0
        for camera_name in sorted(groups.keys()):
            if max_videos and generated >= max_videos:
                break
            target_path = output_dir / f"{camera_name}.mp4"
            if target_path.exists() and not args.overwrite:
                print(f"[Skip] Camera '{camera_name}' target '{target_path}' exists.")
                skipped += 1
                continue
            ok, frame_cnt, dirs_used = build_single_video(groups[camera_name], target_path, args.fps, args.max_frames)
            if ok:
                generated += 1
                print(f"[Write] Camera '{camera_name}' -> {target_path} ({frame_cnt} frames from {dirs_used} folders)")
            else:
                print(f"[Warn] Unable to build camera video for '{camera_name}'.")
        print(f"[Done] Generated {generated} camera video(s), skipped {skipped}. Output stored in '{output_dir}'.")
        return

    if args.single_video:
        target_path = resolve_single_video_path(dataset_path, output_dir, args.single_video_name)
        if target_path.exists() and not args.overwrite:
            print(f"[Skip] Single video target '{target_path}' exists. Use --overwrite to replace it.")
            return

        ok, frame_cnt, dirs_used = build_single_video(sequences, target_path, args.fps, args.max_frames)
        if ok:
            print(f"[Write] Combined {dirs_used} folder(s) into '{target_path}' ({frame_cnt} frames).")
        else:
            print(f"[Warn] Unable to build combined video at '{target_path}'.")
        return

    max_videos = args.max_videos if args.max_videos and args.max_videos > 0 else None
    generated = 0
    skipped = 0

    for seq_dir in sequences:
        if max_videos and generated >= max_videos:
            break

        relative = seq_dir.relative_to(search_root)
        output_path = (output_dir / relative).with_suffix(".mp4")

        if output_path.exists() and not args.overwrite:
            print(f"[Skip] {relative} -> {output_path} (exists)")
            skipped += 1
            continue

        ok, frame_count = build_video_from_dir(seq_dir, output_path, args.fps, args.max_frames)
        if ok:
            generated += 1
            print(f"[Write] {relative} -> {output_path} ({frame_count} frames)")
        else:
            print(f"[Warn] No valid frames found under '{seq_dir}'.")

    print(f"[Done] Generated {generated} video(s), skipped {skipped}. Output stored in '{output_dir}'.")


if __name__ == "__main__":
    main()
