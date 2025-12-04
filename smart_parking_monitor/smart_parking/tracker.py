# smart_parking/tracker.py
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from .detector import Detection

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_update_time: float
    first_seen_time: float
    history: List[Tuple[int, int]] = field(default_factory=list)

def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b1
    x1_, y1_, x2_, y2_ = b2
    xx1 = max(x1, x1_)
    yy1 = max(y1, y1_)
    xx2 = min(x2, x2_)
    yy2 = min(y2, y2_)
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    if inter == 0:
        return 0.0
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    return inter / (area1 + area2 - inter + 1e-6)

class SimpleTracker:
    def __init__(self, max_lost_time: float = 2.0, iou_threshold: float = 0.3):
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1
        self.max_lost_time = max_lost_time
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Detection], now: float) -> List[Track]:
        updated_tracks: Dict[int, Track] = {}
        unmatched_dets = set(range(len(detections)))

        for tid, track in self.tracks.items():
            best_det_idx = None
            best_iou = 0.0
            for di in unmatched_dets:
                det = detections[di]
                iou_score = iou(track.bbox, det.bbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det_idx = di

            if best_det_idx is not None and best_iou > self.iou_threshold:
                det = detections[best_det_idx]
                unmatched_dets.remove(best_det_idx)
                cx, cy = bbox_center(det.bbox)
                track.bbox = det.bbox
                track.last_update_time = now
                track.history.append((cx, cy))
                updated_tracks[tid] = track
            else:
                if now - track.last_update_time < self.max_lost_time:
                    updated_tracks[tid] = track

        for di in unmatched_dets:
            det = detections[di]
            cx, cy = bbox_center(det.bbox)
            new_track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                last_update_time=now,
                first_seen_time=now,
                history=[(cx, cy)],
            )
            updated_tracks[self.next_id] = new_track
            self.next_id += 1

        self.tracks = updated_tracks
        return list(self.tracks.values())