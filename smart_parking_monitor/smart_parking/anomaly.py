# smart_parking/anomaly.py
from dataclasses import dataclass
from typing import Dict, List

from .tracker import Track, bbox_center
from .roi import ParkingSlotDetector, ParkingSlot

@dataclass
class Event:
    event_type: str
    track_id: int
    timestamp: float
    extra_info: Dict

class AnomalyDetector:
    def __init__(self,
                 max_outside_time: float = 10.0,
                 long_parking_time: float = 60.0):
        self.max_outside_time = max_outside_time
        self.long_parking_time = long_parking_time
        self.track_state: Dict[int, Dict] = {}

    def update_and_detect(self,
                          tracks: List[Track],
                          slots: List[ParkingSlot],
                          slot_detector: ParkingSlotDetector,
                          now: float) -> List[Event]:
        events: List[Event] = []

        for track in tracks:
            cx, cy = bbox_center(track.bbox)
            slot = slot_detector.point_in_slot(cx, cy)
            state = self.track_state.get(track.track_id, {
                "last_slot": None,
                "last_slot_enter_time": None,
                "outside_start_time": None,
            })

            if slot is not None:
                if state["last_slot"] is None or state["last_slot"].slot_id != slot.slot_id:
                    state["last_slot"] = slot
                    state["last_slot_enter_time"] = now
                else:
                    if (now - state["last_slot_enter_time"]) > self.long_parking_time:
                        events.append(Event(
                            event_type="LONG_PARKING",
                            track_id=track.track_id,
                            timestamp=now,
                            extra_info={
                                "slot_id": slot.slot_id,
                                "duration": now - state["last_slot_enter_time"],
                            },
                        ))
                        state["last_slot_enter_time"] = now
                state["outside_start_time"] = None
            else:
                if state["outside_start_time"] is None:
                    state["outside_start_time"] = now
                else:
                    if (now - state["outside_start_time"]) > self.max_outside_time:
                        events.append(Event(
                            event_type="OUTSIDE_SLOT_PARKING",
                            track_id=track.track_id,
                            timestamp=now,
                            extra_info={
                                "duration": now - state["outside_start_time"],
                            },
                        ))
                        state["outside_start_time"] = now

            self.track_state[track.track_id] = state

        return events