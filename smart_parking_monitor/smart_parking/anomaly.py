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
                "no_parking_slot_id": None,
                "no_parking_start_time": None,
                "no_parking_alerted": False,
                "outside_alerted": False,
                "long_alerted": False,
                "outside_violation": False,
                "long_violation": False,
                "no_parking_violation": False,
            })

            if slot is not None:
                if state["last_slot"] is None or state["last_slot"].slot_id != slot.slot_id:
                    state["last_slot"] = slot
                    state["last_slot_enter_time"] = now
                    state["long_alerted"] = False
                duration_in_slot = 0.0
                if state["last_slot_enter_time"] is not None:
                    duration_in_slot = now - state["last_slot_enter_time"]
                if duration_in_slot >= self.long_parking_time and not state["long_alerted"]:
                    events.append(Event(
                        event_type="LONG_PARKING",
                        track_id=track.track_id,
                        timestamp=now,
                        extra_info={
                            "slot_id": slot.slot_id,
                            "duration": duration_in_slot,
                        },
                    ))
                    state["long_alerted"] = True
                state["long_violation"] = duration_in_slot >= self.long_parking_time
                # reset outside tracking because we are within a slot
                state["outside_start_time"] = None
                state["outside_alerted"] = False
                state["outside_violation"] = False

                if slot.is_no_parking_zone:
                    if state["no_parking_slot_id"] != slot.slot_id:
                        state["no_parking_slot_id"] = slot.slot_id
                        state["no_parking_start_time"] = now
                        state["no_parking_alerted"] = False
                    if not state["no_parking_alerted"]:
                        duration = 0.0
                        if state["no_parking_start_time"] is not None:
                            duration = now - state["no_parking_start_time"]
                        events.append(Event(
                            event_type="NO_PARKING_ZONE",
                            track_id=track.track_id,
                            timestamp=now,
                            extra_info={
                                "slot_id": slot.slot_id,
                                "duration": duration,
                            },
                        ))
                        state["no_parking_alerted"] = True
                    state["no_parking_violation"] = True
                else:
                    state["no_parking_slot_id"] = None
                    state["no_parking_start_time"] = None
                    state["no_parking_alerted"] = False
                    state["no_parking_violation"] = False
            else:
                state["last_slot"] = None
                state["long_violation"] = False
                state["long_alerted"] = False
                if state["outside_start_time"] is None:
                    state["outside_start_time"] = now
                    state["outside_alerted"] = False
                duration_outside = now - state["outside_start_time"]
                if duration_outside >= self.max_outside_time and not state["outside_alerted"]:
                    events.append(Event(
                        event_type="OUTSIDE_SLOT_PARKING",
                        track_id=track.track_id,
                        timestamp=now,
                        extra_info={
                            "duration": duration_outside,
                        },
                    ))
                    state["outside_alerted"] = True
                state["outside_violation"] = duration_outside >= self.max_outside_time
                state["no_parking_slot_id"] = None
                state["no_parking_start_time"] = None
                state["no_parking_alerted"] = False
                state["no_parking_violation"] = False

            self.track_state[track.track_id] = state

        return events

    def get_active_violation_track_ids(self) -> set[int]:
        active = set()
        for track_id, state in self.track_state.items():
            if state.get("outside_violation") or state.get("long_violation") or state.get("no_parking_violation"):
                active.add(track_id)
        return active
