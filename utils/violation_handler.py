import cv2
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ViolationRecord:
    id: int
    timestamp: float
    frame_path: str
    plate_path: Optional[str]
    frame_number: int
    plate_text: str = "Not Read"


class ViolationHandler:
    def __init__(self, save_dir: str = "violations"):
        self.save_dir = save_dir
        self.frames_dir = os.path.join(save_dir, "frames")
        self.plates_dir = os.path.join(save_dir, "plates")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.plates_dir, exist_ok=True)

        self.records: list = []
        self._counter = 0

    def save_violation(self, frame: np.ndarray, plate_crop: Optional[np.ndarray],
                       frame_number: int) -> ViolationRecord:
        self._counter += 1
        ts = time.time()
        ts_str = time.strftime("%H%M%S", time.localtime(ts))

        frame_filename = f"violation_{self._counter:04d}_{ts_str}.jpg"
        frame_path = os.path.join(self.frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        plate_path = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_filename = f"plate_{self._counter:04d}_{ts_str}.jpg"
            plate_path = os.path.join(self.plates_dir, plate_filename)
            cv2.imwrite(plate_path, plate_crop)

        record = ViolationRecord(
            id=self._counter,
            timestamp=ts,
            frame_path=frame_path,
            plate_path=plate_path,
            frame_number=frame_number,
        )
        self.records.append(record)
        return record

    def clear(self):
        self.records.clear()
        self._counter = 0
