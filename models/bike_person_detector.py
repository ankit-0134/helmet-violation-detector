from ultralytics import YOLO
import numpy as np


class BikePersonDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.4, size_threshold: float = 0.02):
        """
        weights_path   : path to bike_person.pt
        conf_threshold : minimum confidence to keep detection
        size_threshold : minimum bbox area ratio (relative to frame) to filter distant bikes
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.size_threshold = size_threshold

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns list of dicts:
          { 'class': 'bike'|'person', 'conf': float, 'bbox': [x1,y1,x2,y2] }
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_name = self.model.names[int(box.cls)].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            bbox_area = (x2 - x1) * (y2 - y1)

            if any(k in cls_name for k in ['bike', 'motor', 'motorcycle', 'motorbike']):
                # Skip bikes that are too far (too small in frame)
                if bbox_area / frame_area < self.size_threshold:
                    continue
                label = 'bike'
            elif 'person' in cls_name:
                label = 'person'
            else:
                continue  # skip cars, trucks, etc.

            detections.append({
                'class': label,
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })

        return detections
