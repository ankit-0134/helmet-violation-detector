from ultralytics import YOLO
import numpy as np


class HelmetDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.4):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, crop: np.ndarray) -> list:
        """
        crop: cropped bike/person region
        Returns list of dicts:
          { 'class': 'helmet'|'no_helmet'|'license_plate', 'conf': float, 'bbox': [x1,y1,x2,y2] }

        ⚠️  IMPORTANT: Run `print(self.model.names)` once to see your actual class names
            and update the if/elif below to match exactly.
        """
        results = self.model(crop, conf=self.conf_threshold, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_raw = self.model.names[int(box.cls)].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            # ── Adjust these strings to match YOUR model's class names ──
            if ('no' in cls_raw and ('helmet' in cls_raw or 'head' in cls_raw)) or cls_raw == 'no-helmet':
                label = 'no_helmet'
            elif 'helmet' in cls_raw:
                label = 'helmet'
            elif any(k in cls_raw for k in ['plate', 'lp', 'number', 'licence', 'license']):
                label = 'license_plate'
            else:
                label = cls_raw  # pass through unknown classes as-is

            detections.append({
                'class': label,
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })

        return detections
