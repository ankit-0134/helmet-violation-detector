import cv2
import numpy as np

COLOR_MAP = {
    'bike':          (255, 165,   0),   # orange
    'person':        (0,   200, 255),   # cyan
    'helmet':        (0,   230,   0),   # green
    'no_helmet':     (0,     0, 255),   # red
    'license_plate': (200,   0, 200),   # purple
}


def draw_bbox(frame: np.ndarray, bbox: list, label: str, conf: float, violation: bool = False) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    color = COLOR_MAP.get(label, (180, 180, 180))
    thickness = 3 if violation else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label} {conf:.2f}"
    font_scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    if violation:
        cv2.putText(frame, "!! VIOLATION !!", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    return frame
