import cv2
import numpy as np
import time
from models.bike_person_detector import BikePersonDetector
from models.helmet_detector import HelmetDetector
from utils.drawing import draw_bbox
from utils.violation_handler import ViolationHandler

ZONE_TOP    = 0.55
ZONE_BOTTOM = 0.75
IOU_THRESHOLD = 0.3
BIKE_TIMEOUT  = 2.0   # seconds — if bike not seen for this long, remove its ID


def draw_detection_zone(frame):
    h, w = frame.shape[:2]
    y_top    = int(h * ZONE_TOP)
    y_bottom = int(h * ZONE_BOTTOM)
    color = (0, 255, 255)
    cv2.line(frame, (0, y_top),    (w, y_top),    color, 2)
    cv2.line(frame, (0, y_bottom), (w, y_bottom), color, 2)
    cv2.putText(frame, "-- ZONE START --", (10, y_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "-- ZONE END --",   (10, y_bottom + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def is_in_zone(bbox, frame_h):
    _, y1, _, y2 = bbox
    center_y = (y1 + y2) / 2
    return frame_h * ZONE_TOP <= center_y <= frame_h * ZONE_BOTTOM


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def cleanup_old_bikes(bike_tracker: dict):
    """Remove bikes not seen for BIKE_TIMEOUT seconds."""
    now  = time.time()
    gone = [bid for bid, info in bike_tracker.items()
            if (now - info['last_seen']) > BIKE_TIMEOUT]
    for bid in gone:
        del bike_tracker[bid]


def match_or_create_bike(bbox, bike_tracker: dict, next_bike_id: list):
    """
    Match bbox to existing tracked bike via IoU.
    Returns bike_id.
    """
    best_id  = None
    best_iou = 0.0

    for bid, info in bike_tracker.items():
        iou = compute_iou(bbox, info['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_id  = bid

    now = time.time()

    if best_iou >= IOU_THRESHOLD:
        # Same bike — update position and last_seen
        bike_tracker[best_id]['bbox']      = bbox
        bike_tracker[best_id]['last_seen'] = now
        return best_id
    else:
        # New bike — create fresh record
        new_id = next_bike_id[0]
        next_bike_id[0] += 1
        bike_tracker[new_id] = {
            'bbox':               bbox,
            'last_seen':          now,
            'saved':              False,   # violation already saved?
            'no_helmet_since':    None,    # timestamp when continuous no-helmet started
            # ── LP Cache: best license plate crop seen for this bike so far ──
            'best_plate_crop':    None,    # best LP image crop (numpy array)
            'best_plate_conf':    0.0,     # confidence of that crop
        }
        return new_id


def process_frame(
    frame: np.ndarray,
    bike_detector: BikePersonDetector,
    helmet_detector: HelmetDetector,
    violation_handler: ViolationHandler,
    frame_number: int,
    violation_cooldown: dict,        # kept for API compatibility, unused
    cooldown_frames: int = 30,
    bike_tracker: dict = None,
    next_bike_id: list = None,
    fps: float = 25.0,
) -> tuple:
    """
    Returns (annotated_frame, violation_detected_this_frame)
    """
    if bike_tracker is None:   bike_tracker  = {}
    if next_bike_id is None:   next_bike_id  = [1]

    annotated = frame.copy()
    violation_this_frame = False
    h, w = frame.shape[:2]

    draw_detection_zone(annotated)
    cleanup_old_bikes(bike_tracker)

    bike_detections = bike_detector.detect(frame)

    for det in bike_detections:
        label        = det['class']
        conf         = det['conf']
        x1, y1, x2, y2 = det['bbox']

        bike_id = match_or_create_bike([x1, y1, x2, y2], bike_tracker, next_bike_id)
        info    = bike_tracker[bike_id]
        in_zone = is_in_zone([x1, y1, x2, y2], h)

        if in_zone:
            pad = 10
            cx1 = max(0, x1-pad);  cy1 = max(0, y1-pad)
            cx2 = min(w, x2+pad);  cy2 = min(h, y2+pad)
            crop = frame[cy1:cy2, cx1:cx2]

            helmet_dets   = helmet_detector.detect(crop) if crop.size > 0 else []
            has_helmet    = any(d['class'] == 'helmet'    for d in helmet_dets)
            has_no_helmet = any(d['class'] == 'no_helmet' for d in helmet_dets)

            now = time.time()

            if has_helmet:
                # ── Helmet detected → reset no-helmet timer ──
                info['no_helmet_since'] = None
                is_violation = False

            elif has_no_helmet:
                # ── No helmet detected ──
                if info['no_helmet_since'] is None:
                    # Start the timer
                    info['no_helmet_since'] = now

                elapsed = now - info['no_helmet_since']
                is_violation = elapsed >= 1   # 1 sec continuous no-helmet
            else:
                # Nothing detected — don't reset timer, just wait
                is_violation = False

            # Draw bike box with ID
            draw_bbox(annotated, [x1, y1, x2, y2],
                      f"{label} #{ bike_id}", conf, violation=is_violation)

            # ── Draw helmet / no_helmet / license_plate boxes
            # ── AND update LP cache with best-confidence crop seen so far ──
            for hd in helmet_dets:
                hx1, hy1, hx2, hy2 = hd['bbox']
                abs_bbox = [cx1+hx1, cy1+hy1, cx1+hx2, cy1+hy2]
                draw_bbox(annotated, abs_bbox, hd['class'], hd['conf'])

                if hd['class'] == 'license_plate' and crop.size > 0:
                    lp_crop = crop[hy1:hy2, hx1:hx2]
                    # Only keep this crop if it's better than what we already have
                    if lp_crop.size > 0 and hd['conf'] > info['best_plate_conf']:
                        info['best_plate_conf'] = hd['conf']
                        info['best_plate_crop'] = lp_crop.copy()

            # ── Save once per bike, using the best LP seen across ALL frames ──
            if is_violation and not info['saved']:
                violation_handler.save_violation(
                    annotated,
                    info['best_plate_crop'],   # ← cached best LP, not just this frame's
                    frame_number
                )
                info['saved'] = True
                violation_this_frame = True

        else:
            # Outside zone — draw box only, no helmet logic
            draw_bbox(annotated, [x1, y1, x2, y2],
                      f"{label} #{bike_id}", conf, violation=False)
            # Don't reset no_helmet_since here — bike might briefly cross line

    return annotated, violation_this_frame