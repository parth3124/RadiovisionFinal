"""overhead_common.py - shared helpers for the overhead-camera ground-truth tools (marker detection, floor coordinates, zones)."""
import json
import numpy as np
import cv2


def load_calib(path):
    d = json.load(open(path))
    d["H"] = np.array(d["H"], np.float64)
    return d


def hsv_range_from_patch(hsv_patch):
    """median HSV of a clicked patch -> (lo, hi) thresholds. Avoid red caps (hue wraps around 0)."""
    h, s, v = [int(np.median(hsv_patch[..., i])) for i in range(3)]
    lo = [max(h - 12, 0), max(s - 70, 40), max(v - 70, 40)]
    hi = [min(h + 12, 179), 255, 255]
    return lo, hi


def detect_marker(frame_bgr, hsv_lo, hsv_hi, min_area=10, prev_xy=None, max_jump=100):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([max(int(hsv_lo[0]) - 5, 0), 80, 40], dtype=np.uint8)
    hi = np.array([min(int(hsv_hi[0]) + 5, 179), 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # If we have a previous position, penalize or filter out contours too far away
        if prev_xy is not None:
            dist = np.hypot(cx - prev_xy[0], cy - prev_xy[1])
            if dist > max_jump:
                continue
        valid.append((cx, cy, area))

    if not valid:
        return None, mask
    
    # Return the best valid contour (largest area among close ones, or closest if preferred)
    best = max(valid, key=lambda p: p[2])
    return best, mask
def pix_to_world(H, px, py):
    v = H @ np.array([px, py, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def zone_index(x, y, area, zones, tol=0.05):
    """0-based zone of floor point (x, y) in metres; -1 if outside the calibrated rectangle (tolerance in metres)."""
    W, D = area
    cols, rows = zones
    if not (np.isfinite(x) and np.isfinite(y)) or x < -tol or y < -tol or x > W + tol or y > D + tol:
        return -1
    c = min(max(int(x / W * cols), 0), cols - 1)
    r = min(max(int(y / D * rows), 0), rows - 1)
    return r * cols + c


def draw_area(img, calib):
    """draw the calibrated rectangle and zone borders on a camera frame (using the inverse homography)"""
    Hinv = np.linalg.inv(calib["H"])
    W, D = calib["area"]; cols, rows = calib["zones"]

    def p(x, y):
        v = Hinv @ np.array([x, y, 1.0]); return int(v[0] / v[2]), int(v[1] / v[2])
    for c in range(cols + 1):
        cv2.line(img, p(W * c / cols, 0), p(W * c / cols, D), (0, 255, 255), 1)
    for r in range(rows + 1):
        cv2.line(img, p(0, D * r / rows), p(W, D * r / rows), (0, 255, 255), 1)
    for r in range(rows):
        for c in range(cols):
            cx, cy = p(W * (c + 0.5) / cols, D * (r + 0.5) / rows)
            cv2.putText(img, f"Z{r * cols + c + 1}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
