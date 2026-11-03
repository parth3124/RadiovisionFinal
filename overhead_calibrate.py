#!/usr/bin/env python3
"""
overhead_calibrate.py - one-off calibration of the OVERHEAD camera: marker colour + camera pixels -> floor coordinates (metres).

    python overhead_calibrate.py --cam 0 --area-w 2.4 --area-d 1.8 --zones-cols 3 --zones-rows 1 --out calib.json

Why you stand on the marks: the cap is ~1.7 m above the floor, so from a camera 2.7 m up it appears ~2.7x farther from the image
centre than your feet. Calibrating with YOU wearing the cap while standing on the floor marks learns exactly that mapping.

Floor marks (tape on the floor, measure them): the four corners of the area you want to track, in this order, with x along the
TX->RX direction:    mark 1 = (0, 0)   mark 2 = (W, 0)   mark 3 = (W, D)   mark 4 = (0, D)   [metres].
Optional: a 5th tape mark at the centre (W/2, D/2) to verify the calibration.

Keys:  click = pick the cap colour   y = accept colour   SPACE = record the current mark (stand still on it)
       c = check on the centre mark     s = save and quit     r = redo marks     q = quit without saving
"""
import argparse, json, time
import numpy as np
import cv2
import common as C
import overhead_common as OC


def average_detection(cap, lo, hi, n=40, min_valid=25):
    pts = []
    for _ in range(n * 3):
        ok, f = cap.read()
        if not ok:
            continue
        d, _ = OC.detect_marker(f, lo, hi)
        if d:
            pts.append(d[:2])
        if len(pts) >= n:
            break
    return (np.median(np.array(pts), 0) if len(pts) >= min_valid else None), len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--area-w", type=float, required=True, help="metres, x direction (TX -> RX)")
    ap.add_argument("--area-d", type=float, required=True, help="metres, y direction")
    ap.add_argument("--zones-cols", type=int, default=3)
    ap.add_argument("--zones-rows", type=int, default=1)
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--min-area", type=int, default=60, help="minimum marker blob size in pixels")
    a = ap.parse_args()
    W, D = a.area_w, a.area_d
    marks = [(0, 0), (W, 0), (W, D), (0, D)]
    cap = cv2.VideoCapture(a.cam); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.read()[0]:
        raise SystemExit(f"camera {a.cam} not readable (python collect.py --list-cams shows the numbers)")
    st = {"click": None}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["click"] = (x, y)
    cv2.namedWindow("calibrate"); cv2.setMouseCallback("calibrate", on_mouse)
    phase, lo, hi, pix, H, k = "color", None, None, [], None, 0
    msg = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = frame.copy()
        if phase == "color":
            C.put_text(cv2, view, "1) click on the coloured cap (wear it, stand in view)", (10, 24), 0.55)
            if st["click"]:
                x, y = st["click"]; st["click"] = None
                patch = cv2.cvtColor(frame[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5], cv2.COLOR_BGR2HSV)
                lo, hi = OC.hsv_range_from_patch(patch); phase = "confirm"
                if lo[0] < 12 or hi[0] > 167:
                    msg = "warning: red-ish colours wrap around in HSV; a green/blue/orange cap works better"
        elif phase == "confirm":
            d, mask = OC.detect_marker(frame, lo, hi, a.min_area)
            view[mask > 0] = (0.5 * view[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            C.put_text(cv2, view, "colour found (green overlay). y = accept, click again = re-pick", (10, 24), 0.55)
            if d:
                cv2.circle(view, (int(d[0]), int(d[1])), 8, (0, 0, 255), 2)
            if st["click"]:
                x, y = st["click"]; st["click"] = None
                patch = cv2.cvtColor(frame[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5], cv2.COLOR_BGR2HSV)
                lo, hi = OC.hsv_range_from_patch(patch)
        elif phase == "marks":
            mx, my = marks[k]
            C.put_text(cv2, view, f"2) stand on floor mark {k + 1} = ({mx:.2f}, {my:.2f}) m, keep still, press SPACE", (10, 24), 0.55)
            d, _ = OC.detect_marker(frame, lo, hi, a.min_area)
            if d:
                cv2.circle(view, (int(d[0]), int(d[1])), 8, (0, 0, 255), 2)
            for q in pix:
                cv2.circle(view, (int(q[0]), int(q[1])), 5, (255, 0, 0), -1)
        elif phase == "verify":
            C.put_text(cv2, view, "3) walk around: coordinates should match reality. c = check centre mark, s = save, r = redo", (10, 24), 0.5)
            OC.draw_area(view, {"H": H, "area": (W, D), "zones": (a.zones_cols, a.zones_rows)})
            d, _ = OC.detect_marker(frame, lo, hi, a.min_area)
            if d:
                x, y = OC.pix_to_world(H, d[0], d[1])
                z = OC.zone_index(x, y, (W, D), (a.zones_cols, a.zones_rows))
                cv2.circle(view, (int(d[0]), int(d[1])), 8, (0, 0, 255), 2)
                C.put_text(cv2, view, f"x={x:5.2f} y={y:5.2f} m   zone {z + 1 if z >= 0 else '-'}", (10, 50), 0.6, (0, 255, 0))
        if msg:
            C.put_text(cv2, view, msg, (10, view.shape[0] - 12), 0.45, (0, 200, 255))
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if phase == "confirm" and key == ord("y"):
            phase, msg = "marks", ""
        elif phase == "marks" and key == ord(" "):
            p, n = average_detection(cap, lo, hi)
            if p is None:
                msg = f"marker seen in only {n} frames - stand in view, keep still, check the colour"
            else:
                pix.append(p); k += 1; msg = f"mark {k} recorded ({n} frames)"
                if k == len(marks):
                    H = cv2.getPerspectiveTransform(np.array(pix, np.float32), np.array(marks, np.float32)).astype(np.float64)
                    phase, msg = "verify", ""
        elif phase == "verify":
            if key == ord("r"):
                phase, pix, k, msg = "marks", [], 0, ""
            elif key == ord("c"):
                p, n = average_detection(cap, lo, hi)
                if p is None:
                    msg = "no marker seen"
                else:
                    x, y = OC.pix_to_world(H, p[0], p[1])
                    err = np.hypot(x - W / 2, y - D / 2)
                    msg = f"centre check: measured ({x:.2f}, {y:.2f}) vs ({W / 2:.2f}, {D / 2:.2f}) -> error {100 * err:.0f} cm"
                    print(msg)
            elif key == ord("s"):
                ok, f = cap.read()
                json.dump({"cam": a.cam, "frame": [int(f.shape[1]), int(f.shape[0])], "area": [W, D],
                           "zones": [a.zones_cols, a.zones_rows], "pix_pts": [list(map(float, q)) for q in pix],
                           "world_pts": marks, "H": H.tolist(), "hsv_lo": lo, "hsv_hi": hi, "min_area_px": a.min_area,
                           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(a.out, "w"), indent=2)
                print(f"saved {a.out}")
                break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
