#!/usr/bin/env python3
"""
Robot calibration for a fixed-orientation tracked robot.

This replaces the original human/cap calibration procedure. The four
calibration coordinates refer to the CENTRE OF THE ROBOT'S FLOOR FOOTPRINT,
not to the coloured marker itself.

Important:
- Keep the robot facing exactly the same direction at all four points.
- Keep the coloured marker fixed on the robot.
- Put the ROBOT FOOTPRINT CENTRE over each measured floor coordinate.
- The marker must remain visible to the camera.
- The four calibration positions are inset from the area boundary so the
  robot does not need to hang over the edge.

Example:
    python3 overhead_calibrate_robot.py --cam 0 --area-w 1.00 --area-d 1.28 \
        --margin-x 0.20 --margin-y 0.20 --out calib_robot.json

Keys:
    In colour phase: click the blue/green marker.
    In confirm phase: y accepts the colour; click again to select another patch.
    In marks phase: move robot centre to the displayed coordinate, keep its
                    heading fixed, then press SPACE.
    In verify phase: put robot centre at the exact area centre and press c
                     to check; s saves; r repeats the four points; q quits.
"""

import argparse
import json
import time

import cv2
import numpy as np

import common as C
import overhead_common as OC


def average_detection(cap, lo, hi, n=40, min_valid=25, min_area=60):
    """Return median detected marker centre over up to n valid frames."""
    points = []
    attempts = 0
    max_attempts = n * 5

    while len(points) < n and attempts < max_attempts:
        ok, frame = cap.read()
        attempts += 1
        if not ok:
            continue

        detection, _ = OC.detect_marker(frame, lo, hi, min_area)
        if detection is not None:
            points.append(detection[:2])

    if len(points) >= min_valid:
        return np.median(np.asarray(points, dtype=np.float32), axis=0), len(points)

    return None, len(points)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Logitech camera to tracked-robot centre coordinates."
    )
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--area-w", type=float, required=True,
                         help="Area width in metres (X direction).")
    parser.add_argument("--area-d", type=float, required=True,
                         help="Area depth in metres (Y direction).")
    parser.add_argument("--margin-x", type=float, default=0.20,
                         help="Inset from left/right boundaries for robot-centre calibration points.")
    parser.add_argument("--margin-y", type=float, default=0.20,
                         help="Inset from top/bottom boundaries for robot-centre calibration points.")
    parser.add_argument("--zones-cols", type=int, default=3)
    parser.add_argument("--zones-rows", type=int, default=1)
    parser.add_argument("--out", default="calib_robot.json")
    parser.add_argument("--min-area", type=int, default=60)
    args = parser.parse_args()

    width = args.area_w
    depth = args.area_d

    if width <= 0 or depth <= 0:
        parser.error("--area-w and --area-d must be positive.")
    if args.margin_x <= 0 or args.margin_y <= 0:
        parser.error("--margin-x and --margin-y must be positive.")
    if 2 * args.margin_x >= width or 2 * args.margin_y >= depth:
        parser.error("Margins are too large for the specified area.")

    # Robot-centre coordinates, inset so the robot remains inside the area.
    # Keep the robot heading/orientation identical at all four positions.
    points_world = [
        (args.margin_x, args.margin_y),
        (width - args.margin_x, args.margin_y),
        (width - args.margin_x, depth - args.margin_y),
        (args.margin_x, depth - args.margin_y),
    ]

    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"Camera index {args.cam} could not provide a frame.")

    state = {"click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    window = "robot_calibrate"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    phase = "colour"
    hsv_low = None
    hsv_high = None
    pixel_points = []
    homography = None
    point_index = 0
    message = "Click the blue/green marker on top of the robot."

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            view = frame.copy()

            if phase == "colour":
                C.put_text(cv2, view,
                           "1) Click the coloured marker on the robot",
                           (10, 24), 0.55)
                if state["click"] is not None:
                    x, y = state["click"]
                    state["click"] = None
                    patch = cv2.cvtColor(
                        frame[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5],
                        cv2.COLOR_BGR2HSV
                    )
                    hsv_low, hsv_high = OC.hsv_range_from_patch(patch)
                    phase = "confirm"
                    message = ""

            elif phase == "confirm":
                detection, mask = OC.detect_marker(
                    frame, hsv_low, hsv_high, args.min_area
                )
                view[mask > 0] = (
                    0.5 * view[mask > 0]
                    + 0.5 * np.array([0, 255, 0])
                ).astype(np.uint8)

                C.put_text(cv2, view,
                           "Check green overlay on robot marker. y=accept; click=choose again",
                           (10, 24), 0.48)

                if detection is not None:
                    cv2.circle(view, (int(detection[0]), int(detection[1])),
                               8, (0, 0, 255), 2)

                if state["click"] is not None:
                    x, y = state["click"]
                    state["click"] = None
                    patch = cv2.cvtColor(
                        frame[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5],
                        cv2.COLOR_BGR2HSV
                    )
                    hsv_low, hsv_high = OC.hsv_range_from_patch(patch)

            elif phase == "marks":
                x_world, y_world = points_world[point_index]
                instruction = (
                    f"Point {point_index + 1}/4: robot CENTRE at "
                    f"({x_world:.2f}, {y_world:.2f}) m; same heading; SPACE"
                )
                C.put_text(cv2, view, instruction, (10, 24), 0.48)

                detection, _ = OC.detect_marker(
                    frame, hsv_low, hsv_high, args.min_area
                )
                if detection is not None:
                    cv2.circle(view, (int(detection[0]), int(detection[1])),
                               9, (0, 0, 255), 2)

                for point in pixel_points:
                    cv2.circle(view, (int(point[0]), int(point[1])),
                               5, (255, 0, 0), -1)

                C.put_text(cv2, view,
                           "Move robot so its footprint centre is over the measured floor point.",
                           (10, view.shape[0] - 14), 0.42, (0, 220, 255))

            elif phase == "verify":
                C.put_text(cv2, view,
                           "Verification: move robot centre to area centre; c=check, s=save, r=redo",
                           (10, 24), 0.45)

                OC.draw_area(view, {
                    "H": homography,
                    "area": (width, depth),
                    "zones": (args.zones_cols, args.zones_rows)
                })

                detection, _ = OC.detect_marker(
                    frame, hsv_low, hsv_high, args.min_area
                )
                if detection is not None:
                    x_est, y_est = OC.pix_to_world(
                        homography, detection[0], detection[1]
                    )
                    zone = OC.zone_index(
                        x_est, y_est, (width, depth),
                        (args.zones_cols, args.zones_rows)
                    )
                    cv2.circle(view, (int(detection[0]), int(detection[1])),
                               9, (0, 0, 255), 2)
                    C.put_text(
                        cv2, view,
                        f"Robot centre: x={x_est:.2f} m, y={y_est:.2f} m; "
                        f"zone {zone + 1 if zone >= 0 else '-'}",
                        (10, 52), 0.55, (0, 255, 0)
                    )

            if message:
                C.put_text(cv2, view, message,
                           (10, view.shape[0] - 34), 0.42, (0, 200, 255))

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if phase == "confirm" and key == ord("y"):
                phase = "marks"
                message = (
                    "Calibration uses robot-centre coordinates. "
                    "Keep robot heading identical at every point."
                )

            elif phase == "marks" and key == ord(" "):
                measured, count = average_detection(
                    cap, hsv_low, hsv_high,
                    n=40, min_valid=25, min_area=args.min_area
                )

                if measured is None:
                    message = (
                        f"Only {count} valid marker frames. "
                        "Check marker visibility/colour and try SPACE again."
                    )
                else:
                    pixel_points.append(measured.tolist())
                    message = f"Recorded point {point_index + 1}: {count} frames."
                    point_index += 1

                    if point_index == 4:
                        homography = cv2.getPerspectiveTransform(
                            np.asarray(pixel_points, dtype=np.float32),
                            np.asarray(points_world, dtype=np.float32)
                        ).astype(np.float64)
                        phase = "verify"
                        message = (
                            "Four points recorded. Check the centre before saving."
                        )

            elif phase == "verify":
                if key == ord("r"):
                    phase = "marks"
                    pixel_points = []
                    point_index = 0
                    homography = None
                    message = "Redoing calibration points."
                elif key == ord("c"):
                    measured, count = average_detection(
                        cap, hsv_low, hsv_high,
                        n=40, min_valid=25, min_area=args.min_area
                    )
                    if measured is None:
                        message = f"Centre check failed: only {count} valid frames."
                    else:
                        x_est, y_est = OC.pix_to_world(
                            homography, measured[0], measured[1]
                        )
                        error = np.hypot(
                            x_est - width / 2,
                            y_est - depth / 2
                        )
                        message = (
                            f"Centre check: ({x_est:.2f}, {y_est:.2f}) m; "
                            f"expected ({width/2:.2f}, {depth/2:.2f}); "
                            f"error {error*100:.1f} cm"
                        )
                        print(message)

                elif key == ord("s"):
                    ok, final_frame = cap.read()
                    if not ok:
                        message = "Could not read camera frame; not saved."
                        continue

                    calibration = {
                        "cam": args.cam,
                        "frame": [
                            int(final_frame.shape[1]),
                            int(final_frame.shape[0])
                        ],
                        "area": [width, depth],
                        "zones": [args.zones_cols, args.zones_rows],
                        "pix_pts": pixel_points,
                        "world_pts": [list(point) for point in points_world],
                        "H": homography.tolist(),
                        "hsv_lo": np.asarray(hsv_low).astype(int).tolist(),
                        "hsv_hi": np.asarray(hsv_high).astype(int).tolist(),
                        "min_area_px": args.min_area,
                        "position_reference": "robot_footprint_centre",
                        "robot_heading_must_remain_fixed": True,
                        "created": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }

                    with open(args.out, "w") as output_file:
                        json.dump(calibration, output_file, indent=2)

                    print(f"Saved robot calibration to {args.out}")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
