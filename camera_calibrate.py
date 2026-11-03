
import cv2
import numpy as np
import json
import argparse

AREA_W = 85.0
AREA_H = 90.0

points = []
frame_display = None


def mouse_callback(event, x, y, flags, param):
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])

        print(f"Point {len(points)} selected: ({x}, {y})")

        cv2.circle(frame_display, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            frame_display,
            str(len(points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


def main():
    global frame_display

    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--out", default="camera_calibration.json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print("\nCAMERA CALIBRATION")
    print("------------------")
    print("Click the four corners of your taped experimental area.")
    print("Select them in this exact order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("\nPress R to reset points.")
    print("Press Q to quit.\n")

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame unavailable")
            break

        frame_display = frame.copy()

        for i, p in enumerate(points):
            cv2.circle(
                frame_display,
                tuple(p),
                7,
                (0, 0, 255),
                -1
            )
            cv2.putText(
                frame_display,
                str(i + 1),
                (p[0] + 10, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Calibration", frame_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            points.clear()
            print("Points reset")

        elif key == ord("q"):
            break

        if len(points) == 4:
            print("\nAll four corners selected.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Calibration cancelled: four points required.")
        return

    # Camera pixel coordinates.
    src = np.array(points, dtype=np.float32)

    # Physical floor coordinates in centimetres.
    dst = np.array([
        [0, 0],
        [AREA_W, 0],
        [AREA_W, AREA_H],
        [0, AREA_H]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    result = {
        "area_width_cm": AREA_W,
        "area_height_cm": AREA_H,
        "image_corners": points,
        "homography": H.tolist(),
        "coordinate_order": [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left"
        ]
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=4)

    print("\nCalibration saved:", args.out)
    print("Area:", AREA_W, "cm x", AREA_H, "cm")
    print("Homography:")
    print(H)


if __name__ == "__main__":
    main()
