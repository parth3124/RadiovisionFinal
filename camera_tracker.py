import cv2
import numpy as np
import json
import csv
import time

CAMERA_ID = 0

# Load camera calibration
with open("camera_calibration.json", "r") as f:
    calibration = json.load(f)

H = np.array(calibration["homography"], dtype=np.float32)

AREA_W = float(calibration["area_width_cm"])
AREA_H = float(calibration["area_height_cm"])

# Open Logitech camera
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# Green sticker HSV range
lower_green = np.array([85, 100, 70])
upper_green = np.array([110, 255, 255])

# Save tracking data
csv_file = open("trajectory.csv", "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "monotonic_ns",
    "detected",
    "pixel_x",
    "pixel_y",
    "x_cm",
    "y_cm"
])

print("Tracking started. Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = 0
    pixel_x = pixel_y = x_cm = y_cm = ""

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if 4 < area < 5000:
            M = cv2.moments(largest)

            if M["m00"] != 0:
                pixel_x = M["m10"] / M["m00"]
                pixel_y = M["m01"] / M["m00"]

                point = np.array(
                    [[[pixel_x, pixel_y]]],
                    dtype=np.float32
                )

                floor_point = cv2.perspectiveTransform(point, H)

                x_cm = float(floor_point[0][0][0])
                y_cm = float(floor_point[0][0][1])

                detected = 1

                cv2.circle(
                    frame,
                    (int(pixel_x), int(pixel_y)),
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"X={x_cm:.1f} cm Y={y_cm:.1f} cm",
                    (int(pixel_x) + 10, int(pixel_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    timestamp = time.monotonic_ns()

    writer.writerow([
        timestamp,
        detected,
        pixel_x,
        pixel_y,
        x_cm,
        y_cm
    ])

    cv2.imshow("Green Sticker Tracking", frame)
    cv2.imshow("Green Detection Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")
print("Saved data to trajectory.csv")