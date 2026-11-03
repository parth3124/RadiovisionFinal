import AVFoundation
import cv2
import time

devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
    AVFoundation.AVMediaTypeVideo
)

logitech = None

for device in devices:
    if "C270" in device.localizedName():
        logitech = device
        break

if logitech is None:
    raise RuntimeError("Logitech C270 not found")

print("Logitech:", logitech.localizedName())
print("Unique ID:", logitech.uniqueID())

cap = cv2.VideoCapture(logitech.uniqueID(), cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not open Logitech")

print("Logitech opened. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if ret:
        print("Brightness:", round(frame.mean(), 2), end="\r")
        cv2.imshow("Logitech C270", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
