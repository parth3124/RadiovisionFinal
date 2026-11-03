import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Camera could not open")

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param
        patch = hsv[max(0,y-5):y+6, max(0,x-5):x+6]
        pixels = patch.reshape(-1, 3)
        print("Sticker sample HSV (median):",
              np.median(pixels, axis=0).astype(int))

cv2.namedWindow("Click the sticker")
cv2.setMouseCallback("Click the sticker", click)

print("Click directly on the green sticker. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Click the sticker", frame)
    cv2.setMouseCallback("Click the sticker", click, hsv)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
