import cv2

cam0 = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cam1 = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Camera 0 opened:", cam0.isOpened())
print("Camera 1 opened:", cam1.isOpened())

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if ret0:
        cv2.imshow("CAMERA INDEX 0", frame0)

    if ret1:
        cv2.imshow("CAMERA INDEX 1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam0.release()
cam1.release()
cv2.destroyAllWindows()