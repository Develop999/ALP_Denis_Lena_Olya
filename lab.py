import numpy as np
import cv2

cap = cv2.VideoCapture('olya.mp4')
while True:
    status, image = cap.read()
    if not status:
        break
    cv2.imshow("Detection", image)
    if cv2.waitKey(0):
        break

cap.release()
cv2.destroyAllWindows()
