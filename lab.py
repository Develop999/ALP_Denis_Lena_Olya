import numpy as np
import cv2

cap = cv2.VideoCapture('olya.mp4')
while True:
    status, image = cap.read()
    if status:
        cv2.imshow("Detection", image)
cap.release()
cv2.destroyAllWindows()
