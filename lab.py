import numpy as np
import cv2

cap = cv2.VideoCapture('olya.mp4')
_, image = cap.read()
cv2.imshow("Detection", image)
cap.release()
cv2.destroyAllWindows()
