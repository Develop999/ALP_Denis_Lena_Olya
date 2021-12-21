import numpy as np
import cv2

cap = cv2.VideoCapture('olya.mp4')
classes_path = "coco.names"
classes = open(classes_path).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNet(config_path, weights_path)
layer_name = model.getLayerNames()
print(layer_name)
layer_name = [layer_name[i-1] for i in model.getUnconnectedOutLayers()]
print(layer_name)

while True:
    status, image = cap.read()
    H, W, chanels = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 256), swapRB=True)
    print(blob.shape)
    if not status:
        break
    cv2.imshow("Detection", image)
    if cv2.waitKey(0):
        break

cap.release()
cv2.destroyAllWindows()
