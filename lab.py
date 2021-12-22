import numpy as np
import cv2

cap = cv2.VideoCapture('olya.mp4')
classes_path = "coco.names"
classes = open(classes_path).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

while True:
    status, image = cap.read()
    if not status:
        break
    H, W, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 256), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    boxes = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.8)
    if not status:
        break
    cv2.imshow("Detection", image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
