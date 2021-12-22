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
    H, W, chanels = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 256), swapRB=True)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    boxes = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            b1 = detection[0] * W
            b2 = detection[1] * H
            b3 = detection[2] * W
            b4 = detection[3] * H
            x = int(b1 - (b3 / 2))
            y = int(b2 - (b4 / 2))
            boxes.append([x, y, b3, b4])
            confidences.append(float(confidence))

    if not status:
        break
    cv2.imshow("Detection", image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
