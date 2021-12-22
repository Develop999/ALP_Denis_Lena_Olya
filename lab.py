import numpy as np
import cv2

def pedestrian_detection(img, model, layer_name, personid):
    H, W, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    results = []
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
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.6)
    if len(idzs):
        for i in idzs:
            x, y, w, h = boxes[i]
            res = (confidences[i], (x, y), (x + w, y + h))
            results.append(res)
    return results


cap = cv2.VideoCapture(0)
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
    results = pedestrian_detection(image, model, layer_name, classes.index("person"))
    for res in results:
        cv2.rectangle(image, res[1], res[2], (0, 255, 0), 2)
    cv2.imshow("Detection", image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
