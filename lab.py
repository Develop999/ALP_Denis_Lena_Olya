import numpy as np
import cv2
import imutils
import time

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.7


def pedestrian_detection(img, model, layer_name, personid):
    H, W, _ = img.shape
    results = []
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (256, 256), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    boxes = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personid and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    if len(idzs):
        for i in idzs:
            x, y, w, h = boxes[i]
            res = (confidences[i], (x, y), (x + w, y + h))
            results.append(res)
    return results


cap = cv2.VideoCapture('olya.mp4')
classes_path = "coco.names"
classes = open(classes_path).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('captured.mp4', fourcc, 24, (1280, 720))
model = cv2.dnn.readNet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

while True:
    start = time.time()
    status, image = cap.read()
    if not status:
        break
    results = pedestrian_detection(image, model, layer_name, classes.index("person"))
    for res in results:
        cv2.rectangle(image, res[1], res[2], (0, 255, 0), 2)
    seconds = time.time() - start
    fps = 1 / seconds
    fps = round(fps, 2)
    cv2.putText(image, str(fps), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
    image = imutils.resize(image, width=1280)  # сохраняет пропорции в отличие от cv2.resize
    cv2.imshow("Detection", image)
    out.write(image)
    if cv2.waitKey(1) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
