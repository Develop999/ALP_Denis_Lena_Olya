import numpy as np
import cv2
import time

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.5
zona = np.array([[200, 300, 500, 600]])


def check_intersection(body, rect):
    for (ax1, ay1, ax2, ay2) in body:
        break

    for (bx1, by1, bx2, by2) in rect:
        break

    s1 = (ax1 >= bx1 and ax1 <= bx2) or (ax2 >= bx1 and ax2 <= bx2)
    s2 = (ay1 >= by1 and ay1 <= by2) or (ay2 >= by1 and ay2 <= by2)
    s3 = (bx1 >= ax1 and bx1 <= ax2) or (bx2 >= ax1 and bx2 <= ax2)
    s4 = (by1 >= ay1 and by1 <= ay2) or (by2 >= ay1 and by2 <= ay2)

    if ((s1 and s2) or (s3 and s4)) or ((s1 and s4) or (s3 and s2)):
        return True
    else:
        return False


def pedestrian_detection(img, model, layer_name, personid):
    H, W, _ = img.shape
    results = []
    boxes = []
    confidences = []
    labels = []
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = detection[4] * scores[classID]
            if classID == personid and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                labels.append(classID)

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    if len(idzs):
        for i in idzs:
            x, y, w, h = boxes[i]
            res = (confidences[i], (x, y), (x + w, y + h), labels[i])
            results.append(res)
    return results


classes_path = "coco.names"
classes = open(classes_path).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('captured4.mp4', fourcc, 30, (1280, 720))

model = cv2.dnn.readNet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

cap = cv2.VideoCapture('olya.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)

while True:
    start = time.time()
    status, image = cap.read()
    if not status:
        break
    results = pedestrian_detection(image, model, layer_name, classes.index("person"))
    for (x1, y1, x2, y2) in zona:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for i, res in enumerate(results, start=1):
        if len(results):
            x1, y1 = res[1]
            x2, y2 = res[2]
            label = classes[res[3]]
            if check_intersection([res[1] + res[2]], zona):
                print(i, "???????????? ??????????:", res[1], " ???????????? ??????????:", res[2], "??????????????????")
                cv2.putText(image, '{}.{} {:.0%}'.format(i, label, res[0]), (x1 + 2, y1 - 5), font, 0.35, green, 1)
                cv2.rectangle(image, res[1], res[2], red, 2)
            else:
                print(i, "???????????? ??????????:", res[1], " ???????????? ??????????:", res[2])
                cv2.putText(image, '{}.{} {:.0%}'.format(i, label, res[0]), (x1 + 2, y1 - 5), font, 0.35, green, 1)
                cv2.rectangle(image, res[1], res[2], green, 2)
    print("==============================================================")
    seconds = time.time() - start
    fps = 1 / seconds
    fps = round(fps, 2)
    cv2.putText(image, str(fps), (5, 30), font, 1, (100, 255, 0), 2)
    cv2.imshow("Detection", image)
    out.write(image)
    if cv2.waitKey(1) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
