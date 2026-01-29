import cv2
import math

# ---------------- MODELS ----------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# ---------------- LOAD NETWORKS ----------------
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# ---------------- LABELS ----------------
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ---------------- FACE DETECTION ----------------
def detectFace(net, frame, conf_threshold=0.4):
    frameCopy = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameCopy, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

    return frameCopy, faceBoxes

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultImg, faceBoxes = detectFace(faceNet, frame)

    if not faceBoxes:
        cv2.putText(resultImg, "No face detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    for box in faceBoxes:
        face = frame[max(0, box[1]):min(box[3], frame.shape[0]),
                     max(0, box[0]):min(box[2], frame.shape[1])]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            MODEL_MEAN_VALUES, swapRB=False)

        # Gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = GENDER_LIST[genderPred[0].argmax()]

        # Age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = AGE_LIST[agePred[0].argmax()]

        label = f"{gender}, {age}"
        cv2.putText(resultImg, label,
                    (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    cv2.imshow("Age & Gender Detection", resultImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
