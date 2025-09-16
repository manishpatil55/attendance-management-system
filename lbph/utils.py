# lbph/utils.py
import os
import cv2
import numpy as np
from lbph import config

def ensure_dirs():
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    os.makedirs(config.TRAINER_DIR, exist_ok=True)
    os.makedirs(config.ATTENDANCE_LOG_DIR, exist_ok=True)
    os.makedirs("models/face_detector", exist_ok=True)

def preprocess_face(gray_face, size=None):
    """
    Normalize face for LBPH:
    - Resize, histogram equalization
    - Return uint8 numpy array
    """
    if size is None:
        size = config.FACE_SIZE
    face_resized = cv2.resize(gray_face, size)
    face_eq = cv2.equalizeHist(face_resized)
    return np.array(face_eq, dtype="uint8")

def load_dnn_detector():
    """
    Loads OpenCV DNN face detector if files exist, else returns None.
    """
    proto = config.DNN_PROTO
    model = config.DNN_MODEL
    if os.path.exists(proto) and os.path.exists(model):
        net = cv2.dnn.readNetFromCaffe(proto, model)
        return net
    return None

def detect_faces_dnn(net, frame, conf_threshold=0.5):
    """
    Return list of (x,y,w,h) in *frame* using DNN SSD detector.
    frame expected as BGR image.
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # clamp
            startX = max(0, startX); startY = max(0, startY)
            endX = min(w - 1, endX); endY = min(h - 1, endY)
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces