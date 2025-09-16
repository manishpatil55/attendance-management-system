# lbph/recognize_and_attend.py
import cv2
import pickle
import os
from lbph.utils import ensure_dirs, load_dnn_detector, detect_faces_dnn, preprocess_face
from lbph.attendance_logger import mark_attendance
from lbph import config

ensure_dirs()

# load model & labels
model_path = os.path.join(config.TRAINER_DIR, "lbph_trainer.yml")
labels_path = os.path.join(config.TRAINER_DIR, "labels.pickle")
if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print("Model/labels not found. Run lbph/train_model.py first.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

with open(labels_path, "rb") as f:
    labels = pickle.load(f)
inv_labels = {v:k for k,v in labels.items()}

# detector
dnn_net = load_dnn_detector()
use_dnn = (dnn_net is not None and config.FACE_DETECTOR == "dnn")
if use_dnn:
    print("Using DNN detector")
else:
    print("Using Haar Cascade detector")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

marked_today = set()  # additional in-session guard (attendance_logger also guards duplicates)

print("Starting. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()
    if use_dnn:
        faces = detect_faces_dnn(dnn_net, frame, conf_threshold=0.5)
    else:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        roi_proc = preprocess_face(roi)
        try:
            id_, conf = recognizer.predict(roi_proc)
        except Exception:
            continue
        name_id = "Unknown"
        if conf < config.LBPH_CONF_THRESHOLD:
            name_id = inv_labels.get(id_, "Unknown")
            # expect name stored as Name_Roll
            if "_" in name_id:
                name, roll = name_id.rsplit("_", 1)
            else:
                name = name_id; roll = ""
            if name not in marked_today:
                ok = mark_attendance(name, roll)
                if ok:
                    marked_today.add(name)
                    print(f"Marked {name} ({roll}) conf={conf:.1f}")
        else:
            name = "Unknown"

        label = f"{name} ({int(conf)})"
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(display, (x,y), (x+w, y+h), color, 2)
        cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance (ESC to quit)", display)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")