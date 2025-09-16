# lbph/train_model.py
import os
import cv2
import numpy as np
import pickle
from utils import ensure_dirs, preprocess_face
from lbph import config

ensure_dirs()

dataset_dir = config.DATASET_DIR
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_ids = {}
current_id = 0
x_train = []
y_labels = []
counts = {}

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue
    files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    counts[person] = 0
    for file in files:
        path = os.path.join(person_dir, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Warning: cannot read", path); continue

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            roi = img
        else:
            faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x,y,w,h = faces_sorted[0]
            roi = img[y:y+h, x:x+w]

        try:
            roi_proc = preprocess_face(roi, size=config.FACE_SIZE)
        except Exception as e:
            print("Skipping", path, ":", e)
            continue

        if person not in label_ids:
            label_ids[person] = current_id
            current_id += 1
        id_ = label_ids[person]
        x_train.append(roi_proc)
        y_labels.append(id_)
        counts[person] += 1

if len(x_train) == 0:
    print("No training data found. Capture images first.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train, np.array(y_labels))

# Ensure trainer dir
os.makedirs(config.TRAINER_DIR, exist_ok=True)
model_path = os.path.join(config.TRAINER_DIR, "lbph_trainer.yml")
labels_path = os.path.join(config.TRAINER_DIR, "labels.pickle")
recognizer.write(model_path)
with open(labels_path, "wb") as f:
    pickle.dump(label_ids, f)

print("Training complete.")
print("Model saved to", model_path)
print("Labels saved to", labels_path)
print("Training summary:")
total_images = len(x_train)
print(f"  users: {len(label_ids)}")
print(f"  total images: {total_images}")
avg = total_images / len(label_ids) if len(label_ids) else 0
print(f"  avg per user: {avg:.1f}")
for k,v in counts.items():
    print(f"  {k}: {v}")