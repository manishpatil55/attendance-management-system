# lbph/capture_images.py
import os
import cv2
from utils import ensure_dirs, load_dnn_detector, detect_faces_dnn
from lbph import config

ensure_dirs()

def next_index(folder):
    existing = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    nums = []
    for f in existing:
        name = os.path.splitext(f)[0]
        try:
            nums.append(int(name))
        except:
            pass
    return max(nums) + 1 if nums else 1

def main():
    name = input("Enter full name (no underscores): ").strip()
    roll = input("Enter roll number (or id): ").strip()
    if not name or not roll:
        print("Invalid name/roll. Exiting.")
        return
    safe_name = name.replace(" ", "_")
    folder = os.path.join(config.DATASET_DIR, f"{safe_name}_{roll}")
    os.makedirs(folder, exist_ok=True)
    idx = next_index(folder)

    dnn_net = load_dnn_detector()
    use_dnn = (dnn_net is not None and config.FACE_DETECTOR == "dnn")
    if use_dnn:
        print("Using DNN face detector.")
    else:
        print("Using Haar Cascade detector.")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("SPACE to capture face, ESC to exit. Aim 30-100 images for good coverage.")
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        if use_dnn:
            faces = detect_faces_dnn(dnn_net, frame, conf_threshold=0.5)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x,y,w,h) in faces:
            cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(display, f"Saved: {saved}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture (SPACE to save, ESC to quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:  # SPACE
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue
            # choose largest
            faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x,y,w,h = faces_sorted[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = gray[y:y+h, x:x+w]
            out_path = os.path.join(folder, f"{idx}.jpg")
            cv2.imwrite(out_path, face)
            idx += 1
            saved += 1
            print("Saved", out_path)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} images to {folder}")

if __name__ == "__main__":
    main()