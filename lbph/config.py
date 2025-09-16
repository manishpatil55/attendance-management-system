# lbph/config.py

# LBPH threshold: lower conf is better. Tune between 40..90 depending on dataset quality
LBPH_CONF_THRESHOLD = 75

# Face detector choice: "dnn" or "haar"
FACE_DETECTOR = "dnn"

# DNN face detector model files (place them in models/face_detector/)
DNN_PROTO = "models/face_detector/deploy.prototxt"
DNN_MODEL = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

# Image face size for training/prediction
FACE_SIZE = (200, 200)

# Attendance log directory
ATTENDANCE_LOG_DIR = "attendance_logs"
TRAINER_DIR = "trainer"
DATASET_DIR = "dataset"