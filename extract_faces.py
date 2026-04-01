import os
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 224
FRAMES_PER_VIDEO = 15

INPUT_REAL = "datasets/combined/Real"
INPUT_FAKE = "datasets/combined/Fake"

OUTPUT_REAL = "processed_faces/train/Real"
OUTPUT_FAKE = "processed_faces/train/Fake"

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_faces(video_path, save_dir):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return

    frame_ids = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)

    count = 0

    for fid in frame_ids:

        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        filename = os.path.basename(video_path) + f"_{count}.jpg"

        cv2.imwrite(os.path.join(save_dir, filename), face)

        count += 1

    cap.release()


print("Extracting REAL faces...")

for video in tqdm(os.listdir(INPUT_REAL)):
    extract_faces(os.path.join(INPUT_REAL, video), OUTPUT_REAL)

print("Extracting FAKE faces...")

for video in tqdm(os.listdir(INPUT_FAKE)):
    extract_faces(os.path.join(INPUT_FAKE, video), OUTPUT_FAKE)

print("Face extraction completed.")