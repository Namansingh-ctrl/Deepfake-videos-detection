import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
from torchvision import models, transforms

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "models/deepfake_model.pth"
IMG_SIZE = 224
FRAMES_PER_VIDEO = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("Deepfake Detection System")
st.write("Upload a video to detect whether it is REAL or FAKE.")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():

    model = models.efficientnet_b0(pretrained=False)

    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    return model


model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -------------------------
# FACE DETECTOR
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# VIDEO PREDICTION
# -------------------------
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return "ERROR", 0

    frame_ids = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)

    predictions = []

    for fid in frame_ids:

        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            face = frame
        else:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(face)
            probs = torch.softmax(output, dim=1)

        predictions.append(probs.cpu().numpy()[0])

    cap.release()

    if len(predictions) == 0:
        return "NO FACE DETECTED", 0

    avg_pred = np.median(predictions, axis=0)

    pred_class = np.argmax(avg_pred)

    class_map = {
        0: "FAKE",
        1: "REAL"
    }

    label = class_map[pred_class]
    confidence = avg_pred[pred_class]

    return label, confidence


# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Analyze Video"):

        with st.spinner("Analyzing video..."):

            label, confidence = predict_video(temp_file.name)

        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")