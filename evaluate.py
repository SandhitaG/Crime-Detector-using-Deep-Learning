import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="Crime Detection", layout="centered")
st.title("🛡️ Crime Detection System")

# Load model
model = load_model("crime_detection_model.h5")

IMG_SIZE = (128, 128)

# Prediction function
def predict_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return prediction

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    crime_detected = False

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 30:
            break
        prediction = predict_image(frame)
        if prediction >= 0.5:
            crime_detected = True
            break
        frame_count += 1
    cap.release()
    return crime_detected

# File uploader
file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

def display_result(prediction):
    if prediction >= 0.5:
        st.markdown("<h3 style='color:red;'>🔴 Crime Detected</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>🟢 No Crime Detected</h3>", unsafe_allow_html=True)

if file is not None:
    file_ext = file.name.split('.')[-1].lower()
    
    if file_ext in ['jpg', 'jpeg', 'png']:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        pred = predict_image(image)
        display_result(pred)

    elif file_ext == 'mp4':
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(file.read())
        result = process_video(tfile.name)
        st.video(tfile.name)
        if result:
            st.markdown("<h3 style='color:red;'>🔴 Crime Detected</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:green;'>🟢 No Crime Detected</h3>", unsafe_allow_html=True)
        os.unlink(tfile.name)
