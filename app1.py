import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import os

# Load trained model
model = tf.keras.models.load_model("crime_detection_model.h5")

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    return prediction

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_prediction = 0
    crime_detected_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        prediction = predict_image(frame)
        total_prediction += prediction
        
        if prediction == 1:
            crime_detected_frames.append(frame_count)
    
    cap.release()
    
    avg_prediction = total_prediction / frame_count
    return avg_prediction, crime_detected_frames

# Streamlit UI
st.title("Crime Detection AI")

# Upload file
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if "image" in file_type:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = predict_image(image)
        
        if prediction > 0.5:
            st.error("🚨 Crime detected in the image!")
        else:
            st.success("✅ No crime detected in the image.")
    
    elif "video" in file_type:
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
        
        avg_prediction, crime_frames = process_video(temp_video_path)
        os.remove(temp_video_path)  # Clean up
        
        st.video(uploaded_file)
        st.write(f"🔹 Average Prediction Value: {avg_prediction:.2f}")
        
        if round(avg_prediction, 2) >= 0.97:
            st.error("🚨 Crime detected in the video!")
            st.write(f"Frames with detected crime: {crime_frames}")
        else:
            st.success("✅ No crime detected in the video.")
