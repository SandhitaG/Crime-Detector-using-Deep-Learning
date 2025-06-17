# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained crime detection model
model = load_model("crime_detection_model.h5")

# Function to process a video and detect crime
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Open video file
    frame_count = 0  # Track frame number
    predictions_list = []  # Store all prediction values
    
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Stop if the video ends
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame for efficiency
            continue
        
        # Resize frame to match model input size (128x128)
        resized_frame = cv2.resize(frame, (128, 128))
        
        # Convert frame to array and normalize pixel values
        img_array = img_to_array(resized_frame) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
        
        # Get prediction from the model
        prediction = model.predict(img_array)[0][0]
        predictions_list.append(prediction)  # Store prediction
        
        print(f"Frame {frame_count}: Prediction = {prediction:.2f}")  # Print prediction value
    
    cap.release()  # Release video resource

    # Calculate average prediction value
    if predictions_list:
        avg_prediction = sum(predictions_list) / len(predictions_list)
        print(f"\n🔹 Average Prediction Value: {avg_prediction:.2f}")
        
        # Updated decision: Crime detected if avg_prediction is close to 1
        if round(avg_prediction, 2) >= 0.97:  # Use a threshold to avoid float precision issues
            print("🚨 Crime detected in the video!")
        else:
            print("✅ No crime detected in the video.")

# Process a sample video file (Replace "video.mp4" with actual file path)
process_video("Images/normal1.mp4")
