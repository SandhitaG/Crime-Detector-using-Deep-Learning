import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("crime_detection_model.h5")

# Load and preprocess the image
img_path = "Images/crime/2.jpeg"  # Replace with the actual image path
img = image.load_img(img_path, target_size=(128,128))  # Resize to match training size
img_array = image.img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch input
img_array = img_array / 255.0  # Normalize

# Make a prediction
prediction = model.predict(img_array)

# Interpret the result
class_names = ["Normal", "Crime"]  # Adjust based on how you labeled your dataset
predicted_class = class_names[int(prediction[0] > 0.5)]

# Display the image and prediction result
plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {predicted_class}")
plt.show()
