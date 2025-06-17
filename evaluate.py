import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = load_model("crime_detection_model.h5")

# Load test dataset paths
test_crime_dir = "UCF_Crime_Dataset/Test/Crime"  # Update path if needed
test_normal_dir = "UCF_Crime_Dataset/Test/Normal"  # Update path if needed

# Function to load images and labels
def load_data(directory, label):
    data, labels = [], []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64)) / 255.0  # Resize and normalize
        data.append(img_to_array(img))
        labels.append(label)
    return np.array(data), np.array(labels)

# Load crime and normal images
crime_images, crime_labels = load_data(test_crime_dir, 1)  # 1 for Crime
normal_images, normal_labels = load_data(test_normal_dir, 0)  # 0 for Normal

# Combine data
X_test = np.vstack((crime_images, normal_images))
y_test = np.concatenate((crime_labels, normal_labels))

# Predict using the model
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert probabilities to binary values

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

# Print metrics
print(f"🔹 Accuracy: {accuracy:.2f}%")
print(f"🔹 Precision: {precision:.2f}%")
print(f"🔹 Recall: {recall:.2f}%")
print(f"🔹 F1 Score: {f1:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()  # Extract TP, FP, FN, TN

print(f"\n🔹 Confusion Matrix Values:")
print(f"✅ True Positives (TP): {tp}")
print(f"❌ False Positives (FP): {fp}")
print(f"❌ False Negatives (FN): {fn}")
print(f"✅ True Negatives (TN): {tn}")

# Plot confusion matrix as heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Crime"], yticklabels=["Normal", "Crime"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Class distribution plot for training dataset
train_crime_count = len(os.listdir("UCF_Crime_Dataset/Train/Crime"))
train_normal_count = len(os.listdir("UCF_Crime_Dataset/Train/Normal"))

plt.figure(figsize=(6, 5))
plt.bar(["Normal", "Crime"], [train_normal_count, train_crime_count], color=["green", "red"])
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Training Dataset Class Distribution")
plt.show()
