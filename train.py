import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW

# Enable XLA Compiler (Boosts CPU Performance)
tf.config.optimizer.set_jit(True)

# Define dataset path
train_dir = "UCF_Crime_Dataset/Train"

# Image Data Generator (with smaller image size)
train_datagen = ImageDataGenerator(
    rescale=1./255,       
    rotation_range=15,    
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.1,  
    zoom_range=0.1,  
    horizontal_flip=True,
    validation_split=0.2  
)

# Load training images (80%)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # ⬇️ Reduced size for faster training
    batch_size=64,  # ⬆️ Increased batch size (if RAM allows)
    class_mode='binary',
    subset='training'  
)

# Load validation images (20%)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,  
    class_mode='binary',
    subset='validation'
)

# Define the CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),  # Prevent overfitting
        layers.Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer=AdamW(learning_rate=0.001),  # 🚀 Faster optimizer
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Build the model
model = build_model()
model.summary()

# Train the model with optimized settings
print("🚀 Training started...")
history = model.fit(
    train_generator,         
    validation_data=val_generator,  
    epochs=10 
)



# Save the trained model
model.save("crime_detection_model1.h5")

print("✅ Model training complete! Saved as 'crime_detection_model.h5'!")
