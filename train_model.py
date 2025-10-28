import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

def create_eye_model():
    """Create CNN model for eye state classification"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def load_dataset(data_dir):
    """Load and preprocess the eye dataset"""
    images = []
    labels = []
    
    # Open eyes
    open_dir = os.path.join(data_dir, 'open')
    for img_file in os.listdir(open_dir):
        img_path = os.path.join(open_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (24, 24))
        images.append(img)
        labels.append(0)  # 0 for open eyes
    
    # Closed eyes
    closed_dir = os.path.join(data_dir, 'closed')
    for img_file in os.listdir(closed_dir):
        img_path = os.path.join(closed_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (24, 24))
        images.append(img)
        labels.append(1)  # 1 for closed eyes
    
    # Convert to numpy arrays
    images = np.array(images).reshape(-1, 24, 24, 1)
    labels = np.array(labels)
    
    # Normalize images
    images = images.astype('float32') / 255.0
    
    return images, labels

def train_model():
    """Train the eye state classification model"""
    print("Loading dataset...")
    X, y = load_dataset('dataset/')
    
    print("Creating model...")
    model = create_eye_model()
    
    print("Training model...")
    history = model.fit(X, y, 
                       epochs=50,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=1)
    
    # Save model
    model.save('model/eye_model.h5')
    print("Model saved successfully!")
    
    return model, history

if __name__ == "__main__":
    train_model()