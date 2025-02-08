import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

IMG_HEIGHT = 112
IMG_WIDTH = 92

def load_dataset():
    """Load images from user_dataset/[username] into arrays."""
    X, y = [], []
    labels = {}
    label_id = 0

    for user_folder in os.listdir("user_dataset"):
        user_path = os.path.join("user_dataset", user_folder)
        if os.path.isdir(user_path):
            if user_folder not in labels:
                labels[user_folder] = label_id
                label_id += 1

            for image_file in os.listdir(user_path):
                img_path = os.path.join(user_path, image_file)
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
                img_array = img_to_array(img) / 255.0

                X.append(img_array)
                y.append(labels[user_folder])

    return np.array(X), np.array(y), labels

def build_model():
    """Define CNN model architecture."""
    model = Sequential([
        Conv2D(36, (7,7), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(2,2),
        Conv2D(54, (5,5), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(2024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(labels), activation='softmax')  
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading dataset...")
    X, y, labels = load_dataset()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    print("Saving model...")
    model.save("model/cnn_model.h5")
    
    print("Model retrained successfully!")
