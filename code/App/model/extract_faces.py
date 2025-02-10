from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
import numpy as np
import os
import cv2
import pickle

# =============== CHOOSE THE MODEL =============== #
USE_MODEL = "FaceNet"  # Change to "InceptionV3" if needed
# ================================================ #

# ✅ Load Pretrained FaceNet Model (Download facenet_keras.h5 from: https://github.com/nyoki-mtl/keras-facenet)
if USE_MODEL == "FaceNet":
    facenet_model = load_model("facenet_keras.h5", compile=False)  # ✅ Safe loading
    IMG_SIZE = 160  # FaceNet requires 160x160 images

# ✅ Load Pretrained InceptionV3 Model
elif USE_MODEL == "InceptionV3":
    inception_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    IMG_SIZE = 299  # InceptionV3 requires 299x299 images


def extract_features_facenet(img_path):
    """Extract features using FaceNet."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize for FaceNet
    img = np.expand_dims(img, axis=0)
    
    img = img.astype("float32") / 255.0  # Normalize

    features = facenet_model.predict(img)
    return features.flatten()


def extract_features_inception(img_path):
    """Extract features using InceptionV3."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize for InceptionV3
    img = np.expand_dims(img, axis=0)
    
    img = img.astype("float32")
    img = inception_preprocess(img)  # Apply preprocessing

    features = inception_model.predict(img)
    return features.flatten()


# ✅ Feature Extraction Process
embeddings = []
labels = []

dataset_path = os.path.abspath("../../../datasets/processed_train")  # Ensure absolute path

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    if not os.path.isdir(person_path):  # Skip if not a directory
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        print(f"Processing {img_path}...")

        # ✅ Choose the model
        if USE_MODEL == "FaceNet":
            features = extract_features_facenet(img_path)
        elif USE_MODEL == "InceptionV3":
            features = extract_features_inception(img_path)

        embeddings.append(features)
        labels.append(person)

# Convert lists to NumPy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# ✅ Save extracted features
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)

print("Feature extraction completed!")
