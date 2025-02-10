from sklearn.svm import SVC
import pickle
import numpy as np

# Load extracted features
with open("face_embeddings.pkl", "rb") as f:
    embeddings, labels = pickle.load(f)

print(f"Loaded {len(embeddings)} embeddings with {len(set(labels))} unique labels.")

# Convert labels to NumPy array (to prevent any string or object issues)
labels = np.array(labels)

print("Training SVM classifier...")

# Train SVM classifier
classifier = SVC(kernel='linear', probability=True)

# Debugging: Check the shape of embeddings
print(f"Shape of embeddings: {embeddings.shape}, Shape of labels: {labels.shape}")

# Fit the classifier
classifier.fit(embeddings, labels)

# Save model
with open("face_recognition_svm.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("Classifier training completed!")
