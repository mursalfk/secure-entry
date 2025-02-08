import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("./model/cnn_model.h5")

# Define image shape (must match training shape)
IMG_HEIGHT = 112
IMG_WIDTH = 92

# Load OpenCV's face detector (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_image(image):
    """Preprocess the image to match the model input."""
    # Check if image is already grayscale (1 channel)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not already

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def recognize_face_live():
    """Real-time face recognition from webcam."""
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not access the camera")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]  # Extract face ROI
            processed_face = preprocess_image(face_roi)  # Preprocess face
            
            # Predict face ID
            prediction = model.predict(processed_face)
            predicted_class = np.argmax(prediction)  # Get class ID
            confidence = np.max(prediction) * 100  # Confidence score

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display prediction result on the frame
            text = f"ID: {predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the real-time frame
        cv2.imshow("Real-Time Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close all windows

# Run the real-time face recognition function
if __name__ == "__main__":
    recognize_face_live()
