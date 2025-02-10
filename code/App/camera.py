import cv2
import numpy as np
import tensorflow as tf
import time
from database import get_user_name_by_id  # Fetch name from DB

# Load trained CNN model
model = tf.keras.models.load_model("./model/cnn_model.h5")

# Define image shape (must match training shape)
IMG_HEIGHT = 112
IMG_WIDTH = 92

# Confidence threshold
CONF_THRESHOLD = 70

# Load OpenCV's face detector (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = None
door_opened = False
pause_time = None
recognized_id = None
last_detected_face = None
face_detected_time = None  # Time when a face is detected

def preprocess_image(image):
    """Preprocess the image to match the model input."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def generate_frames():
    global door_opened, pause_time, cap, recognized_id, last_detected_face, face_detected_time

    if cap is None:
        cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            if face_detected_time is None:
                face_detected_time = time.time()  # Start timer when face appears

            elapsed_time = time.time() - face_detected_time

            # **Wait for the face to be stable for 3 seconds before recognition**
            if elapsed_time < 3:
                cv2.putText(frame, "Analyzing...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            else:
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    processed_face = preprocess_image(face_roi)

                    prediction = model.predict(processed_face)
                    predicted_class = int(np.argmax(prediction))  # Convert NumPy int64 to Python int
                    confidence = np.max(prediction) * 100  # Confidence score

                    # Fetch username from database
                    user_name = get_user_name_by_id(predicted_class)

                    if not user_name:
                        user_name = "Unknown User"  # Handle missing users

                    print(f"User Detected: {user_name}")

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{user_name} ({confidence:.2f}%)"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    last_detected_face = cv2.resize(frame[y:y+h, x:x+w], (100, 100))

                    if confidence >= CONF_THRESHOLD and user_name != "Unknown User":
                        door_opened = True
                        pause_time = time.time()
                        recognized_id = predicted_class  # Now it's a standard int

                        # **Display "Welcome Home" message for 2 seconds**
                        cv2.putText(frame, f"Welcome Home {user_name}!", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Face Recognition", frame)
                        cv2.waitKey(2000)  # Show message for 2 seconds
                        
                        # **Fix: Release cap only if it's not None**
                        if cap is not None:
                            cap.release()
                            cap = None

                        cv2.destroyAllWindows()
                        return  # Stop after recognition
        else:
            face_detected_time = None  # Reset timer if face disappears

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_face_preview():
    """Stream last detected face."""
    global last_detected_face
    while True:
        if last_detected_face is not None:
            ret, buffer = cv2.imencode('.jpg', last_detected_face)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
