import cv2
import numpy as np
import tensorflow as tf
import time

# Load trained CNN model
model = tf.keras.models.load_model("./model/cnn_model.h5")

# Define image shape (must match training shape)
IMG_HEIGHT = 112
IMG_WIDTH = 92

# Authenticated ID
AUTH_ID = 18
CONF_THRESHOLD = 70

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = None
door_opened = False
pause_time = None
recognized_id = None
last_detected_face = None

def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def generate_frames():
    global door_opened, pause_time, cap, recognized_id, last_detected_face

    if cap is None:
        cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            processed_face = preprocess_image(face_roi)

            prediction = model.predict(processed_face)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"ID: {predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_detected_face = cv2.resize(frame[y:y+h, x:x+w], (100, 100))

            if predicted_class == AUTH_ID and confidence >= CONF_THRESHOLD:
                door_opened = True
                pause_time = time.time()
                recognized_id = int(predicted_class)
                cap.release()
                cap = None
                return

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_face_preview():
    global last_detected_face
    while True:
        if last_detected_face is not None:
            ret, buffer = cv2.imencode('.jpg', last_detected_face)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
