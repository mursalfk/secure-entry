from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time

app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model("./model/cnn_model.h5")

# Define image shape (must match training shape)
IMG_HEIGHT = 112
IMG_WIDTH = 92

# Authenticated ID
AUTH_ID = 18

# Confidence threshold
CONF_THRESHOLD = 70

# Load OpenCV's face detector (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Global Variables
cap = None  # Camera is initially closed
door_opened = False
pause_time = None
recognized_id = None  # Store the recognized user's ID
last_detected_face = None  # Store last detected face preview

def preprocess_image(image):
    """Preprocess the image to match the model input."""
    if len(image.shape) == 3:  # Convert to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize to model input size
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def generate_frames():
    """Captures frames from the webcam and performs real-time face recognition."""
    global door_opened, pause_time, cap, recognized_id, last_detected_face

    if cap is None:  # Ensure camera is opened only when needed
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

            # Draw rectangle and prediction text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"ID: {predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save last detected face preview
            last_detected_face = cv2.resize(frame[y:y+h, x:x+w], (100, 100))

            # If Face ID is detected, stop the camera immediately
            if predicted_class == AUTH_ID and confidence >= CONF_THRESHOLD:
                door_opened = True
                pause_time = time.time()
                recognized_id = int(predicted_class)  # Convert to Python int
                cap.release()  # Stop the camera
                cap = None
                return  # Stop streaming

        # Convert frame to bytes for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_face_preview():
    """Generate the last detected face preview for display in the top right corner."""
    global last_detected_face
    while True:
        if last_detected_face is not None:
            ret, buffer = cv2.imencode('.jpg', last_detected_face)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    """Homepage with buttons for opening door or exiting."""
    return render_template('index.html', door_opened=door_opened, recognized_id=recognized_id)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_preview')
def face_preview():
    """Face preview streaming route."""
    return Response(generate_face_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/open_door')
def open_door():
    """Start face recognition and open webcam."""
    global door_opened, recognized_id
    door_opened = False
    recognized_id = None
    return jsonify({'status': 'started'})

@app.route('/exit')
def exit_app():
    """Close the application."""
    global cap
    if cap:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    return jsonify({'status': 'closed'})

@app.route('/door_status')
def door_status():
    """Check door status (if opened) and send response to frontend."""
    global door_opened, pause_time, recognized_id
    if door_opened:
        elapsed_time = int(time.time() - pause_time)
        remaining_time = max(0, 5 - elapsed_time)
        if remaining_time == 0:
            return jsonify({'door_opened': False, 'recognized_id': recognized_id})
        return jsonify({'door_opened': True, 'remaining_time': remaining_time})
    return jsonify({'door_opened': False, 'recognized_id': recognized_id})

if __name__ == '__main__':
    app.run(debug=True)
