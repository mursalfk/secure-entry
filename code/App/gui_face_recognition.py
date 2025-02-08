import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import time

# Load trained CNN model
model = load_model("./model/cnn_model.h5")

# Define image shape (must match training shape)
IMG_HEIGHT = 112
IMG_WIDTH = 92

# Authenticated ID
AUTH_ID = 18

# Confidence threshold
CONF_THRESHOLD = 70

# Load OpenCV's face detector (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        
        # Start the app in fullscreen
        self.root.attributes("-fullscreen", True)

        # Label for Webcam Feed
        self.video_label = Label(self.root)
        self.video_label.pack()

        # Buttons
        self.open_door_btn = Button(self.root, text="Open Door", command=self.start_recognition, font=("Arial", 14))
        self.open_door_btn.pack(pady=20)

        self.exit_btn = Button(self.root, text="Exit", command=self.exit_app, font=("Arial", 14), bg="red", fg="white")
        self.exit_btn.pack(pady=10)

        self.cap = None  # Webcam capture object
        self.running = False  # Webcam status
        self.text_label = Label(self.root, text="", font=("Arial", 16, "bold"))
        self.text_label.pack(pady=20)

        self.door_opened = False  # Track if the door is opened
        self.timer_start = None  # Track when the timer starts
        self.pause_camera_time = None  # Track when to pause camera

    def preprocess_image(self, image):
        """Preprocess the image to match the model input."""
        if len(image.shape) == 3:  # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize
        image = image.astype("float32") / 255.0  # Normalize
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def start_recognition(self):
        """Starts real-time face recognition."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        self.running = True
        self.door_opened = False  # Reset door status
        self.timer_start = None  # Reset timer
        self.pause_camera_time = None  # Reset pause timer
        self.show_frame()

    def show_frame(self):
        """Continuously captures frames from the webcam and performs recognition."""
        if not self.running:
            return
        
        if self.door_opened:
            # If door is opened, pause webcam after 1 second
            elapsed_time = time.time() - self.pause_camera_time
            if elapsed_time >= 1 and self.cap is not None:
                self.cap.release()  # Pause the camera feed
                self.cap = None  # Remove camera instance

            # Start countdown (5 seconds before exit)
            elapsed_exit_time = time.time() - self.timer_start
            remaining_time = max(0, 5 - int(elapsed_exit_time))  # Ensure non-negative countdown
            
            if remaining_time == 0:
                self.exit_app()  # Exit after 5 seconds
                return
            
            self.text_label.config(text=f"🚪 Door Opened! App closing in {remaining_time} second(s)", fg="green")
            self.root.after(1000, self.show_frame)  # Update countdown every second
            return

        ret, frame = self.cap.read()
        if not ret:
            self.text_label.config(text="Error: Failed to capture image")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            processed_face = self.preprocess_image(face_roi)

            prediction = model.predict(processed_face)
            predicted_class = np.argmax(prediction)  # Get class ID
            confidence = np.max(prediction) * 100  # Confidence score

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display prediction
            text = f"ID: {predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Check if the ID is AUTH_ID with ≥ CONF_THRESHOLD confidence
            if predicted_class == AUTH_ID and confidence >= CONF_THRESHOLD:
                self.door_opened = True  # Mark door as opened
                self.pause_camera_time = time.time()  # Start pause timer (1 sec before pausing)
                self.timer_start = time.time()  # Start countdown (5 sec before closing)
                self.text_label.config(text="🚪 Door Opened! Pausing camera in 1 second...", fg="green")
                self.root.after(1000, self.show_frame)  # Update for 1 sec delay before pausing
                return  # Continue for 1 second before pausing

            else:
                self.text_label.config(text="❌ Access Denied", fg="red")

        # Convert frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Display the webcam feed in the GUI
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        # Keep updating the frames
        self.root.after(10, self.show_frame)

    def exit_app(self):
        """Stops the camera and closes the app."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
