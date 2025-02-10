from flask import Flask, render_template, Response, redirect, url_for, jsonify, session
import cv2
import time
from camera import generate_frames, generate_face_preview
from admin import admin_blueprint
from config import SECRET_KEY
import base64

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Function to encode images in Base64
def b64encode_filter(data):
    return base64.b64encode(data).decode('utf-8') if data else ""

# Register the filter in Jinja2 **BEFORE running the app**
app.jinja_env.filters['b64encode'] = b64encode_filter

# Global Variables
cap = None
door_opened = False
pause_time = None
recognized_id = None

@app.route('/')
def home():
    """Homepage with navigation buttons."""
    return render_template('index.html')

@app.route('/open_door')
def open_door():
    """Redirect to face recognition."""
    return render_template('face_recognition.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_preview')
def face_preview():
    """Face preview streaming route."""
    return Response(generate_face_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/exit')
def exit_app():
    """Close the application."""
    cv2.destroyAllWindows()
    exit(0)

# Register Admin Blueprint
app.register_blueprint(admin_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
