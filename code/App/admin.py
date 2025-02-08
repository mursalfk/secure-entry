import cv2
import os
import subprocess
import time
from flask import Blueprint, request, redirect, url_for, flash, render_template, session, jsonify
from database import get_user_by_username, get_all_users, add_user, delete_user, update_last_entered

# **Initialize the Blueprint before defining routes**
admin_blueprint = Blueprint("admin", __name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

### --- ADMIN LOGIN PANEL --- ###
@admin_blueprint.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    """Admin login page."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123":  # Update credentials if needed
            session['admin_logged_in'] = True
            return redirect(url_for('admin.dashboard'))  # Redirect after login
        else:
            flash("Invalid Credentials", "danger")

    return render_template('admin_login.html')

### --- ADMIN DASHBOARD --- ###
@admin_blueprint.route('/dashboard')
def dashboard():
    """Admin dashboard page."""
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin.admin_panel'))

    users = get_all_users()  # Fetch updated user list
    return render_template('dashboard.html', users=users)

### --- ADD NEW USER --- ###
@admin_blueprint.route('/add_user', methods=['POST'])
def add_new_user():
    """Route to add a new user (Upload Name & Image)."""
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin.admin_panel'))

    name = request.form['name']
    username = request.form['username']
    image_file = request.files['image']

    if image_file:
        image_data = image_file.read()
        add_user(name, username, image_data)
        flash("User added successfully!", "success")
    
    return redirect(url_for('admin.dashboard'))

### --- DELETE USER --- ###
@admin_blueprint.route('/delete_user/<string:username>')
def remove_user(username):
    """Delete a user from the database."""
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin.admin_panel'))  # Ensure only logged-in admins can delete users

    delete_user(username)  # Calls function from database.py
    flash("User deleted successfully!", "success")
    
    return redirect(url_for('admin.dashboard'))

### --- CAPTURE FACE POSES (Now with AJAX Updates) --- ###
@admin_blueprint.route('/capture_faces/<string:username>', methods=['POST'])
def capture_faces(username):
    """Capture multiple face poses for a user with live status updates."""
    if 'admin_logged_in' not in session:
        return jsonify({"status": "error", "message": "Not Authorized"}), 403

    user = get_user_by_username(username)
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404

    user_folder = f"user_dataset/{username}"
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    # **Send message to UI: Capturing Faces Started**
    status_updates = ["Capturing face poses..."]

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_path = f"{user_folder}/{count}.jpg"
            cv2.imwrite(face_path, face_img)
            count += 1

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # **Send message to UI: Training Started**
    status_updates.append("Model training in progress...")

    # **Trigger Model Retraining**
    subprocess.run(["python", "retrain_model.py"], check=True)

    # **Update Last Entered Timestamp**
    update_last_entered(username)

    # **Send message to UI: Training Completed**
    status_updates.append("Training completed!")

    return jsonify({"status": "success", "update": status_updates[-1]})  # Show only latest update

### --- LOGOUT ADMIN --- ###
@admin_blueprint.route('/logout')
def logout():
    """Logout Admin"""
    if 'admin_logged_in' in session:
        session.pop('admin_logged_in', None)
        flash("Logged out successfully!", "info")
    
    return redirect(url_for('admin.admin_panel'))
