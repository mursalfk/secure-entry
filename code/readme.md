### **✅ Features Implemented in the Secure Entry System**

### Project Log: 5:23 PM, 8th February 2025
---

### **🖥️ General Features**
- 🔹 **Flask-based Web Application** for managing secure facial recognition entry.
- 🔹 Uses **PostgreSQL as the database** (via `psycopg2`).
- 🔹 Integrated **OpenCV & TensorFlow** for **real-time face recognition**.
- 🔹 **Bootstrap-styled dashboard** for better UI/UX.

---

### **🔑 User Authentication & Access Control**
- 🔹 **Admin Panel** that is accessible via `/admin_panel` with a **username & password**.
- 🔹 **Session-based authentication** ensures only logged-in admins can access the dashboard.
- 🔹 **Logout Functionality** to end admin sessions securely.

---

### **🚪 Secure Entry System**
- 🔹 **Real-time face recognition system** using OpenCV & a trained CNN model.
- 🔹 **Webcam-based facial recognition** with real-time feedback.
- 🔹 **Door opens only when a registered user is recognized** with **ID=18** and confidence **≥ 70%**.
- 🔹 **Shows the user's face at the top-right corner** during recognition.
- 🔹 **Displays real-time messages** (e.g., `"Door Opened! Camera closing in X seconds"`).
- 🔹 **Automatically logs the last entry timestamp** in the database.

---

### **👤 User & Database Management**
- 🔹 **Admin Dashboard (`/dashboard`)** displays:
  - ✅ **All registered users** with **ID, Name, Username, Face, Last Entered**.
  - ✅ **Profile images** displayed from the database using **Base64 encoding**.
- 🔹 **Add new users** via the admin panel (name, username, and face image).
- 🔹 **Unique usernames** (enforced as **Primary Key** in PostgreSQL).
- 🔹 **Delete users** from the database with a **one-click delete button**.
- 🔹 **Last entry timestamp** is automatically **updated** upon a successful face scan.

---

### **📸 Face Pose Capturing & Model Retraining**
- 🔹 Admin can **capture multiple face poses** for a user.
- 🔹 Captured faces are saved in `user_dataset/[username]` folder.
- 🔹 **Automatic model retraining** after capturing new faces (`retrain_model.py`).
- 🔹 **Live Status Modal (Popup)**
  - ✅ **Shows "Capturing Face Poses..."** while capturing.
  - ✅ **Shows "Model Training in Progress..."** when training starts.
  - ✅ **Shows "Training Completed!"** once retraining is done.
  - ✅ **Has an "OK" button** to close the modal manually.

---

### **🔄 Flask & PostgreSQL Integration**
- 🔹 Flask **Blueprints** for modular code organization (`admin.py`, `database.py`).
- 🔹 **PostgreSQL Database Schema Updated:**
  - ✅ `ID` (Primary Key)
  - ✅ `Name`
  - ✅ `Username` (Unique, Not Null)
  - ✅ `Face` (Stored as `BYTEA` in PostgreSQL)
  - ✅ `Last Entered` (Updated on successful recognition)
- 🔹 Flask **session-based authentication** for better security.
- 🔹 **AJAX-based updates** for capturing faces and training the model.

---

### **🚀 Additional Functionalities**
- 🔹 **Exit Application Button** to stop the app completely.
- 🔹 **Admin Logout Button** to log out securely.
- 🔹 **Error Handling & Debugging Messages** for database issues.
- 🔹 **Optimized UI for a better user experience**.

---

### **🎯 What's Next? (If Needed)**
- [ ] **Improve model accuracy** by training with more diverse face poses.
- [ ] **Enhance UI/UX** with a more polished frontend.
- [ ] **Add logs/history of entries** for better security tracking.

---

🚀 **Current System is Fully Functional & Working Smoothly!** ✅