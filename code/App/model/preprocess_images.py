import cv2
import os
import dlib
import numpy as np

# Load Dlib's face detector
face_detector = dlib.get_frontal_face_detector()

def preprocess_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for person in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person)
        output_person_path = os.path.join(output_dir, person)
        os.makedirs(output_person_path, exist_ok=True)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            # ✅ Check if the image loaded properly
            if img is None:
                print(f"❌ ERROR: Skipping {img_path} (Cannot read image)")
                continue

            # ✅ Convert to grayscale & check shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if gray.dtype != np.uint8:
                gray = cv2.convertScaleAbs(gray)  # Ensure uint8 format

            if gray.ndim != 2 or gray.shape[0] == 0 or gray.shape[1] == 0:
                print(f"⚠️ ERROR: Skipping {img_path} (Invalid dimensions: {gray.shape})")
                continue

            # 🔍 DEBUG: Save a sample grayscale image for checking
            debug_gray_path = os.path.join(output_person_path, f"debug_{img_name}")
            cv2.imwrite(debug_gray_path, gray)
            print(f"✅ DEBUG: Saved grayscale image for {img_name}")

            try:
                faces = face_detector(gray)

                # ✅ If no faces are found, log and skip
                if len(faces) == 0:
                    print(f"⚠️ WARNING: No face detected in {img_path}")
                    continue

                for i, face in enumerate(faces):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()

                    # ✅ Check valid face dimensions
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        print(f"⚠️ ERROR: Skipping {img_path} (Invalid face dimensions: {x},{y},{w},{h})")
                        continue

                    cropped_face = img[y:y+h, x:x+w]
                    resized_face = cv2.resize(cropped_face, (112, 92))

                    save_path = os.path.join(output_person_path, f"{i}_{img_name}")
                    cv2.imwrite(save_path, resized_face)

                    print(f"✅ SUCCESS: Processed {save_path}")

            except Exception as e:
                print(f"❌ ERROR: Face detection failed for {img_path}: {e}")

# Run Preprocessing for Train & Test Sets
preprocess_images("../../../datasets/train", "../../../datasets/processed_train")
preprocess_images("../../../datasets/test", "../../../datasets/processed_test")

print("✅ Preprocessing Completed!")
