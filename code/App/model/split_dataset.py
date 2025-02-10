import os
import shutil
import random

# Paths
DATASET_DIR = "../../datasets"  # Original dataset folder
TRAIN_DIR = "../../datasets/train"
TEST_DIR = "../../datasets/test"

# Split ratio
TEST_SIZE = 0.2  # 20% for testing

# Create train/test folders
for folder in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

# Split data
for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    images = os.listdir(person_path)
    
    if len(images) < 5:  # Skip identities with too few images
        continue

    random.shuffle(images)
    split_index = int(len(images) * (1 - TEST_SIZE))

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create new directories
    os.makedirs(os.path.join(TRAIN_DIR, person), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, person), exist_ok=True)

    # Move files
    for img in train_images:
        shutil.copy(os.path.join(person_path, img), os.path.join(TRAIN_DIR, person, img))

    for img in test_images:
        shutil.copy(os.path.join(person_path, img), os.path.join(TEST_DIR, person, img))

print("Dataset split completed!")
