import os
import random
import shutil

# Set the path to the directory containing the image subdirectories
data_dir = "path/to/data/directory"

# Define the directories for the training, validation, and testing sets
train_dir = "path/to/training/set"
val_dir = "path/to/validation/set"
test_dir = "path/to/testing/set"

# Set the random seed for reproducibility
random.seed(42)

# Create the directories for the training, validation, and testing sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the number of images to use for the training, validation, and testing sets
train_pct = 0.6
val_pct = 0.2
test_pct = 0.2

# Loop through each subdirectory in the data directory
for sub_dir in os.listdir(data_dir):
    sub_dir_path = os.path.join(data_dir, sub_dir)

    # Create a list of the image file paths in the subdirectory
    file_paths = [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path)]

    # Randomly shuffle the list of file paths
    random.shuffle(file_paths)

    # Calculate the number of images for each set
    num_images = len(file_paths)
    num_train = int(num_images * train_pct)
    num_val = int(num_images * val_pct)
    num_test = num_images - num_train - num_val

    # Split the list of file paths into training, validation, and testing sets
    train_files = file_paths[:num_train]
    val_files = file_paths[num_train:num_train+num_val]
    test_files = file_paths[num_train+num_val:]

    # Copy the files to the appropriate directory
    for file_path in train_files:
        shutil.copy(file_path, os.path.join(train_dir, sub_dir))

    for file_path in val_files:
        shutil.copy(file_path, os.path.join(val_dir, sub_dir))

    for file_path in test_files:
        shutil.copy(file_path, os.path.join(test_dir, sub_dir))
