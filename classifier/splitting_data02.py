import os
import numpy as np
from sklearn.model_selection import train_test_split

# path data directory
data_dir = "/Users/anders/Documents/IN4310/mandatory/mandatory1_data/"

# Get the subdirectories containing the images
subdirs = [x for x in os.listdir(data_dir) if not x.startswith('.')]  # skip hidden files/directories

# Create a list to store the file paths and labels
file_paths = []
labels = []

# Loop over the subdirectories and get the file paths and labels
for i, subdir in enumerate(subdirs):
    class_dir = os.path.join(data_dir, subdir)
    image_files = os.listdir(class_dir)
    image_paths = [os.path.join(class_dir, f) for f in image_files]
    file_paths.extend(image_paths)
    labels.extend([i]*len(image_files))  # label each image with its corresponding class index

# Split the data into train, validation, and test sets
# train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)
# train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.25, random_state=42, stratify=train_labels)

# Split the data into train, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=3000, random_state=42, stratify=labels)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=2000, random_state=42, stratify=train_labels)


# Print the number of samples in each set
print(f'Train set: {len(train_files)} samples')
print(f'Validation set: {len(val_files)} samples')
print(f'Test set: {len(test_files)} samples')
