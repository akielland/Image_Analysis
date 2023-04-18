import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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

# Convert the labels list to a numpy array
labels = np.array(labels)

# Perform stratified splitting of the data into train, validation, and test sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(file_paths, labels))
train_files, test_files = [file_paths[i] for i in train_indices], [file_paths[i] for i in test_indices]
train_labels, test_labels = labels[train_indices], labels[test_indices]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_indices, val_indices = next(splitter.split(train_files, train_labels))
train_files, val_files = [train_files[i] for i in train_indices], [train_files[i] for i in val_indices]
train_labels, val_labels = train_labels[train_indices], train_labels[val_indices]

# Print the number of samples in each set
print(f'Train set: {len(train_files)} samples')
print(f'Validation set: {len(val_files)} samples')
print(f'Test set: {len(test_files)} samples')


