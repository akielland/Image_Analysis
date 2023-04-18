import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


data_dir = "/Users/anders/Documents/IN4310/mandatory/mandatory1_data/"
# subdirectories containing each class
subdirs = [x for x in os.listdir(data_dir) if not x.startswith('.')]  # skip hidden files/directories

file_paths = []   # file paths and labels
labels = []       # labels corresponding to the above list of images

# Reduce the number of samples for coding testing
num_samples_per_class = 1000//6

# Loop over the subdirectories and get the file paths and labels: HERE THE FULL DATASET
for i, subdir in enumerate(subdirs):
    class_dir = os.path.join(data_dir, subdir)
    image_files = os.listdir(class_dir)
    image_paths = [os.path.join(class_dir, f) for f in image_files]
    file_paths.extend(image_paths)
    labels.extend([i]*len(image_files))  # label each image with its corresponding class index

# Loop over the subdirectories and get a random selection of file paths and labels: HERE THE REDUCED DATASET
for i, subdir in enumerate(subdirs):
    class_dir = os.path.join(data_dir, subdir)
    image_files = os.listdir(class_dir)
    random.shuffle(image_files)  # shuffle the list of images
    image_files = image_files[:num_samples_per_class]  # select the first num_samples_per_class images
    image_paths = [os.path.join(class_dir, f) for f in image_files]
    file_paths.extend(image_paths)
    labels.extend([i]*len(image_files))  # label each image with its corresponding class index

# Split the data into train, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=300, random_state=123, stratify=labels)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=200, random_state=123, stratify=train_labels)
# train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=3000, random_state=123, stratify=labels)
# train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=2000, random_state=123, stratify=train_labels)

# Print the number of samples in each set
print(f'Train set: {len(train_files)} samples')
print(f'Validation set: {len(val_files)} samples')
print(f'Test set: {len(test_files)} samples')

train_set = set(train_files)
val_set = set(val_files)
test_set = set(test_files)

if len(train_set.intersection(val_set)) > 0:
    print("Warning: Train and validation sets overlap!")
if len(train_set.intersection(test_set)) > 0:
    print("Warning: Train and test sets overlap!")
if len(val_set.intersection(test_set)) > 0:
    print("Warning: Validation and test sets overlap!")

# Save the file paths and labels to a .npz file
np.savez("image_data.npz", train_files=train_files, train_labels=train_labels, val_files=val_files, val_labels=val_labels, test_files=test_files, test_labels=test_labels)