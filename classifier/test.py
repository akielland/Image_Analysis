import os
from urllib.request import urlretrieve
import tarfile
import xml.etree.ElementTree as ET
import scipy.io
import pickle


import os

file_path = '/Users/anders/Documents/IN4310/mandatory/cifar-10-batches-py/test_batch/domestic_cat_s_000907.png'

if os.path.exists(file_path):
    print(f"The file {file_path} exists.")
else:
    print(f"The file {file_path} does not exist.")






# Load the CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
cifar10_subset = Subset(cifar10_dataset, range(200))
print(cifar10_subset)


def load_cfar10_batch(cifar10_dataset_folder_path): #, batch_id):
    with open(cifar10_dataset_folder_path + 'cifar-10-batches-py/test_batch', mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

data_dir = "/Users/anders/Documents/IN4310/mandatory/"
load_cfar10_batch(data_dir)



def get_imagenet_data(data_dir, num_images=200):
    # Download ILSVRC2012_devkit_t12.tar.gz file if not already present
    labels_url = 'https://academictorrents.com/download/5d6d0df7ed81efd5c0834e9a4ebd5583524a4a88.torrent'
    labels_path = os.path.join(data_dir, 'ILSVRC2012_devkit_t12.tar.gz')
    if not os.path.exists(labels_path):
        urlretrieve(labels_url, labels_path)

    # Extract and load synset to integer label mapping
    with tarfile.open(labels_path, 'r:gz') as tar:
        tar.extractall(data_dir)
        meta = scipy.io.loadmat(os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/meta.mat'))
        synset_to_label = {row[0][1][0]: row[0][0] for row in meta['synsets']}

    # Extract XML files and load class labels into a dictionary with filename as key
    labels = {}
    with tarfile.open(os.path.join(data_dir, 'ILSVRC2012_bbox_val_v3.tgz'), 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.xml'):
                xml_file = tar.extractfile(member)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.find('filename').text
                synset = root.find('object').find('name').text
                label = synset_to_label[synset]
                labels[filename] = label

    # Get paths to image files and their corresponding labels
    images_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    image_paths = []
    image_labels = []
    for i, filename in enumerate(os.listdir(images_dir)):
        if i >= num_images:
            break
        if filename.endswith('.JPEG'):
            image_path = os.path.join(images_dir, filename)
            image_paths.append(image_path)
            label = labels[filename]
            image_labels.append(label)

    return image_paths, image_labels


def get_imagenet_data(data_dir, num_images=200):
    # extract labels from ILSVRC2012_bbox_val_v3.tgz file
    labels_path = os.path.join(data_dir, 'ILSVRC2012_bbox_val_v3.tgz')
    with tarfile.open(labels_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    labels_file = os.path.join(data_dir, 'ILSVRC2012_bbox_val_v3/val/ILSVRC2012_bbox_val_v3.txt')

    # load labels into a dictionary with filename as key
    labels = {}
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            label = int(parts[1])
            labels[filename] = label

    # Get paths to image files and their corresponding labels
    # loop over the image files in the ILSVRC2012_img_val directory, get the full path to each image file, and add to the image_paths list
    # get the label of each image from the labels dictionary and add it to the image_labels list
    images_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    image_paths = []
    labels = []
    for i, filename in enumerate(os.listdir(images_dir)):
        if i >= num_images:
            break
        if filename.endswith('.JPEG'):
            image_path = os.path.join(images_dir, filename)
            image_paths.append(image_path)
            label = labels[filename]
            labels.append(label)

    return image_paths, labels


def get_imagenet_data(data_dir, num_images=200):
    # extract labels from ILSVRC2012_bbox_val_v3.tgz file
    labels_path = os.path.join(data_dir, 'ILSVRC2012_bbox_val_v3.tgz')
    with tarfile.open(labels_path, 'r:gz') as tar:
        labels_file = tar.extractfile('ILSVRC2012_bbox_val_v3/val/ILSVRC2012_bbox_val_v3.txt')
        labels_content = labels_file.read().decode('utf-8')

    # load labels into a dictionary with filename as key
    labels = {}
    for line in labels_content.split('\n'):
        parts = line.strip().split()
        if len(parts) == 2:
            filename = parts[0]
            label = int(parts[1])
            labels[filename] = label

    # Get paths to image files and their corresponding labels
    # loop over the image files in the ILSVRC2012_img_val directory, get the full path to each image file, and add to the image_paths list
    # get the label of each image from the labels dictionary and add it to the image_labels list
    images_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    image_paths = []
    image_labels = []
    for i, filename in enumerate(os.listdir(images_dir)):
        if i >= num_images:
            break
        if filename.endswith('.JPEG'):
            image_path = os.path.join(images_dir, filename)
            image_paths.append(image_path)
            label = labels[filename]
            image_labels.append(label)

    return image_paths, image_labels