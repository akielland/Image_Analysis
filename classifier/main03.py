import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os

import PIL.Image

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# from getimagenetclasses import parseclasslabel, parsesynsetwords, get_classes
# Try other models https://pytorch.org/vision/stable/models.html
from torchvision.models import resnet18


# Import the train, validation, and test file paths and labels from the .npz file
# Load the data from the .npz file
data = np.load("image_data.npz")
# Load the data from the .npz file
data = np.load("image_data.npz")
# Access the arrays by name
train_files = data["train_files"]
train_labels = data["train_labels"]
val_files = data["val_files"]
val_labels = data["val_labels"]
test_files = data["test_files"]
test_labels = data["test_labels"]

# check data
print(f'Train set: {len(train_files)} samples')
print(f'Validation set: {len(val_files)} samples')
print(f'Test set: {len(test_files)} samples')

data_dir = "/Users/anders/Documents/IN4310/mandatory/mandatory1_data/"

class DatasetSixClasses(Dataset):
  def __init__(self, root_dir, trvaltest, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.imgfilenames=[]
    self.labels=[]

    if trvaltest==0:
        self.imgfilenames = train_files
        self.labels = train_labels
    if trvaltest==1:
        self.imgfilenames = val_files
        self.labels = val_labels
    if trvaltest==2:
        self.imgfilenames = test_files
        self.labels = test_labels

  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):
      image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
      label = self.labels[idx]

      if self.transform:
        image = self.transform(image)

      sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}
      return sample



  if __name__ == '__main__':    # Press the green button in the gutter to run the script.
      config = {
          'batch_size': 20,
          'use_cuda': False,  # True=use Nvidia GPU | False use CPU
          'log_interval': 5,  # How often to display (batch) loss during training
          'epochs': 20,  # Number of epochs
          'learningRate': 0.001
      }

      train_dataset = DatasetSixClasses(data_dir)
      # DataLoaders
      train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
      val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)





