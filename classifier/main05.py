import PIL.Image
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
import torch
# torch.set_num_threads(2)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
import efficientnet_pytorch as efn
import itertools


# Ignore the DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def split_data(reduced_num_samples=False):
    # list of subdirectories containing each class
    subdirs = [x for x in os.listdir(data_dir) if not x.startswith('.')]  # skip hidden files/directories

    file_paths = []   # list to store file paths of images
    labels = []       # labels corresponding to the above list of images

    # reduce the number of samples for testing of code
    num_samples_per_class = 1000//6

    # loop over the subdirectories and get a random selection of file paths and labels
    for i, subdir in enumerate(subdirs):
        class_dir = os.path.join(data_dir, subdir)
        image_files = os.listdir(class_dir)
        if reduced_num_samples:
            image_files = image_files[:num_samples_per_class]  # select the first num_samples_per_class images
        image_paths = [os.path.join(class_dir, f) for f in image_files]
        file_paths.extend(image_paths)
        labels.extend([i]*len(image_files))  # label each image with its corresponding class index

    # split data into train, validation, and test sets which are stored as paths in lists
    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=300, random_state=123, stratify=labels)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=200, random_state=123, stratify=train_labels)

    # check if all sets are disjoint
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    if len(train_set.intersection(val_set)) > 0:
        print("Warning: train and validation sets overlap")
    if len(train_set.intersection(test_set)) > 0:
        print("Warning: train and test sets overlap")
    if len(val_set.intersection(test_set)) > 0:
        print("Warning: validation and test sets overlap")

    return train_files, val_files, test_files, train_labels, val_labels, test_labels


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
        else:
            image = transforms.ToTensor()(image)  # ? NOT enough: whay NOT
            image = transforms.Resize(size=224)(image)

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}
        return sample


def train_epoch(model, train_loader, criterion, device, optimizer):
    model.train()   # set the model to training mode

    losses = list()
    for batch_idx, data in enumerate(train_loader):
        inputs = data['image']
        inputs.to(device)
        labels = data['label']
        labels.to(device)

        # Forward
        output = model(inputs)  # output tensor = prediction scores (logits) for each class in the output space
        loss = criterion(output, labels)
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % config['log_interval'] == 0:
            print('current mean of losses ', np.mean(losses))

    return np.mean(losses)  # DO I NEED TO RETURN ANYTHING?


def train_model(train_loader, val_loader,  model,  criterion, optimizer, device, num_epochs):
    best_measure_val = 0
    best_epoch = -1
    losses_epochs_train1 = []
    losses_epochs_train2 = []
    losses_epochs_val = []
    measure_epochs_train = []
    measure_epochs_val = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        model.train()
        losses = train_epoch(model, train_loader, criterion, device, optimizer)
        losses_epochs_train1.append(losses)

        model.eval()
        measure, mean_losses = evaluate_acc(model, train_loader, criterion, device)
        losses_epochs_train2.append(mean_losses)
        measure_epochs_train.append(measure)

        measure, mean_losses = evaluate_acc(model, val_loader, criterion, device)
        losses_epochs_val.append(mean_losses)
        measure_epochs_val.append(measure)

        print('mean losses train1:', losses_epochs_train1)
        print('mean losses train2:', losses_epochs_train2)
        print('performance measures (acc) train:', measure_epochs_train)
        print('mean losses val:', losses_epochs_val)
        print('performance measures (acc) val:', measure_epochs_val)

        if measure > best_measure_val:
           #  best_weights = model.state_dict()
            best_model = model
            best_measure_val = measure
            best_epoch = epoch
            print('current best acc at validation set is', best_measure_val, 'achieved in  epoch ', best_epoch+1)

    return best_measure_val, best_model, losses_epochs_train1, losses_epochs_val, measure_epochs_train, measure_epochs_val


def run_training():
    device = torch.device("cuda" if config['use_cuda'] else "cpu")

    # prepare data
    train_dataset = DatasetSixClasses(data_dir, trvaltest=0, transform=transform)
    val_dataset = DatasetSixClasses(data_dir, trvaltest=1, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)  # ??
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # model: load and updating the output layer with the respective number of classes (6)
    num_classes = 6
    # Load the pre-trained EfficientNet-B0 model
    model = efn.EfficientNet.from_pretrained('efficientnet-b0')
    # Replace the last layer with a new fully connected layer for 6 classes and reset the parameters for this layer
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    learning_rates = config['learning_rate']
    best_hyperparameter = None
    best_measure_val_lr = 0
    for lr in learning_rates:

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        best_measure_val, best_model, losses_epochs_train, losses_epochs_val, measure_epochs_train, measure_epochs_val = \
            train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs = config['num_epochs'])

        if best_hyperparameter is None:
            best_hyperparameter = lr
            model_chosen = best_model
            best_measure_val_lr = best_measure_val
        elif best_measure_val > best_measure_val_lr:
            best_hyperparameter = lr
            model_chosen = best_model
            best_measure_val_lr = best_measure_val

            losses_epochs_train1 = losses_epochs_train1
            losses_epochs_val = losses_epochs_val
            measure_epochs_train = measure_epochs_train
            measure_epochs_val = measure_epochs_val

    print('best_hyperparameter:', best_hyperparameter)
    torch.save(model_chosen, 'best_model.pt')
    return losses_epochs_train, losses_epochs_val, measure_epochs_train, measure_epochs_val


def run_model_test(model):
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # prepare dataset with test data
    test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    predictions_tensor = torch.empty(len(test_loader.dataset), dtype=torch.long) # empty tensor: predictions
    softmax_scores_tensor = torch.empty(len(test_loader.dataset), model._fc.out_features)# empty tensor: softmax scores

    # lists for true labels and predicted labels for computing accuracy and average precision
    true_labels = []
    predicted_labels = []
    class_probs = [[] for _ in range(6)]
    class_scores = {'accuracy': [], 'average_precision': []}

    # Loop over the test dataset and make predictions
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # run test data through model and calculate predictions and softmax score from logits
            outputs = model(inputs)
            outputs = outputs.to('cpu')
            probabilities = torch.softmax(outputs, dim=1)  # softmax scores of the predictions (kind of probabilities for sample to each class)
            predicted_classes = torch.argmax(probs, dim=1)  # the predicted class of the samples of the batch

            # save predicted class label and softmax scores to the tensors
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            softmax_scores_tensor[start_idx:end_idx] = probabilities
            predictions_tensor[start_idx:end_idx] = predicted_classes

            # save true and predicted labels for computing accuracy and precision (AP)
            true_labels.extend(labels.cpu().numpy()) # Convert pytorch tensors to numpy arrays
            predicted_labels.extend(predicted_classes.cpu().numpy())
            # collect class probabilities from each batch
            for i in range(6):
                class_probs[i].extend(probabilities[:,i].cpu().numpy())  # the inner lists contains the softmax probabilities for the corresponding class across all predictions in the dataset

        for i in range(6):
            class_labels = [1 if x == i else 0 for x in true_labels]  # recode to 1 and 0, length of the whole dataset in order to operate on binary classification
            class_preds = [1 if x == i else 0 for x in predicted_labels]

            class_scores['accuracy'].append(metrics.accuracy_score(class_labels, class_preds))
            # calculate average precision score based on softmax probabilities
            class_scores['average_precision'].append(metrics.average_precision_score(class_labels, class_probs[i]))

    target_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # mean over the classes
    mean_accuracy = np.mean(class_scores['accuracy'])
    mean_average_precision = np.mean(class_scores['average_precision'])

    # save files with the tensor of the predictions and softmax score, respectively
    torch.save(predictions_tensor, 'predictions.pt')
    torch.save(softmax_scores_tensor, 'softmax_scores.pt')

    print(f"class wise accuracy: {class_scores['accuracy']}")
    print(f'accuracy for all classifications of the model: {mean_accuracy}')
    print(f"class wise average precision score: {class_scores['average_precision']}")
    print(f'average precision score (mean over all classes): {mean_average_precision}')

    return predictions_tensor, softmax_scores_tensor, class_scores


def plot_loss(train_loss, val_loss):
    # Plot the loss training and validation
    # plt.figure()
    fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(train_loss, 'b', label='train loss')
    plt.plot(val_loss, 'r', label='validation loss')
    plt.grid()
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    fig.savefig('loss_vs_epoch.png', dpi=300, bbox_inches='tight')

def extract_module_names(model):
    for module_name, mod in model.named_modules():
        print(module_name)

def getActivation(name, activation):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def update_mean(mean, percentage, n):
    u = percentage.mean()  # calculate the average percentage of non-positive values
    n_step = len(percentage)  # number of elements in this batch
    m_new = (mean * n + u * n_step) / (n + n_step)  # calculate new mean
    n_new = n + n_step  # update the number of elements seen so far
    return m_new, n_new

def feature_map_statistics(model):
    test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)
    # Create a new DataLoader with the first 200 samples
    test_loader_200 = DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, sampler=itertools.islice(test_loader.sampler, 200))

    activation = {} # dict to store the activations for each feature map
    # register forward hooks on the layers of choice
    h1 = model._blocks[0]._depthwise_conv.register_forward_hook(getActivation('L0', activation))
    h2 = model._blocks[1]._expand_conv.register_forward_hook(getActivation('L1', activation))
    h3 = model._blocks[5]._expand_conv.register_forward_hook(getActivation('L5', activation))
    h4 = model._blocks[8]._expand_conv.register_forward_hook(getActivation('L8', activation))
    h5 = model._blocks[14]._expand_conv.register_forward_hook(getActivation('L14', activation))
    L0, L1, L5, L8, L14 = [], [], [], [], []

    # go through all the batches in the dataset and calculate percentage
    overall_mean_perc = 0
    n_step = test_loader.batch_size
    n = test_loader.batch_size
    for batch_idx, data in enumerate(test_loader_200):
        inputs = data['image']
        # forward pass -- getting the outputs
        outputs = model(inputs)
        # collect the activations in the correct list
        L0.append(activation['L0'])
        L1.append(activation['L1'])
        L5.append(activation['L5'])

        # calculate the percentage of non-positive values

        inputs = activation['L0']
        percent_non_positive = torch.mean((inputs <= 0).float())
        print(percent_non_positive)
        # update the overall mean value
        # overall_mean_perc = (overall_mean_perc * batch_idx + percent_non_positive.item()) / (batch_idx + 1)
        # overall_mean_perc, n = update_mean(overall_mean_perc, percent_non_positive, n)
        overall_mean_perc = (overall_mean_perc * n + percent_non_positive * n_step) / (n + n_step)  # calculate new mean
        n = n + n_step  # update the number of elements seen so far

    print(activation['L0'].size())
    print(overall_mean_perc)

if __name__ == '__main__':
    data_dir = "/Users/anders/Documents/IN4310/mandatory/mandatory1_data/"
    # data_dir = "../mandatory1_data/"
    config =    {
                'batch_size': 32,
                'use_cuda': False,  # True=use Nvidia GPU | False use CPU
                'log_interval': 5,  # How often to display (batch) loss during training
                'num_epochs': 2,    # Number of epochs
                'learning_rate': [0.01]  # [1e-2, 1e-3, 1e-4]
                }
    # define transforms for data augmentation and normalization; WHY is this necessary?
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split the dataset
    train_files, val_files, test_files, train_labels, val_labels, test_labels = split_data(reduced_num_samples=True)
    # Print the number of samples in each set
    print(f'Train set: {len(train_files)} samples')
    print(f'Validation set: {len(val_files)} samples')
    print(f'Test set: {len(test_files)} samples')


    # run training
    losses_epochs_train, losses_epochs_val, measure_epochs_train, measure_epochs_val = run_training()

    # Print result of training and evaluation
    print(f'losses_epochs_train: {losses_epochs_train}')
    print(f'losses_epochs_val: {losses_epochs_val}')
    print(f'measure_epochs_train: {measure_epochs_train}')
    print(f'measure_epochs_val: {measure_epochs_val}')

    plot_loss(losses_epochs_train, losses_epochs_val)

    # load best model and evaluate it with the test dataset
    best_model = torch.load('best_model.pt')

    predictions_tensor, softmax_scores_tensor, class_scores = run_model_test(best_model)

    # extract_module_names(best_model)
    feature_map_statistics(best_model)

