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
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
import efficientnet_pytorch as efn

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
        # random.shuffle(image_files)  # shuffle the list of images: I dont think is necessary
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
        # inputs = data['image'].to(device)
        # labels = data['label'].to(device)

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

    #return best_epoch, best_measure_val, best_model, losses_epochs_train1, losses_epochs_val, measure_epochs_val
    return best_measure_val, best_model, losses_epochs_train1, losses_epochs_val, measure_epochs_train, measure_epochs_val


def run_training():
    device = torch.device("cuda" if config['use_cuda'] else "cpu")

    # prepare data
    train_dataset = DatasetSixClasses(data_dir, trvaltest=0, transform=transform)
    val_dataset = DatasetSixClasses(data_dir, trvaltest=1, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # model: load and updating the output layer with the respective number of classes (6)
    num_classes = 6
    # Load the pre-trained Resnet18 model
    model = models.resnet18(pretrained=True)  # model = models.resnet18(weights="imagenet")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.fc.reset_parameters()  # initialize weights and biases of the new fully connected last layer with random values

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
    # weights_chosen = None
    best_measure_val_lr = 0
    for lr in learning_rates:

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

        best_measure_val, best_model, losses_epochs_train1, losses_epochs_val, measure_epochs_train, measure_epochs_val = \
            train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs = config['num_epochs'])

        if best_hyperparameter is None:
            best_hyperparameter = lr
            # weights_chosen = best_weights
            model_chosen = best_model
            best_measure_val_lr = best_measure_val
        elif best_measure_val > best_measure_val_lr:
            best_hyperparameter = lr
            # weights_chosen = best_weights
            model_chosen = best_model
            best_measure_val_lr = best_measure_val

            losses_epochs_train1 = losses_epochs_train1
            losses_epochs_val = losses_epochs_val
            measure_epochs_train = measure_epochs_train
            measure_epochs_val = measure_epochs_val

    # Print result of training and evaluation
    print('best_hyperparameter:', best_hyperparameter)
    # print(f'best_epoch: {best_epoch+1}')
    # print(f'best_measure: {measure_val}')

    # measure_test, test_mean_losses = evaluate_acc(model, test_loader, criterion, device)

    torch.save(model_chosen, 'best_model.pt')

    # return best_epoch, best_measure_val_lr, measure_test, losses_epochs_train, losses_epochs_val, weights_chosen, model_chosen
    return losses_epochs_train1, losses_epochs_val, measure_epochs_train, measure_epochs_val


def run_model_test(model):
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # prepare dataset with test data
    test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    predictions = torch.empty(len(test_loader.dataset), dtype=torch.long) # empty tensor: predictions
    softmax_scores = torch.empty(len(test_loader.dataset), model._fc.out_features)# empty tensor: softmax scores

    # lists for true labels and predicted labels for computing accuracy and average precision
    true_labels = []
    predicted_labels = []

    # Loop over the test dataset and make predictions
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # run test data through model and calculate predictions and softmax score
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)  # softmax scores of the predictions (kind of probabilities for sample to each class)
            _, predicted_classes = torch.max(probabilities, 1)  # the predicted class of the samples of the batch

            # save predicted class label and softmax scores to the tensors
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            predictions[start_idx:end_idx] = predicted_classes
            softmax_scores[start_idx:end_idx] = probabilities

            # save true and predicted labels for computing accuracy and precision (AP)
            # Convert pytorch tensors to numpy arrays
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    # Calculate the mean class-wise accuracy using sklearn.metrics.accuracy_score
    # normalize=True: output is the fraction of correctly classified samples
    # normalize=False: output is the number of correctly classified samples (NumPy array of shape (num_classes, ) with number of true positives for each class)
    # all_classes_accuracy = accuracy_score(true_labels, predicted_labels, normalize=False)
    # all_classes_accuracy = torch.tensor(all_classes_accuracy, dtype=torch.float) / len(test_loader.dataset)

    target_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    # class_wise_accuracy = classification_report(true_labels, predicted_labels, target_names=target_names)
    # class_wise_accuracy = torch.tensor(class_wise_accuracy, dtype=torch.float) / len(test_loader.dataset)

    # Compute the average precision score for each class using using sklearn.metrics.average_precision_score
    # need to one-hot encode the labeling first
    true_labels_binarized = np.zeros((len(true_labels), 6))   # 6=num_classes
    true_labels_binarized[np.arange(len(true_labels)), true_labels] = 1
    # Compute the average precision score for all classes
    average_precision = average_precision_score(true_labels_binarized, softmax_scores.cpu().numpy(), average='macro')

    # Compute the mean average precision score across all classes
    # mean_average_precision = np.mean(list(average_precision.values()))

    # Calculate the average precision
    # DONT WORK
    # average_precision = average_precision_score(true_labels, softmax_scores.cpu(), average='macro')

    # save files with the tensor of the predictions and softmax score, respectively
    torch.save(predictions, 'predictions.pt')
    torch.save(softmax_scores, 'softmax_scores.pt')

    print(f'class wise accuracy: {class_wise_accuracy}')
    print(f'accuracy for all classifications of the model: {all_classes_accuracy}')
    print(f'class wise average precision score: {average_precision}')
    print(f'average precision score (mean over all classes): {average_precision}')

    class_wise_average_precision = "missing"
    return predictions, softmax_scores, class_wise_accuracy, class_wise_average_precision


def evaluate_acc(model, dataloader, criterion, device):
    model.eval()

    losses = []
    curcount = 0  # keep track of number of samples evaluated
    accuracy = 0 # The accuracy variable is defined as a running average of the accuracy across all batches seen so far,
    # and is updated using an exponential moving average formula. This allows us to weight the accuracy of each batch
    # based on the number of samples in that batch, and maintain a smooth estimate of the overall accuracy

    with torch.no_grad():                      # disables gradient calculation, gradients not tracked by PyTorch
        for ctr, data in enumerate(dataloader):
            inputs = data['image'].to(device)
            outputs = model(inputs)
            cpu_output = outputs.to('cpu')
            labels = data['label']

            loss = criterion(cpu_output, labels)   # compute total loss for the batch
            losses.append(loss.item())  # loss.item() extracts  scalar value of the loss tensor (average loss for current batch)

            _, preds = torch.max(cpu_output, 1)   # get predicted class for each sample in the batch
            labels = labels.float()


            corrects = torch.sum(preds == labels.data) / float(labels.shape[0])
            accuracy = accuracy * (curcount / float(curcount + labels.shape[0])) + \
                       corrects.float() * (labels.shape[0] / float(curcount + labels.shape[0]))
            curcount += labels.shape[0]

    return accuracy.item(), np.mean(losses)  # this mean will be wrong if batches has different size


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
    # plt.show()

def hook(best_model):

    for nam, mod in best_model.named_modules():
        print(nam)

if __name__ == '__main__':
    data_dir = "/Users/anders/Documents/IN4310/mandatory/mandatory1_data/"
    # data_dir = "../mandatory1_data/"
    config =    {
                'batch_size': 32,
                'use_cuda': False,  # True=use Nvidia GPU | False use CPU
                'log_interval': 5,  # How often to display (batch) loss during training
                'num_epochs': 2,  # Number of epochs
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

    '''
    # run training
    losses_epochs_train1, losses_epochs_val, measure_epochs_train, measure_epochs_val = run_training()

    # Print result of training and evaluation
    print(f'losses_epochs_train: {losses_epochs_train1}')
    print(f'losses_epochs_val: {losses_epochs_val}')
    print(f'measure_epochs_train: {measure_epochs_train}')
    print(f'measure_epochs_val: {measure_epochs_val}')

    plot_loss(losses_epochs_train1, losses_epochs_val)
    '''
    # load best model and evaluate it with the test dataset
    best_model = torch.load('best_model.pt')
    hook(best_model)
    predictions, softmax_scores, class_wise_accuracy, average_precision = run_model_test(best_model)



