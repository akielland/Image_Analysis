import PIL.Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import itertools
import pickle

# import efficientnet_pytorch as efn

# Ignore the DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_data(data_dir, reduced_num_samples=False):
    data_dir = os.path.join(data_dir, 'mandatory1_data')
    # list of the subdirectories containing each class
    subdirs = [x for x in os.listdir(data_dir) if not x.startswith('.')]  # skip hidden files/directories

    file_paths = []   # list to store file paths of images
    labels = []       # labels corresponding to the above list of images

    # reduce the number of samples for testing of code
    num_samples_per_class = 1000//6

    # loop over the subdirectories with file paths and labels
    for i, subdir in enumerate(subdirs):
        class_dir = os.path.join(data_dir, subdir)
        image_files = os.listdir(class_dir)
        if reduced_num_samples:
            image_files = image_files[:num_samples_per_class]  # select the first num_samples_per_class images
        image_paths = [os.path.join(class_dir, f) for f in image_files]
        file_paths.extend(image_paths)  # collect all the paths to image of this class
        labels.extend([i]*len(image_files))  # label each image with its corresponding class index

    # split data into train, validation, and test sets which are stored as paths in lists
    if reduced_num_samples:
        train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=300, random_state=123, stratify=labels)
        train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=200, random_state=123, stratify=train_labels)
    else:
        train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=3000, random_state=123, stratify=labels)
        train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=2000, random_state=123, stratify=train_labels)

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
        if trvaltest == 3:
            self.imgfilenames = get_imagenet_data(root_dir, 200)
            self.labels = [0] * len(self.imgfilenames)
        if trvaltest == 4:
            # I don manage to set up this right...
            self.data = get_cifar10_data(root_dir, 200)
            self.labels = [0] * len(self.data)
            self.imgfilenames = [None] * len(self.data)

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        # image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
        if hasattr(self, 'data'):
            # in case an image and not a path; but I didnt mange to use the images in my feature_map_statistics() function later
            image = PIL.Image.fromarray(self.data[idx]).convert('RGB')
        else:
            image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Resize(size=224)(image)
            image = transforms.CenterCrop(224)(image)
            image = transforms.ToTensor()(image)

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}
        return sample  # out: directory with images and labels


def train_epoch(model, train_loader, criterion, device, optimizer):
    model.train()   # set the model to training mode

    losses = list()
    i = 1
    for batch_idx, data in enumerate(train_loader):
        inputs = data['image']
        inputs = inputs.to(device)
        labels = data['label']
        labels = labels.to(device)

        # Forward
        output = model(inputs)  # output tensor = prediction scores (logits) for each class in the output space
        loss = criterion(output, labels)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % config['log_interval'] == 0:
            print('current mean of losses ', np.mean(losses), i)
            i += 1

    return np.mean(losses)


def train_model(train_loader, val_loader,  model,  criterion, optimizer, device):
    # loop over number of epochs and train the model
    # evaluate the model after an epoch of training on both training and evaluation data: mean loss; accuracy; average precision
    # save model if mean over all classes is improved for accuracy
    losses_epochs_train = []
    losses_epochs_val = []
    accuracy_epochs_train = []
    accuracy_epochs_val = []
    average_precision_train = []
    average_precision_val = []

    best_measure_val = 0
    best_epoch = -1

    for epoch in range(config['num_epochs']):
        print('Epoch {}/{}'.format(epoch+1, config['num_epochs']))

        model.train()
        losses = train_epoch(model, train_loader, criterion, device, optimizer)
        losses_epochs_train.append(losses)

        model.eval()
        # calculate ... fro training set
        loss, class_scores, mean_accuracy, mean_average_precision  = evaluate(model, train_loader, criterion, device)
        accuracy_epochs_train.append(class_scores['accuracy'])  # append the accuracy list from class_scores
        average_precision_train.append(class_scores['average_precision'])

        # calculate ... for validation set
        loss, class_scores, mean_accuracy, mean_average_precision  = evaluate(model, val_loader, criterion, device)
        accuracy_epochs_val.append(class_scores['accuracy'])
        average_precision_val.append(['average_precision'])
        losses_epochs_val.append(loss)

        # save new model if mean over all classes for accuracy is improved
        measure_val = np.mean(class_scores['accuracy'])
        if measure_val > best_measure_val:
            best_model = model
            best_measure_val = measure_val
            best_epoch = epoch
            print('current best mean accuracy at validation set is', best_measure_val, 'achieved in  epoch ', best_epoch+1)

    return best_model, losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val, average_precision_train, average_precision_val


def run_training():
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    torch.manual_seed(123)

    # prepare data
    train_dataset = DatasetSixClasses(data_dir, trvaltest=0, transform=transform)
    val_dataset = DatasetSixClasses(data_dir, trvaltest=1, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)  # ??
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    learning_rates = config['learning_rate']
    best_hyperparameter = None

    for lr in learning_rates:
        best_measure_val_lr = 0
        # model: load and updating the output layer with the respective number of classes (6)
        num_classes = 6
        model = models.resnet18(pretrained=True)  #  load the pre-trained Resnet18 model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        torch.manual_seed(123)
        model.fc.reset_parameters()  # initialize weights and biases of the new fully connected last layer with random values
        model.to(device)
        print(model.conv1.weight.device)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        best_model, losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val, average_precision_train, average_precision_val = \
            train_model(train_loader, val_loader, model, criterion, optimizer, device)

        best_measure_val = np.mean(accuracy_epochs_val)

        if best_hyperparameter is None:
            best_hyperparameter = lr
            model_chosen = best_model
            best_measure_val_lr = best_measure_val
        elif best_measure_val > best_measure_val_lr:
            best_hyperparameter = lr
            model_chosen = best_model
            best_measure_val_lr = best_measure_val

            losses_epochs_train = losses_epochs_train
            losses_epochs_val = losses_epochs_val
            accuracy_epochs_train = accuracy_epochs_train
            accuracy_epochs_val = accuracy_epochs_val

    print('best_hyperparameter:', best_hyperparameter)
    torch.save(model_chosen, 'best_model.pt')
    return losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val


def evaluate(model, data_loader, criterion, device):
    all_labels = []
    all_preds = []
    class_probs = [[] for _ in range(6)]  # to store softmax_scores for each class
    losses = []
    class_scores = {'accuracy': [], 'average_precision': []}

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = torch.softmax(outputs, dim=1) # calculate softmax probabilities for each prediction

            preds = torch.argmax(probs, dim=1) # get predicted class labels for this batch
            all_labels.extend(labels.cpu().numpy())  # iteratively fill the list batch-wise with all labels in the end
            all_preds.extend(preds.cpu().numpy())    # same for preds

            # collect class probabilities from each batch
            for i in range(6):
                class_probs[i].extend(probs[:, i].cpu().numpy()) #  the inner lists contains the softmax probabilities for the corresponding class across all predictions in the dataset

        for i in range(6):
            class_labels = [1 if x == i else 0 for x in all_labels]  # recode to 1 and 0, length of the whole dataset
            class_preds = [1 if x == i else 0 for x in all_preds]

            class_scores['accuracy'].append(metrics.accuracy_score(class_labels, class_preds))
            # calculate average precision score based on softmax probabilities
            class_scores['average_precision'].append(metrics.average_precision_score(class_labels, class_probs[i]))

    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(class_scores['accuracy'])
    mean_average_precision = np.mean(class_scores['average_precision'])
    return mean_loss, class_scores, mean_accuracy, mean_average_precision


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


def plot_accuracy(accuracy_epochs_train, accuracy_epochs_val):
    num_epochs = len(accuracy_epochs_train)
    num_classes = len(accuracy_epochs_train[0])

    # Create plot for train accuracy
    fig_train = plt.figure()
    for i in range(num_classes):
        plt.plot(range(num_epochs), [epoch_scores[i] for epoch_scores in accuracy_epochs_train], label=f'Train Class {i + 1}')
    plt.legend()
    fig_train.savefig('train_accuracy_vs_epoch.png', dpi=300, bbox_inches='tight')

    # Create plot for validation accuracy
    fig_val = plt.figure()
    for i in range(num_classes):
        plt.plot(range(num_epochs), [epoch_scores[i] for epoch_scores in accuracy_epochs_val], label=f'Validation Class {i + 1}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    fig_val.savefig('val_accuracy_vs_epoch.png', dpi=300, bbox_inches='tight')

################################################
## Here starts the evaluation on the test set ##
################################################
def run_model_test(model):
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    model.to(device)
    model.eval()

    # prepare dataset with test data
    test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    predictions_tensor = torch.empty(len(test_loader.dataset), dtype=torch.long) # empty tensor: predictions
    softmax_scores_tensor = torch.empty(len(test_loader.dataset), model.fc.out_features)  # empty tensor: softmax scores

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
            probabilities = torch.softmax(outputs, dim=1)  # softmax scores of the predictions (kind of probabilities for sample to each class)
            predicted_classes = torch.argmax(probabilities, dim=1)  # the predicted class of the samples of the batch

            # save predicted class label and softmax scores to the tensors
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            softmax_scores_tensor[start_idx:end_idx] = probabilities
            predictions_tensor[start_idx:end_idx] = predicted_classes

            # save true and predicted labels for computing accuracy and precision (AP)
            true_labels.extend(labels.cpu().numpy()) # Convert pytorch tensors to numpy arrays. need to be at the cpu since using numpy
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


def get_top_and_bottom_images(class_idx, num_img):
    softmax_scores = torch.load('softmax_scores.pt')  # Load softmax scores
    class_scores = softmax_scores[:, class_idx]  # Extract scores for the given class

    # Sort softmax scores in descending order
    sorted_indices = torch.argsort(class_scores, descending=True)
    # Get top and bottom ranked indices
    top_indices = sorted_indices[:num_img]
    bottom_indices = sorted_indices[-num_img:]

    # Get top and bottom ranked images by soft max score
    test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
    top_images = [test_dataset[i]['image'] for i in top_indices]
    bottom_images = [test_dataset[i]['image'] for i in bottom_indices]
    return top_images, bottom_images

def figure_top_bottom_images(class_idx, num_img = 10):
    top_images, bottom_images = get_top_and_bottom_images(class_idx, num_img)
    # Create figure with 2 rows and num_img columns
    fig, ax = plt.subplots(nrows=2, ncols=num_img, figsize=(15, 4))

    # plot top images
    for i, image in enumerate(top_images):
        # image = image / torch.max(image)  # normalize pixel values to be between 0 and 1; needed when input image has pixel values outside the valid range for visualization with imshow
        image = torch.clamp(image, 0, 1)  # make sure pixel values are within the valid range
        ax[0, i].imshow(image.permute(1, 2, 0), vmin=0, vmax=1) # # set vmin and vmax to the valid range
        ax[0, i].axis('off')

    # plot bottom images
    for i, image in enumerate(bottom_images):
        image = torch.clamp(image, 0, 1)  # make sure pixel values are within the valid range
        ax[1, i].imshow(image.permute(1, 2, 0), vmin=0, vmax=1)
        ax[1, i].axis('off')

    fig.savefig(f't_b_images_class{class_idx}.png', dpi=300, bbox_inches='tight')  # string concatenation with image label index


def extract_module_names(model):
    for module_name, mod in model.named_modules():
        print(module_name)

def getActivation(name, activation):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def feature_map_statistics(model, test_dataset):
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    model.eval()
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)

    # Create a new DataLoader with the first 200 samples
    sample_size = 200
    test_loader_200 = DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, sampler=itertools.islice(test_loader.sampler, sample_size))

    activation = {} # dict to store the activations for each feature map
    # register forward hooks on the layers of choice
    h1 = model.layer1[0].conv1.register_forward_hook(getActivation('L0', activation))
    h2 = model.layer1[1].conv1.register_forward_hook(getActivation('L1', activation))
    h3 = model.layer2[0].conv1.register_forward_hook(getActivation('L2', activation))
    h4 = model.layer2[1].conv1.register_forward_hook(getActivation('L3', activation))
    h5 = model.layer3[0].conv1.register_forward_hook(getActivation('L4', activation))
    # go through all the batches in the dataset and calculate percentage
    percentage = {'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0}  # dict to store the percentage for each feature map
    # mean_feature_map = {'L0': 0, 'L1': 0, 'L5': 0, 'L8': 0, 'L14': 0}  # dict to store the mean feature map of each channel within each layer

    mean_feature_map = {'L0': torch.zeros((sample_size, model.layer1[0].conv1.out_channels)),
                        'L1': torch.zeros((sample_size, model.layer1[1].conv1.out_channels)),
                        'L2': torch.zeros((sample_size, model.layer2[0].conv1.out_channels)),
                        'L3': torch.zeros((sample_size, model.layer2[1].conv1.out_channels)),
                        'L4': torch.zeros((sample_size, model.layer3[0].conv1.out_channels))}

    print(mean_feature_map['L0'].size())

    empirical_covariance = {'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0}  # dict to store the empirical covariance matrix over the channels
    n = 0
    for batch_idx, data in enumerate(test_loader_200):
        # if isinstance(data, (list, tuple)) and len(data) >= 2:
        #    images = data[0]
        # elif isinstance(data, dict):
        images = data['image']
        inputs = images.to(device)

        batch_size = inputs.size(0)
        # forward pass - extract the features maps via the hooks: stored in activation dictionary
        outputs = model(inputs)

        # Loop over features maps and calculate the percentage of non-positive values
        for layer in percentage.keys():
            feature_map = activation[layer]  # tensor with activations values for feature map of a specific layer (batch_size, channel, height, width)
            percent_non_positive = torch.mean((feature_map <= 0).float())
            percentage[layer] = (percentage[layer] * n + percent_non_positive * batch_size) / (n + batch_size)  # calculate new mean

            # compute mean feature map over the spatial dimensions (no called mean feature map)
            # mean_feature_map[layer][batch_idx * batch_size:batch_idx * batch_size + batch_size, :] = torch.mean(feature_map.view(batch_size, feature_map.size(1), -1), dim=2)
            mean_feature_map[layer][n:n+batch_size,:] = torch.mean(feature_map.view(batch_size, feature_map.size(1), -1), dim=2)
            # -1 tell PyTorch to calculate appropriate size for the last dimension to match the total number of elements, i.e. height*width

        n += batch_size  # update the number of samples seen so far

    del activation # delete the feature maps to free up memory

    # compute empirical covariance matrix over the channels
    for layer in mean_feature_map.keys():
        # compute mean over all the samples for each channel
        mean_per_channel = torch.mean(mean_feature_map[layer], dim=0)
        # subtract the mean feature map from the original feature map
        center_mean_feature_map = mean_feature_map[layer] - mean_per_channel
        empirical_covariance[layer] = torch.matmul(center_mean_feature_map.transpose(0,1), center_mean_feature_map) / n

    h1.remove(); h2.remove(); h3.remove(); h4.remove(); h5.remove()  # detach the hooks
    print('Percentage of non-positive values:')
    print(percentage)
    # print('Mean feature maps over all spatial dimensions:')
    # print(mean_feature_map)
    # print('Empirical covariance matrix over the channels:')
    # print(empirical_covariance)
    # torch.save(empirical_covariance, 'empirical_covariance.pt')
    return empirical_covariance


def plot_top_k_eigenvalues(empirical_covariance, id, k=10):
    fig, ax = plt.subplots()
    for layer in empirical_covariance.keys():
        cov = empirical_covariance[layer]
        eigenvalues, _ = np.linalg.eigh(cov.cpu().numpy())
        # sort eigenvalues in descending order and select topp k of them
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        top_k_eigenvalues = sorted_eigenvalues[:k]

        ax.plot(top_k_eigenvalues, label=layer)

    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Top {k} eigenvalues for each layer')
    ax.legend()
    filename = f"top_k_eigenvalues_{id}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')


def no_fine_tuning():
    model = models.resnet18(pretrained=True)
    # Replace the last layer with a new fully connected layer for 6 classes and reset the parameters for this layer
    num_classes = 6
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # model.fc.reset_parameters()  # initialize weights and biases of the new fully connected last layer with random values

    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    model.to(device)
    return model

def get_imagenet_data(data_dir, num_images=200):
    # Get paths to image files
    images_dir = os.path.join(data_dir, 'imagenet_val')
    image_paths = []

    for i, filename in enumerate(os.listdir(images_dir)):
        if i >= num_images:
            break
        if filename.endswith('.JPEG'):
            image_path = os.path.join(images_dir, filename)
            image_paths.append(image_path)

    return image_paths


def get_cifar10_data(data_dir, num_images=200):
    test_batch_file = os.path.join(data_dir, 'cifar-10-batches-py/test_batch')

    with open(test_batch_file, 'rb') as f:     #  opens the test batch file in binary mode for reading.
        test_data = pickle.load(f, encoding='bytes')   # loads pickled data
    data = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # reshape the images to the right shape
    data_subset = data[:num_images]

    return data_subset

def compare_softmax_scores(tolerance=1e-2):
    saved_softmax_scores = torch.load('saved_softmax_scores.pt')
    test_softmax_scores = torch.load('softmax_scores.pt')
    is_equal = torch.allclose(test_softmax_scores, saved_softmax_scores, rtol=tolerance, atol=tolerance)
    if is_equal:
        print("The softmax scores are equal")
    else:
        print(f"The softmax scores are not equal using a tolerance of {tolerance}")


if __name__ == '__main__':
    # These are the two thing you need to set:
    # 1) path
    data_dir = "/itf-fi-ml/shared/courses/IN3310/"
    # data_dir = "/Users/anders/Documents/IN4310/mandatory/"
    # 2) train or evaluate or both
    train = False
    evalu = True

    config =    {
                'batch_size': 16,
                'num_workers': 0,
                'use_cuda': True,  # True=use Nvidia GPU | False use CPU
                'log_interval': 75,  # How often to display (batch) loss during training
                'num_epochs': 8,    # Number of epochs
                'learning_rate': [1e-2, 1e-3, 1e-4]
                }

    # define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split the dataset
    # Set boolean argument in split_data(reduced_num_samples=True/False) for using the whole data set or just a small sample
    train_files, val_files, test_files, train_labels, val_labels, test_labels = split_data(data_dir, reduced_num_samples=False)
    print("The number of samples in each data set")
    print(f'Train set: {len(train_files)} samples')
    print(f'Validation set: {len(val_files)} samples')
    print(f'Test set: {len(test_files)} samples')
    print("-" * 10)

    # run training and compute loss and accuracy (on training and validation set)
    if train:
        losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
        print("Result of training per epoch")
        print(f'losses_epochs_train: {losses_epochs_train}')
        print(f'losses_epochs_val: {losses_epochs_val}')
        print(f'accuracy_epochs_train: {accuracy_epochs_train}')
        print(f'accuracy_epochs_val: {accuracy_epochs_val}')

        # save some plots of epoch-wise development of model measurements
        plot_loss(losses_epochs_train, losses_epochs_val)
        plot_accuracy(accuracy_epochs_train, accuracy_epochs_val)


    ## Here starts evaluation of the model on the test data sets (and some other dataset)
    if evalu:
        print("-"*10)
        print('START evaluating the model:')
        print("Load the best model and evaluate (accuracy, average precision) at the test dataset")
        print("Also save two tensors: predictions.pt and softmax_scores.pt from the test dataset")
        best_model = torch.load('best_model.pt')
        predictions_tensor, softmax_scores_tensor, class_scores = run_model_test(best_model)

        # save figure with images of best and worst classifications to actually observe that the classifier is working
        # using the saved softmax score on the test set to pic images
        figure_top_bottom_images(class_idx=0)
        figure_top_bottom_images(class_idx=1)
        figure_top_bottom_images(class_idx=2)

       # best_model = torch.load('best_model.pt')
       # extract_module_names(best_model)  # these line was used only once to get the names of the layers

        print("-" * 10)
        print("Statistics on 5 feature maps from best model on the mandatory dataset")
        best_model = torch.load('best_model.pt')
        test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
        empirical_covariance = feature_map_statistics(best_model, test_dataset)
        plot_top_k_eigenvalues(empirical_covariance, id='mandaData_fine_tuned', k=10)
    
        print('-' * 10)
        print("Statistics on 5 feature maps from NON-fine-tuned model (just pretrained weights on the mandatory dataset) ")
        model_w_no_fine_tuning = no_fine_tuning() # make a resnet18 model with the pretrained weights
        test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
        empirical_covariance = feature_map_statistics(model_w_no_fine_tuning, test_dataset)
        plot_top_k_eigenvalues(empirical_covariance, id='mandaData_NON_tuned', k=10)

        print('Statistics on the ImageNet datasdet')
        model_w_no_fine_tuning = no_fine_tuning()  # make a resnet18 model with the pretrained weights
        ImageNet_dataset = DatasetSixClasses(data_dir, trvaltest=3, transform=transform)
        empirical_covariance = feature_map_statistics(model_w_no_fine_tuning, ImageNet_dataset)
        plot_top_k_eigenvalues(empirical_covariance, id='ImageNetData_NON_tuned', k=10)

        print('Statistics on the CIFAR-10 datasdet')
        print('Dont work!!!')
        '''
        model_w_no_fine_tuning = no_fine_tuning()  # make a resnet18 model with the pretrained weights
        CIFAR_dataset = DatasetSixClasses(data_dir, trvaltest=4, transform=transform)

        empirical_covariance = feature_map_statistics(model_w_no_fine_tuning, CIFAR_dataset)
        plot_top_k_eigenvalues(empirical_covariance, id='CifarData_NON_tuned', k=10)
        '''

        print('-'*10)
        print('compare my saved softmax score with on the fly softmax scorers:')
        compare_softmax_scores(tolerance=1e-1)



