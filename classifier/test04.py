import torch
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np

def evaluate_classification(true_labels, predicted_classes, softmax_scores):
    """
    Evaluates a classification model and computes several performance metrics.

    Args:
        true_labels: PyTorch tensor of true labels, shape (N,)
        predicted_classes: PyTorch tensor of predicted classes, shape (N,)
        softmax_scores: PyTorch tensor of softmax scores, shape (N, num_classes)

    Returns:
        A dictionary containing the following performance metrics:
        - class_accuracy: PyTorch tensor of class-wise accuracy, shape (num_classes,)
        - mean_class_accuracy: mean class-wise accuracy, scalar
        - class_average_precision: PyTorch tensor of class-wise average precision, shape (num_classes,)
        - mean_average_precision: mean average precision, scalar
    """
    # Convert PyTorch tensors to NumPy arrays
    true_labels = true_labels.cpu().numpy()
    predicted_classes = predicted_classes.cpu().numpy()
    softmax_scores = softmax_scores.cpu().numpy()

    # Compute class-wise accuracy
    class_accuracy = torch.tensor([accuracy_score(true_labels == i, predicted_classes == i) for i in range(softmax_scores.shape[1])], dtype=torch.float32)

    # Compute mean class-wise accuracy
    mean_class_accuracy = torch.mean(class_accuracy)

    # Convert true labels to one-hot encoding
    true_labels_onehot = np.eye(softmax_scores.shape[1])[true_labels]

    # Compute class-wise average precision
    class_average_precision = torch.tensor(average_precision_score(true_labels_onehot, softmax_scores, average=None), dtype=torch.float32)

    # Compute mean average precision
    mean_average_precision = torch.mean(class_average_precision)

    # Create a dictionary of performance metrics
    metrics = {
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': mean_class_accuracy,
        'class_average_precision': class_average_precision,
        'mean_average_precision': mean_average_precision
    }

    return metrics



def evaluate(model, data_loader, criterion, device):
    print('inside evaluate()')
    all_labels = []
    all_preds = []
    class_probs = [[] for _ in range(6)]
    losses = []
    class_scores = {'accuracy': [], 'average_precision': []}

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            outputs = outputs.to('cpu')
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

    print(class_scores)
    loss = np.mean(losses)
    # accuracy = metrics.accuracy_score(all_labels, all_preds)
    # average_precision = metrics.average_precision_score(all_labels, probs[:,1])
    mean_accuracy = np.mean(class_scores['accuracy'])
    mean_average_precision = np.mean(class_scores['average_precision'])

    return loss, class_scores, mean_accuracy, mean_average_precision