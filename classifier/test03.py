import torch
import numpy as np
from sklearn.metrics import average_precision_score


from sklearn import metrics

def evaluate(model, loader, criterion, device):
    all_labels = []
    all_preds = []
    losses = []
    class_scores = {'accuracy': [], 'average_precision': []}

    with torch.no_grad():
        for data in loader:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        for i in range(model.num_classes):
            class_labels = [1 if x == i else 0 for x in all_labels]
            class_preds = [1 if x == i else 0 for x in all_preds]

            class_scores['accuracy'].append(metrics.accuracy_score(class_labels, class_preds))
            class_scores['average_precision'].append(metrics.average_precision_score(class_labels, class_preds))

    loss = np.mean(losses)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    average_precision = metrics.average_precision_score(all_labels, all_preds)
    mean_accuracy = np.mean(class_scores['accuracy'])
    mean_average_precision = np.mean(class_scores['average_precision'])

    return loss, accuracy, average_precision, class_scores, mean_accuracy, mean_average_precision


def compare_softmax_scores(model, test_softmax_scores, saved_softmax_scores):
    model.to(device)

    # Load the test dataset and create a dataloader
    test_dataset = DatasetSixClasses(data_dir, trvaltest=2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)

    model.eval()  # switch model to evaluation mode

    # Loop over test data and calculate the predicted scores
    predicted_scores = []
    for batch_idx, data in enumerate(test_loader):
        inputs = data['image'].to(device)
        outputs = model(inputs)
        predicted_scores.append(outputs.softmax(dim=1).cpu().detach())

    predicted_scores = torch.cat(predicted_scores, dim=0)

    # Compare the predicted scores to the saved scores
    np.testing.assert_array_almost_equal(test_softmax_scores.numpy(), predicted_scores.numpy(), decimal=6)
    np.testing.assert_array_almost_equal(saved_softmax_scores.numpy(), predicted_scores.numpy(), decimal=6)

    print('The softmax scores computed on the fly match the saved softmax scores.')



# Assume true_labels is a PyTorch tensor with multi-class labels
true_labels = torch.tensor([0, 1, 2, 1, 0, 2])

# Manually binarize the labels
num_classes = 3
true_labels_binarized = np.zeros((len(true_labels), num_classes))
true_labels_binarized[np.arange(len(true_labels)), true_labels] = 1
print(true_labels_binarized)

# Assume softmax_scores is a PyTorch tensor with softmax scores for all classes
softmax_scores = torch.tensor([
    [0.8, 0.1, 0.1],
    [0.1, 0.7, 0.2],
    [0.3, 0.3, 0.4],
    [0.2, 0.6, 0.2],
    [0.9, 0.1, 0.0],
    [0.4, 0.4, 0.2]
])

# Compute the average precision score for all classes
average_precision = average_precision_score(true_labels_binarized, softmax_scores.cpu().numpy(), average='macro')

# Print the result
print('Mean average precision score: {:.4f}'.format(average_precision))
