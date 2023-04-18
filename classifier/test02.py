import torch
import numpy as np
from sklearn.metrics import average_precision_score, label_binarize

# Assume true_labels is a PyTorch tensor with multi-class labels
true_labels = torch.tensor([0, 1, 2, 1, 0, 2])

# Binarize the labels
num_classes = 3
true_labels_binarized = label_binarize(true_labels.cpu().numpy(), classes=np.arange(num_classes))

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



def train_epoch(model, train_loader, criterion, device, optimizer):
    model.train()   # set the model to training mode

    losses = []
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

    return np.mean(losses)


def train_model(train_loader, val_loader,  model,  criterion, optimizer, device, num_epochs):
    losses_epochs_train = []
    losses_epochs_val = []
    accuracy_epochs_train = []
    accuracy_epochs_val = []
    average_precision_train = []
    average_precision_val = []

    best_measure_val = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        model.train()
        losses = train_epoch(model, train_loader, criterion, device, optimizer)
        losses_epochs_train.append(losses)

        model.eval()
        accuracy, average_precision  = evaluate(model, train_loader_loader, criterion, device)
        accuracy_epochs_train.append(accuracy)
        average_precision_train.append(average_precision)
        accuracy, average_precision  = evaluate(model, val_loader_loader, criterion, device)
        accuracy_epochs_val.append(accuracy)
        average_precision_val.append(average_precision)

        measure_val = np.mean(accuracy_epochs_val)

        if measure_val > best_measure_val:
            best_model = model
            best_measure_val = measure_val
            best_epoch = epoch
            print('current best acc at validation set is', best_measure_val, 'achieved in  epoch ', best_epoch+1)


    return best_measure_val, best_model, losses_epochs_train, losses_epochs_val, /
    accuracy_epochs_train, accuracy_epochs_val, average_precision_train, average_precision_val


from sklearn import metrics

def evaluate(model, loader, criterion, device):
    all_labels = []
    all_preds = []
    losses = []

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

    loss = np.mean(losses)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    average_precision = metrics.average_precision_score(all_labels, all_preds)

    return loss, accuracy, average_precision
