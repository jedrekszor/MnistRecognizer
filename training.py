import torch, torchvision
import copy
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import BATCH_SIZE, EPOCHS, PATH
from sklearn.metrics import confusion_matrix

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
validation_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

# plt.imshow(train_data[0][0][0], cmap='gray')
# plt.show()

# model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)

# device = 'cuda'
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device used: ", device)


def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )


def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        # images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct * 100 / total


def mse(model, data):
    results = []
    for i, (images, labels) in enumerate(data):
        # images = images.cuda()
        value, pred = torch.max(model(images), 1)
        pred = pred.data.cpu()
        results.append(torch.sum(pred - labels) ** 2)
    return sum(results) / len(results)

def cel(model, data, ce):
    results = []
    for i, (images, labels) in enumerate(data):
        # images = images.cuda()
        # value, pred = torch.max(model(images), 1)
        pred = model(images)
        results.append(ce(pred, labels))
        # pred = pred.data.cpu()
        # results.append(torch.sum(pred - labels) ** 2)
    return sum(results) / len(results)

def train(epochs, device, learning_rate=1e-3):
    best_model = None
    accuracies = []
    training_losses = []
    validation_losses = []
    cnn = create_model().to(device)
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    max_accuracy = 0

    for epoch in range(epochs):
        losses = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = ce(pred, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, validation_loader))
        training_loss = float(sum(losses)/len(losses))
        validation_loss = float(cel(cnn, validation_loader, ce))
        accuracies.append(accuracy)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving best model with accuracy: ", accuracy)
            torch.save(best_model.state_dict(), PATH)
        print("Epoch: ", epoch + 1, ", Accuracy: ", accuracy, "%", ", Trainig error: ", training_loss, ", Validation "
                                                                                                       "error: ",
              validation_loss)
        plt.plot(training_losses, label='training')
        plt.plot(validation_losses, label='validation')
        plt.legend()
        plt.show()
    return best_model


def predict_dl(model, data):
    y_pred = []
    y_true = []
    for i, (images, labels) in enumerate(data):
        # images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(labels.numpy()))
    return np.array(y_pred), np.array(y_true)


lenet = train(EPOCHS, device)
print("Training finished")

y_pred, y_true = predict_dl(lenet, validation_loader)
pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))
