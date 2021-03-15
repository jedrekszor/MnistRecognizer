import torch, torchvision
import copy
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import BATCH_SIZE, EPOCHS

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


def train(epochs, device, learning_rate=1e-3):
    best_model = None
    accuracies = []
    cnn = create_model().to(device)
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    max_accuracy = 0

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = ce(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, validation_loader))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving best model with accuracy: ", accuracy)
        print("Epoch: ", epoch + 1, " Accuracy: ", accuracy, "%")
    plt.plot(accuracies)
    return best_model


lenet = train(EPOCHS, device)
