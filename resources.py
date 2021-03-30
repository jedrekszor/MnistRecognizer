import torch
from torch import nn
import torchvision
import torchvision.models as models


T = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.expand(3, -1, -1))
])

def create_squeezenet():
    model = models.squeezenet1_1(pretrained=True)

    model.classifier[1].out_channels = 10

    for p in model.classifier.parameters():
        p.requires_grad = False

    return model

def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Flatten(),
        nn.Linear(400, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax(dim=1)
    )


def validate(model, data):  # accuracy
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += float(x.size(0))
        correct += float(torch.sum(pred == labels))
    return correct * 100 / total


def mse(model, data):  # mean squared error
    results = []
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        value, pred = torch.max(model(images), 1)
        pred = pred.data.cpu()
        results.append(torch.sum(pred - labels) ** 2)
    return sum(results) / len(results)


def cel(model, data, ce):  # cross-entropy
    results = []

    with(torch.set_grad_enabled(False)):
        for i, (images, labels) in enumerate(data):
            images = images.cuda()
            labels = labels.cuda()
            pred = model(images)
            results.append(ce(pred, labels))
    return sum(results) / len(results)
