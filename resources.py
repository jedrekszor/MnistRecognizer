import torch, torchvision
from torch import nn

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def create_model():
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)

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


def validate(model, data):  # accuracy
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct * 100 / total


def mse(model, data):  # mean squared error
    results = []
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        value, pred = torch.max(model(images), 1)
        pred = pred.data.cpu()
        results.append(torch.sum(pred - labels) ** 2)
    return sum(results) / len(results)


