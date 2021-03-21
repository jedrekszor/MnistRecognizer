import torch, torchvision
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
from config import BATCH_SIZE, PATH, MODELPATH
from resources import create_model, T
from sklearn.metrics import confusion_matrix

train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
validation_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device used: ", device)


def predict_dl(model, data):
    y_pred = []
    y_true = []
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        # labels = labels.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        for j, dupa in enumerate(images):
            if pred[j] != labels[j]:
            # if torch.sum(pred[j] == labels[j]) != 1:
                save_wrong((i+j), images[j], pred[j], labels[j])
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(labels.numpy()))
    return np.array(y_pred), np.array(y_true)


def save_wrong(id, image, pred, true):
    image = np.squeeze(image)
    image = image * 255.
    img = TF.to_pil_image(image)
    img.save(PATH + "/wrong/" + "{}_pred_{}_actual_{}.png".format(
        id, pred, true))


lenet = create_model().to(device)
lenet.load_state_dict(torch.load(MODELPATH))

y_pred, y_true = predict_dl(lenet, validation_loader)  # Confusion
matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))  # Matrix
print(matrix)
