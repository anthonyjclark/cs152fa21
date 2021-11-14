# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from utils import (
    get_cifar10_data_loaders,
    compute_validation_accuracy_multi,
    NN_FC_CrossEntropy,
    train_one_epoch,
)

from fastprogress.fastprogress import master_bar

import torch
from torch import nn
from torchvision.utils import make_grid
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")

# %%
# Configuration parameters
data_path = "../data"
seed = 0
torch.manual_seed(seed)

# Hyperparameters
num_epochs = 4
batch_size = 128
valid_batch_size = 0
# The optimizer includes default hyperparameter values

# Training device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# %%
# Get data loaders
train_loader, valid_loader = get_cifar10_data_loaders(
    data_path, batch_size, valid_batch_size
)

train_loader.dataset.data.shape, valid_loader.dataset.data.shape, train_loader.dataset.classes

# %%
X, y = next(iter(train_loader))
X.shape, y.shape

# %%
# torch.std_mean(X, dim=(0, 2, 3))

# %%
n = 64

# Grab a bunch of images and change the range to [0, 1]
images = torch.tensor(train_loader.dataset.data[:n] / 255)

# Create a grid of the images (make_grid expects (BxCxHxW))
image_grid = make_grid(images.permute(0, 3, 1, 2))

_, axis = plt.subplots(figsize=(16, 16))
axis.imshow(image_grid.permute(1, 2, 0))

targets = train_loader.dataset.targets[:n]
classes = train_loader.dataset.classes

labels = [f"{classes[target]:>10}" for target in targets]

images_per_row = int(n ** 0.5)

for row in range(images_per_row):
    start_index = row * images_per_row
    print(" ".join(labels[start_index : start_index + images_per_row]))

# %%

# %%
# model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))

# nx = torch.prod(torch.tensor(train_loader.dataset.data.shape[1:]))
# ny = len(train_loader.dataset.classes)
# layer_sizes = (nx, 20, 20, ny)
# model = NN_FC_CrossEntropy(layer_sizes, torch.nn.Sigmoid).to(device)

model = resnet18()
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

model.to(device)

# %%
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# %%
mb = master_bar(range(num_epochs))
compute_validation_accuracy_multi(valid_loader, model, criterion, device, mb, 0)
for epoch in mb:
    train_one_epoch(train_loader, model, criterion, optimizer, device, mb)
    loss, accuracy = compute_validation_accuracy_multi(
        valid_loader, model, criterion, device, mb, epoch + 1
    )

# %%
correct = 0
total = 0

class_correct = [0] *len(classes)
class_total = [0] *len(classes)

model.eval()

with torch.no_grad():

    for images, targets in valid_loader:

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1, keepdim=True)
        
        comparisons = predictions.eq(targets.view_as(predictions))
        for comp, label in zip(comparisons, targets):
            class_correct[label] += comp.item()
            class_total[label] += 1

        total += targets.shape[0]
        correct += int(comparisons.double().sum().item())
        
accuracy = correct / total
print(f"Accuracy on validation set: {correct}/{total} = {accuracy*100:.2f}%")

for i, cls in enumerate(classes):
    ccorrect = class_correct[i]
    ctotal = class_total[i]
    caccuracy = ccorrect / ctotal
    print(f"  Accuracy on {cls:>10} class: {ccorrect}/{ctotal} = {caccuracy*100:.2f}%")

# %%
