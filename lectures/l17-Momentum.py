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
from utils import get_mnist_data_loaders, NN_FC_CrossEntropy

from fastprogress.fastprogress import master_bar, progress_bar

import torch

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")


# %%
def train_one_epoch(
    dataloader, model, criterion, learning_rate, weight_decay, momentum, device, mb
):
    
    if not hasattr(model, 'momentum_grads'):
        model.momentum_grads = [torch.zeros_like(p) for p in model.parameters()]
    
    model.train()

    num_batches = len(train_loader)
    dataiter = iter(dataloader)

    for batch in progress_bar(range(num_batches), parent=mb):

        X, Y = next(dataiter)
        X, Y = X.to(device), Y.to(device)

        output = model(X)

        loss = criterion(output, Y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param, grad in zip(model.parameters(), model.momentum_grads):
                grad.set_(momentum * grad + (1 - momentum) * param.grad)
                param -= learning_rate * grad + weight_decay * param


# %%
def compute_validation_accuracy(dataloader, model, criterion, device, mb, epoch):

    model.eval()

    N = len(dataloader.dataset)
    num_batches = len(dataloader)

    valid_loss, num_correct = 0, 0

    with torch.no_grad():

        for X, Y in dataloader:

            X, Y = X.to(device), Y.to(device)
            output = model(X)

            valid_loss += criterion(output, Y).item()
            num_correct += (output.argmax(1) == Y).type(torch.float).sum().item()

        valid_loss /= num_batches
        valid_accuracy = num_correct / N

    mb.write(
        f"{epoch:>3}: validation accuracy={(100*valid_accuracy):5.2f}% and loss={valid_loss:.3f}"
    )
    return valid_loss, valid_accuracy


# %%
# Configuration parameters
data_path = "../data"
seed = 0
torch.manual_seed(seed)

# Hyperparameters
num_epochs = 4
batch_size = 128
valid_batch_size = 0

learning_rate = 1e-2
weight_decay = 1e-3
momentum = 0.9

# Training device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# %%
# Get data loaders
train_loader, valid_loader = get_mnist_data_loaders(
    data_path, batch_size, valid_batch_size
)

# %%
# Create neural network model
nx = train_loader.dataset.data.shape[1:].numel()
ny = len(train_loader.dataset.classes)
layer_sizes = (nx, 20, 20, ny)

model = NN_FC_CrossEntropy(layer_sizes).to(device)

# Training utilities
criterion = torch.nn.CrossEntropyLoss()

# %%
# # ?torch.optim.SGD

# %%
# Training loop
mb = master_bar(range(num_epochs))
compute_validation_accuracy(valid_loader, model, criterion, device, mb, 0)
for epoch in mb:
    train_one_epoch(
        train_loader,
        model,
        criterion,
        learning_rate,
        weight_decay,
        momentum,
        device,
        mb,
    )
    loss, accuracy = compute_validation_accuracy(
        valid_loader, model, criterion, device, mb, epoch + 1
    )
