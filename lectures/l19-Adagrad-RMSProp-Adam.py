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
    get_mnist_data_loaders,
    NN_FC_CrossEntropy,
    compute_validation_accuracy_multi,
)

from fastprogress.fastprogress import master_bar, progress_bar

import torch

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")

# %%
# Show bias correction
# Plot effective learning rate


def train_one_epoch_adagrad(
    dataloader, model, criterion, learning_rate, decay_rate, device, mb
):

    if not hasattr(model, "sum_square_grads"):
        model.sum_square_grads = [torch.zeros_like(p) for p in model.parameters()]
        model.ms = [torch.zeros_like(p) for p in model.parameters()]
        model.vs = [torch.zeros_like(p) for p in model.parameters()]
        model.t = 1

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
            # for param, G in zip(model.parameters(), model.sum_square_grads):

            # Adagrad
            # G.set_(G + param.grad * param.grad)
            # param -= learning_rate * param.grad / (torch.sqrt(G) + 1e-8)

            # RMSProp
            # G.set_(decay_rate * G + (1 - decay_rate) * param.grad * param.grad)
            # param -= learning_rate * param.grad / (torch.sqrt(G) + 1e-8)

            for param, m, v in zip(model.parameters(), model.ms, model.vs):
                # Adam
                beta1, beta2 = betas
                m.set_(beta1 * m + (1 - beta1) * param.grad)
                v.set_(beta2 * v + (1 - beta2) * param.grad * param.grad)

                mt = m / (1 - beta1 ** model.t)
                vt = v / (1 - beta2 ** model.t)

                param -= learning_rate * mt / (torch.sqrt(vt) + 1e-8)

                model.t += 1


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
lr_decay = 0.95  # Adagrad
alpha = 0.99  # RMSProp
betas = (0.9, 0.999)  # Adam

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
# Training loop
mb = master_bar(range(num_epochs))
compute_validation_accuracy_multi(valid_loader, model, criterion, device, mb, 0)
for epoch in mb:
    train_one_epoch_adagrad(
        train_loader,
        model,
        criterion,
        learning_rate,
        device,
        mb,
    )
    loss, accuracy = compute_validation_accuracy_multi(
        valid_loader, model, criterion, device, mb, epoch + 1
    )
