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

# %% [markdown]
# # Plan
#
# 1. Read through code (~5 minutes)
# 2. Get into groups and discuss code (~2 minutes)
# 3. Ask questions on the sheet (~5 minutes)
# 4. Work on "Questions to answer" (~10 minutes)
# 5. Work on "Things to explore" (~10 minutes)
# 6. Work on the "Challenge" (~20 minutes)
# 7. Work on "What's next?"
#
# Getting started:
#
# - I recommend cloning this repository (or pulling changes if you already have it cloned)
# - Starting jupyter
# - Then duplicating this file so that you can alter it without confusing `git`
#
# Some tools to use:
#
# - You can create a cell above the current cell by typing "esc" then "a"
# - You can create a cell below the current cell by typing "esc" then "b"
# - You should copy code into newly created cells, alter it, print out the results, etc.
# - You can do this for single lines or you can copy, for example, the `for batch, (X, Y) in enumerate(dataloader):` loop out of `train_one_epoch` and make minor changes so that it works outside of the function
# - I will frequently put a break a the end of the for-loop so that it only iterates one time (so that I don't have to wait for every iteration)

# %%
from contextlib import contextmanager
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


# %%
@contextmanager
def stopwatch(label: str):
    start = timer()
    try:
        yield
    finally:
        print(f"{label}: {timer() - start:6.3f}s")


# %%
def get_mnist_data_loaders(path, batch_size, valid_batch_size):

    # MNIST specific transforms
    mnist_xforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Training data loader
    train_dataset = MNIST(root=path, train=True, download=True, transform=mnist_xforms)

    tbs = len(train_dataset) if batch_size == 0 else batch_size
    train_loader = DataLoader(train_dataset, batch_size=tbs, shuffle=True)

    # Validation data loader
    valid_dataset = MNIST(root=path, train=False, download=True, transform=mnist_xforms)

    vbs = len(valid_dataset) if valid_batch_size == 0 else valid_batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=vbs, shuffle=True)

    return train_loader, valid_loader


# %%
class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()

        first_layer = nn.Flatten()
        middle_layers = [
            nn.Sequential(nn.Linear(nlminus1, nl), nn.ReLU())
            for nl, nlminus1 in zip(layer_sizes[1:-1], layer_sizes)
        ]
        last_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        all_layers = [first_layer] + middle_layers + [last_layer]

        self.layers = nn.Sequential(*all_layers)

    def forward(self, X):
        return self.layers(X)


# %%
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):

    model.train()

    num_batches = len(train_loader)
    batches_to_print = [0, num_batches // 3, 2 * num_batches // 3, num_batches - 1]

    for batch, (X, Y) in enumerate(dataloader):

        X, Y = X.to(device), Y.to(device)

        output = model(X)

        loss = loss_fn(output, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch in batches_to_print:
            print(f"Batch {batch+1:>5} of {num_batches}: loss={loss.item():>6.3f}")


# %%
def compute_validation_accuracy(dataloader, model, loss_fn, device):

    model.eval()

    N = len(dataloader.dataset)
    num_batches = len(dataloader)

    valid_loss, num_correct = 0, 0

    with torch.no_grad():

        for X, Y in dataloader:

            X, Y = X.to(device), Y.to(device)
            output = model(X)

            valid_loss += loss_fn(output, Y).item()
            num_correct += (output.argmax(1) == Y).type(torch.float).sum().item()

        valid_loss /= num_batches
        valid_accuracy = num_correct / N

    print(f"Validation accuracy : {(100*valid_accuracy):>6.3f}%")
    print(f"Validation loss     : {valid_loss:>6.3f}")


# %% [markdown]
# # Configuration

# %%
# Configuration parameters
data_path = "../data"
seed = 0
torch.manual_seed(seed)

# Hyperparameters
batch_size = 1024
valid_batch_size = 0
learning_rate = 1e-2
num_epochs = 2

# Training device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# %% [markdown]
# # Data

# %%
# Get data loaders
train_loader, valid_loader = get_mnist_data_loaders(
    data_path, batch_size, valid_batch_size
)

# %% [markdown]
# # Model

# %%
# Create neural network model
nx = train_loader.dataset.data.shape[1:].numel()
ny = len(train_loader.dataset.classes)
layer_sizes = (nx, 512, 50, ny)

model = NeuralNetwork(layer_sizes).to(device)
print(model)

# %% [markdown]
# # Training Loop

# %%
# Training utilities
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%
with stopwatch(f"\nDone! Total time for {num_epochs} epochs"):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        with stopwatch("Epoch time          "):
            train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        compute_validation_accuracy(valid_loader, model, loss_fn, device)

# %% [markdown]
# # Questions to answer
#
# (Try to answer these in your group prior to running or altering any code.)
#
# - What is the shape of `output` in the function `train_one_epoch`?
# - What values would you expect to see in `output`?
# - What is the shape of `Y` in the function `train_one_epoch`?
# - Describe each part of `(output.argmax(1) == Y).type(torch.float).sum().item()`
# - What happens when you rerun the training cell for additional epoch (without rerunning any other cells)?
# - What happens to if force device to be `"cpu"`?

# %% [markdown]
# # Things to explore
#
# - change the hidden layer activation functions to sigmoid
# - change the hidden layer activation functions to [something else](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
# - change the optimizer from `SGD` to `Adam` and try to train the network again
#
# You can also try these if you feel like you have plenty of time. You can also choose to come back to them after working on the Challenge below
#
# - (optional) try adding a [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout) layer somewhere in your network
# - (optional) try switching the dataset to either [KMNIST](https://pytorch.org/vision/0.8/datasets.html#kmnist) or [FashionMNIST](https://pytorch.org/vision/0.8/datasets.html#fashion-mnist)

# %% [markdown]
# # Challenge
#
# Train a model and get the highest accuracy possible by adjusting hyperparameters and the model architecture (i.e., the number of layers, the number of neurons per layer, etc.).

# %% [markdown]
# # What's next?
#
# Move the inference cells below to a new file, and then try to make them work.

# %% [markdown]
# # Inference

# %%
model_filename = "l14-model.pth"
torch.save(model.state_dict(), model_filename)
print("Saved PyTorch Model State to", model_filename)

# %%
model = NeuralNetwork(layer_sizes)
model.load_state_dict(torch.load(model_filename))

model.eval()

# Index of example
i = 0

# Example input and output
x, y = valid_loader.dataset[i][0], valid_loader.dataset[i][1]

with torch.no_grad():
    output = model(x)
    prediction = output[0].argmax(0)
    print(f"Prediction : {prediction}")
    print(f"Target     : {y}")

# %%
