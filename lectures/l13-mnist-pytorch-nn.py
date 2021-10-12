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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


# %%
def get_mnist_data_loaders(path, batch_size, valid_batch_size):

    # MNIST specific transforms
    mnist_xforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Training data loader
    train_dataset = MNIST(root=path, train=True, download=True, transform=mnist_xforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
def train_one_epoch(dataloader, model, loss_fn, optimizer):

    model.train()

    size = len(dataloader.dataset)

    for batch, (X, Y) in enumerate(dataloader):

        X, Y = X.to(device), Y.to(device)

        output = model(X)

        loss = loss_fn(output, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %%
def compute_validation_accuracy(dataloader, model, loss_fn):

    model.eval()

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    valid_loss, correct = 0, 0

    with torch.no_grad():

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

        valid_loss /= num_batches
        correct /= size

        print(
            f"Validation Metrics:\n\tAccuracy: {(100*correct):>0.1f}%\n\tAvg loss: {valid_loss:>8f}"
        )


# %%
# Configuration parameters
data_path = "../data"
seed = 0
log_interval = 1

torch.manual_seed(seed)

# Hyperparameters
batch_size = 64
valid_batch_size = 0
learning_rate = 1e-3
num_epochs = 5

# Training device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# %%
# Get data loaders
train_loader, valid_loader = get_mnist_data_loaders(
    data_path, batch_size, valid_batch_size
)
batch_X, batch_Y = next(iter(train_loader))

# %%
# Neural network model
nx = batch_X.shape[1:].numel()
ny = int(torch.unique(batch_Y).shape[0])
layer_sizes = (nx, 512, 50, ny)

model = NeuralNetwork(layer_sizes).to(device)
print(model)

# %%
# Training utilities
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_one_epoch(train_loader, model, loss_fn, optimizer)
    compute_validation_accuracy(valid_loader, model, loss_fn)
print("Done!")

# %%
torch.save(model.state_dict(), "l13-model.pth")
print("Saved PyTorch Model State to l13-model.pth")

# %%
model = NeuralNetwork(layer_sizes)
model.load_state_dict(torch.load("l13-model.pth"))

model.eval()

i = 0
x, y = valid_loader.dataset[i][0], valid_loader.dataset[i][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# %%
