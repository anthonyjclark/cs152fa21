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
import torchvision

# %%
torch.rand(3)

# %%
torch.rand(3).shape

# %%
torch.rand(16, 17)

# %%
torch.rand(16, 17).shape

# %%
torch.rand(1, 2, 3, 4, 5)

# %%
torch.rand(1, 2, 3, 4, 5).shape

# %%
a = torch.rand(5, 12)
b = torch.rand(12, 16)

# %%
a.shape, b.shape

# %%
a * b # Element-wise multiplication

# %%
c = a @ b

# %%
c.shape

# %%
c = b @ a

# %%
c = b.T @ a.T

# %%
c.shape

# %%
# MNIST : hello world
# EMNIST : extended with letters in addition to digits
# KMNIST : Kuzushiji, Japanese characters
# QMNIST : newer MNIST with better source information

data_path = "../data/"

mnist_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root=data_path, train=True, download=True, transform=mnist_transforms
)

# %%
train_dataset

# %%
train_dataset.data[0]

# %%
train_loader = torch.utils.data.DataLoader(train_dataset)

# %%
for (image, label) in train_loader:
    print(image.shape, label.shape)
    break

# %%
