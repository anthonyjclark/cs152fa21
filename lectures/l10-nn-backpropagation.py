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
# # Backpropagation

# %%
import torch

# %% [markdown]
# ## Create fake input and output

# %%
# Total number of training examples
N = 100

# Number of inputs and outputs (based on diagram)
nx = 3
ny = 2

# Random inputs and outputs (just for sake of computation)
X = torch.randn(N, nx)
Y = torch.randn(N, ny)


# %% [markdown]
# ## Create a simple model based on the diagram

# %%
def linear(A, W, b):
    return A @ W.T + b


def sigmoid(Z):
    return 1 / (1 + torch.exp(-Z))


# A two-layer network with 3 neurons in the only hidden layer
n0 = nx
n1 = 3
n2 = ny

# Layer 1 parameters
W1 = torch.randn(n1, n0)
b1 = torch.randn(n1)

# Layer 2 parameters
W2 = torch.randn(n2, n1)
b2 = torch.randn(n2)

# %% [markdown]
# ## Compute model output (forward propagation)

# %%
A0 = X

# Forward propagation
Z1 = linear(A0, W1, b1)
A1 = sigmoid(Z1)

Z2 = linear(A1, W2, b2)
A2 = sigmoid(Z2)

Yhat = A2

# %% [markdown]
# ## Backpropagation from scratch

# %%
# Compute loss as the mean-square-error
bce_loss = torch.mean(Y * torch.log(Yhat) + (1 - Y) * torch.log(1 - Yhat))
print("Loss:", bce_loss.item())

# Compute gradients for W^[2] and b^[2]
# dL_dY = Yhat - Y
dL_dY = (Y / Yhat - (1 - Y) / (1 - Yhat)) / 2
dY_dZ2 = Yhat * (1 - Yhat)

dZ2 = dL_dY * dY_dZ2

dW2 = (1 / N) * dZ2.T @ A1
db2 = dZ2.mean(dim=0)

# Compute gradients for W^[1] and b^[1]
dZ1 = dZ2 @ W2 * ((A1 * (1 - A1)))

dW1 = (1 / N) * dZ1.T @ X
db1 = dZ1.mean(dim=0)

# %% [markdown]
# ## Forward and backward propagation using Pytorch

# %%
# Let's copy the Ws and bs from above, but set them
# up for auto-differentiation

# Layer 1 parameters
W1Auto = W1.clone().detach().requires_grad_(True)
b1Auto = b1.clone().detach().requires_grad_(True)

# Layer 2 parameters
W2Auto = W2.clone().detach().requires_grad_(True)
b2Auto = b2.clone().detach().requires_grad_(True)

# Forward propagation (same as above, but using PyTorch functionality)
A0 = X
Z1 = torch.nn.functional.linear(A0, W1Auto, b1Auto)
A1 = torch.sigmoid(Z1)

Z2 = torch.nn.functional.linear(A1, W2Auto, b2Auto)
A2 = torch.sigmoid(Z2)
Yhat = A2

# Compute loss (same as above)
# bce_loss = torch.mean(Y * torch.log(Yhat) + (1 - Y) * torch.log(1 - Yhat))
bce_loss = -torch.nn.functional.binary_cross_entropy(Yhat, Y)
print("Loss:", bce_loss.item())

# Automatically compute derivatives
bce_loss.backward()

# %% [markdown]
# ## Compare computed gradients

# %%
# We shouldn't compare floating-point numbers using "==" since results
#  can differ based on the order of operations.
assert torch.allclose(dW2, W2Auto.grad)
assert torch.allclose(db2, b2Auto.grad)

assert torch.allclose(dW1, W1Auto.grad)
assert torch.allclose(db1, b1Auto.grad)

# %% [markdown]
# - Adding additional layers
# - Changing the loss function
# - Changing the activation function(s)

# %%
