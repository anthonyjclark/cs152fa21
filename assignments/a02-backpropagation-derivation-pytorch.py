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
def sigmoid(Z):
    return 1 / (1 + torch.exp(-Z))


# Number of layers and neurons per layer (based on diagram)
# Our class convention is to refer the input as layer "0"
neurons_per_layer = (nx, 3, 2, ny)
num_layers = len(neurons_per_layer) - 1

# Layer parameters (W and b)
Ws = {}
bs = {}

# Layers 1, 2, ..., L
for layer in range(1, num_layers + 1):
    nl = neurons_per_layer[layer]
    prev_nl = neurons_per_layer[layer - 1]

    Ws[layer] = torch.randn(nl, prev_nl)
    bs[layer] = torch.randn(nl)

# %% [markdown]
# ## Compute model output (forward propagation)

# %%
# Forward propagation (we need to save A matrices to compute gradients later)
As = [X]
for W, b in zip(Ws.values(), bs.values()):
    Z = As[-1] @ W.T + b
    print(f"{As[-1].shape} @ {W.T.shape} + {b.shape} = {Z.shape}")
    As.append(sigmoid(Z))

Yhat = As[-1]

print("Output shape (N, ny):", Yhat.shape)

# %% [markdown]
# ## Backpropagation from scratch

# %%
# Compute loss as the mean-square-error
mse_loss = torch.mean((Yhat - Y) ** 2)
print("Loss:", mse_loss.item())

# Compute gradients for W^[3] and b^[3]
dL_dY = Yhat - Y
dY_dZ3 = Yhat * (1 - Yhat)

dZ3 = dL_dY * dY_dZ3

dW3 = (1 / N) * dZ3.T @ As[2]
db3 = dZ3.mean(dim=0)

# Compute gradients for W^[2] and b^[2]
dZ2 = dZ3 @ Ws[3] * ((As[2] * (1 - As[2])))

dW2 = (1 / N) * dZ2.T @ As[1]
db2 = dZ2.mean(dim=0)

# Compute gradients for W^[1] and b^[1]
dZ1 = dZ2 @ Ws[2] * ((As[1] * (1 - As[1])))

dW1 = (1 / N) * dZ1.T @ X
db1 = dZ1.mean(dim=0)

# %% [markdown]
# ## Backpropagation using a loop

# %%
dWs = {}
dbs = {}

# Compute dZ for last layer
dL_dY = Yhat - Y
dY_dZ3 = Yhat * (1 - Yhat)

dZ = dL_dY * dY_dZ3

# Start at the last layer and move to the first
for layer in range(num_layers, 0, -1):
    dWs[layer] = (1 / N) * dZ.T @ As[layer - 1]
    dbs[layer] = dZ.mean(dim=0)

    if layer != 1:
        dZ = dZ @ Ws[layer] * ((As[layer - 1] * (1 - As[layer - 1])))

# %% [markdown]
# ## Forward and backward propagation using Pytorch

# %%
# Let's copy the Ws and bs from above, but set them
# up for auto-differentiation
WsAuto = {}
bsAuto = {}
for layer in range(1, num_layers + 1):
    WsAuto[layer] = Ws[layer].clone().detach().requires_grad_(True)
    bsAuto[layer] = bs[layer].clone().detach().requires_grad_(True)

# Forward propagation (same as above, but using PyTorch functionality)
prev_A = X
for W, b in zip(WsAuto.values(), bsAuto.values()):
    Z = torch.nn.functional.linear(prev_A, W, b)
    prev_A = torch.sigmoid(Z)
Yhat = prev_A

# Compute loss (same as above)
mse_loss = torch.mean((Yhat - Y) ** 2)
print("Loss:", mse_loss.item())

# Automatically compute derivatives
mse_loss.backward()

# %% [markdown]
# ## Compare computed gradients

# %%
# We shouldn't compare floating-point numbers using "==" since results
#  can differ based on the order of operations.
assert torch.allclose(dW3, WsAuto[3].grad)
assert torch.allclose(db3, bsAuto[3].grad)

assert torch.allclose(dW2, WsAuto[2].grad)
assert torch.allclose(db2, bsAuto[2].grad)

assert torch.allclose(dW1, WsAuto[1].grad)
assert torch.allclose(db1, bsAuto[1].grad)

for layer in range(1, num_layers + 1):
    assert torch.allclose(WsAuto[layer].grad, dWs[layer])
    assert torch.allclose(bsAuto[layer].grad, dbs[layer])

# %%
