#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Tuple


def initialize_parameters(
    n0: int, n1: int, n2: int, scale: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initialize parameters for a 2-layer neural network.

    Args:
        n0 (int): Number of input features (aka nx)
        n1 (int): Number of neurons in layer 1
        n2 (int): Number of output neurons
        scale (float): Scaling factor for parameters

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: weights and biases for 2 layers
            W1 : (n1, n0)
            b1 : (n1)
            W2 : (n2, n1)
            b2 : (n2)
    """
    # TODO: initialize and return W1, b1, W2, and b2


def forward_propagation(
    A0: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute the output of a 2-layer neural network.

    Args:
        A0 (Tensor): (N, n0) input matrix (aka X)
        W1 (Tensor): (n1, n0) layer 1 weight matrix
        b1 (Tensor): (n1) layer 1 bias matrix
        W2 (Tensor): (n2, n1) layer 2 weight matrix
        b2 (Tensor): (n2) layer 2 bias matrix

    Returns:
        Tuple[Tensor, Tensor]: outputs for layers 1 (N, n1) and 2 (N, n2)
    """
    # TODO: compute and return A1 and A2


def sigmoid_to_binary(A2: Tensor) -> Tensor:
    """Convert the output of a final layer sigmoids to zeros and ones.

    Args:
        A2 (Tensor): (N, n2) output of the network

    Returns:
        Tensor: binary predictions of a 2-layer neural network
    """
    # TODO: convert matrix to rounded zeros and ones


def backward_propagation(
    A0: Tensor, A1: Tensor, A2: Tensor, Y: Tensor, W2: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute gradients of a 2-layer neural network's parameters.

    Args:
        A0 (Tensor): (N, n0) input matrix (aka X)
        A1 (Tensor): (N, n1) output of layer 1 from forward propagation
        A2 (Tensor): (N, n2) output of layer 2 from forward propagation (aka Yhat)
        Y (Tensor): (N, n2) correct targets (aka targets)
        W2 (Tensor): (n2, n1) weight matrix

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: gradients for weights and biases
    """
    # TODO: compute and return gradients


def update_parameters(
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    dW1: Tensor,
    db1: Tensor,
    dW2: Tensor,
    db2: Tensor,
    lr: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Update parameters of a 2-layer neural network.

    Args:
        W1 (Tensor): (n1, n0) weight matrix
        b1 (Tensor): (n1) bias matrix)
        W2 (Tensor): (n2, n1) weight matrix)
        b2 (Tensor): (n2) bias matrix
        dW1 (Tensor): (n1, n0) gradient matrix
        db1 (Tensor): (n1) gradient matrix)
        dW2 (Tensor): (n2, n1) gradient matrix)
        db2 (Tensor): (n2) gradient matrix
        lr (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: updated network parameters
    """
    # TODO: Update and return parameters


def compute_loss(A2: Tensor, Y: Tensor) -> Tensor:
    """Compute mean loss using binary cross entropy loss.

    Args:
        A2 (Tensor): (N, n2) matrix of neural network output values (aka Yhat)
        Y (Tensor): (N, n2) correct targets (aka targets)

    Returns:
        Tensor: computed loss
    """
    # TODO: implement this function


def train_2layer(
    X: Tensor,
    Y: Tensor,
    num_hidden: int,
    param_scale: float,
    num_epochs: int,
    learning_rate: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """A function for performing batch gradient descent with a 2-layer network.

    Args:
        X (Tensor): (N, nx) matrix of input features
        Y (Tensor): (N, ny) matrix of correct targets (aka targets)
        num_hidden (int): number of neurons in layer 1
        param_scale (float): scaling factor for initializing parameters
        num_epochs (int): number of training passes through all data
        learning_rate (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: learned parameters of a 2-layer neural network
    """
    # TODO: implement this function
    # Steps:
    # 1. create and initialize parameters
    # 2. loop
    #   1. compute outputs with forward propagation
    #   2. compute loss (for analysis)
    #   3. compute gradients with backward propagation
    #   4. update parameters
    # 3. return final parameters
