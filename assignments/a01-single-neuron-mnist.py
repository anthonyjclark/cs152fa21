#!/usr/bin/env python
# coding: utf-8

import torch
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST

from argparse import ArgumentParser
from time import time
from typing import Tuple


MNIST_DGX01_PATH = "/raid/cs152/data/"


def get_mnist_subset_loader(train: bool, c1: int, c2: int) -> Tuple[DataLoader, int]:
    """Return an MNIST dataloader for the two specified classes.

    Args:
        train (bool): Should this be a training set or validation set
        c1 (int): a number in [0, 9] denoting a MNIST class/number
        c2 (int): a number in [0, 9] denoting a MNIST class/number

    Returns:
        Tuple[DataLoader, int]: Return a dataloader and its size
    """

    # All inputs must be converted into torch tensors, and the normalization values
    # have been precomputed and provided below.
    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = MNIST(
        root=MNIST_DGX01_PATH, train=train, download=True, transform=mnist_transforms
    )

    # Grab indices for the two classes we care about
    idx_class1 = [i for i, t in enumerate(dataset.targets) if t == c1]
    idx_class2 = [i for i, t in enumerate(dataset.targets) if t == c2]

    idxs = idx_class1 + idx_class2
    size = len(idxs)

    loader = DataLoader(dataset, sampler=SubsetRandomSampler(idxs), batch_size=size)

    return loader, size


def get_mnist_data_binary(c1: int, c2: int) -> Tuple[DataLoader, int, DataLoader, int]:
    """Return data loaders for two classes from MNIST.

    Args:
        c1 (int): a number in [0, 9] denoting a MNIST class/number
        c2 (int): a number in [0, 9] denoting a MNIST class/number

    Returns:
        Tuple[DataLoader, int, DataLoader, int]: Return a training dataloader the
            training set size (and the same for the validation dataset)
    """

    train_loader, train_size = get_mnist_subset_loader(True, c1, c2)
    valid_loader, valid_size = get_mnist_subset_loader(False, c1, c2)

    return train_loader, train_size, valid_loader, valid_size


def linear(w: Tensor, b: Tensor, x: Tensor) -> Tensor:
    # TODO: implement the linear part of a neuron: z = w^T x + b
    return None


def sigmoid(z: Tensor) -> Tensor:
    # TODO: implement the sigmoid activation function: σ(z) = 1 / (1 + e^-z)
    return None


def binary_cross_entropy_loss(preds: Tensor, targs: Tensor) -> Tensor:
    # TODO: implement binary cross entropy loss: ylog(yhat) + (1-y)log(1-yhat)
    return None


def target_to_sigmoid(target: Tensor, c1: int) -> Tensor:
    """Convert the target classes into zeros and ones.

    Args:
        target (Tensor): A tensor of target values (integers)
        c1 (int): The classname that should be one

    Returns:
        Tensor: The correct label for a sigmoidal output
    """
    new_target = torch.zeros_like(target)
    new_target[target == c1] = 1
    return new_target


def compute_accuracy(
    x: Tensor, y: Tensor, m: int, nx: int, c1: int, w: Tensor, b: Tensor
) -> float:
    """Compute accuracy using the given parameters and data.

    Args:
        x (Tensor): The input tensor (MNIST images)
        y (Tensor): The output tensor (a column vector of zeros and ones)
        m (int): Size of the dataset
        nx (int): Number of input features (number of pixels)
        c1 (int): Class that should be compared with one
        w (Tensor): Weight parameters
        b (Tensor): Bias parameter

    Returns:
        float: Accuracy as a percentage
    """
    x = x.view(m, nx)
    y = target_to_sigmoid(y, c1)
    y_hat = sigmoid(linear(w, b, x.T))

    num_wrong = (torch.round(y_hat) - y).abs().sum()

    return (1 - num_wrong / m) * 100


def batch_gradient_descent_mnist(
    c1: int, c2: int, num_epochs: int, learning_rate: float, noprint: bool
) -> float:
    """The batch gradient descent algorithm implemented for MNIST and binary classification.

    Args:
        c1 (int): Number corresponding to the first class
        c2 (int): Number corresponding to the second class
        num_epochs (int): Number of training epochs
        learning_rate (float): Training learning rate
        noprint (bool): Do not print epoch information if True

    Returns:
        float: Final accuracy
    """

    train_loader, train_size, valid_loader, valid_size = get_mnist_data_binary(c1, c2)

    num_pixels = 28 * 28

    # Neuron parameters
    weights = torch.randn(num_pixels, 1) * 0.01
    bias = torch.zeros(1)

    # Validate with accuracy
    images, targets = next(iter(valid_loader))
    valid_accuracy = compute_accuracy(
        images, targets, valid_size, num_pixels, c1, weights, bias
    )

    print(f"Accuracy before training is {valid_accuracy:2.1f}%")

    for epoch in range(num_epochs):

        start = time()

        # Reset derivatives for this epoch
        weights_derivatives = torch.zeros_like(weights)
        bias_derivative = torch.zeros_like(bias)

        # Grab all images from the training loader and process
        images, targets = next(iter(train_loader))

        images = images.view(train_size, num_pixels)
        targets = target_to_sigmoid(targets, c1)

        # Compute predictions for all images (forward pass)
        predictions = sigmoid(linear(weights, bias, images.T))

        # Compute the loss for all images
        loss = binary_cross_entropy_loss(predictions, targets)
        cost = -(1 / train_size) * loss.sum()

        # Compute derivatives or cross entropy loss (backward pass)
        dZ = predictions - targets
        weights_derivatives = (1 / train_size) * (dZ @ images)
        bias_derivative = (1 / train_size) * dZ.sum()

        # Update parameters
        weights -= learning_rate * weights_derivatives.T
        bias -= learning_rate * bias_derivative

        # Validate with accuracy
        images, targets = next(iter(valid_loader))
        valid_accuracy = compute_accuracy(
            images, targets, valid_size, num_pixels, c1, weights, bias
        )

        if not noprint:
            text = [
                f"{epoch+1:>2}/{num_epochs}",
                f"Cost={cost:0.1f}",
                f"Accuracy={valid_accuracy:.2f}",
                f"Time={time()-start:0.1f}s",
            ]
            print(", ".join(text))

    return valid_accuracy


def main():
    arg_parser = ArgumentParser("Run batch gradient descent on MNIST for two classes.")
    arg_parser.add_argument("class1", type=int, help="First digit to classify.")
    arg_parser.add_argument("class2", type=int, help="Second digit to classify.")
    arg_parser.add_argument("epochs", type=int, help="Number of training epochs.")
    arg_parser.add_argument("lr", type=float, help="Training learning rate.")
    arg_parser.add_argument("--seed", type=int, help="Seed for random numbers.")
    arg_parser.add_argument(
        "--noprint", action="store_true", help="Disable periodic output."
    )

    args = arg_parser.parse_args()

    if args.seed != None:
        torch.manual_seed(args.seed)

    final_accuracy = batch_gradient_descent_mnist(
        args.class1, args.class2, args.epochs, args.lr, args.noprint
    )

    print(f"Accuracy after training is {final_accuracy:2.1f}%")


if __name__ == "__main__":
    main()
