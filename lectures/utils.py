# -*- coding: utf-8 -*-
from contextlib import contextmanager
from math import floor, log10
from timeit import default_timer as timer
from typing import Tuple

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, ToTensor


def mnist_target_to_binary(target, B):
    """Convert target digits into a zeros (class A) and ones (class B)."""
    new_target = torch.zeros_like(target, dtype=torch.float)
    new_target[target == B] = 1.0
    return new_target.unsqueeze(-1)


def get_binary_mnist_dataloader(
    path: str, train: bool, A: int, B: int, bs: int
) -> Tuple[DataLoader, int]:
    """Helper method returning either a training or validation data loader."""

    # Convert images to normalized tensors with mean 0 and standard deviation 1
    image_xforms = Compose(
        [
            ConvertImageDtype(torch.float),#ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = MNIST(root=path, train=train, download=True)#, transform=image_xforms)

    # Grab indices for the two requested classes
    idx_classA = [i for i, t in enumerate(dataset.targets) if t == A]
    idx_classB = [i for i, t in enumerate(dataset.targets) if t == B]

    idxs = idx_classA + idx_classB
    size = len(idxs)

    if bs == 0:
        bs = size

    X = image_xforms(dataset.data[idxs])
    y = mnist_target_to_binary(dataset.targets[idxs], B)

    dataset = TensorDataset(X, y)

    dataset.classes = [A, B]
    dataset.data = X
    dataset.targets = y

    loader = DataLoader(dataset, batch_size=bs)

    return loader, size


def get_binary_mnist_dataloaders(
    path: str, A: int, B: int, bs_train: int = 0, bs_valid: int = 0
) -> Tuple[DataLoader, int, DataLoader, int]:
    """Return an MNIST dataloader for the two specified classes.

    Args:
        path (str): location in which to store downloaded data
        A (int): class A, a number in [0, 9] denoting a MNIST class/number
        B (int): class B, a number in [0, 9] denoting a MNIST class/number
        bs_train (int): batch size for training data loader
        bs_valid (int): batch size for validation data loader

    Returns:
        Tuple[DataLoader, int, DataLoader, int]: [description]
    """
    train_loader, train_size = get_binary_mnist_dataloader(path, True, A, B, bs_train)
    valid_loader, valid_size = get_binary_mnist_dataloader(path, False, A, B, bs_valid)
    return train_loader, train_size, valid_loader, valid_size


def get_binary_mnist_one_batch(path: str, A: int, B: int, flatten: bool):
    """Return entire MNIST training and validation partitions as single batches.

    Args:
        path (str): location in which to store downloaded data
        A (int): class A, a number in [0, 9] denoting a MNIST class/number
        B (int): class B, a number in [0, 9] denoting a MNIST class/number
    """
    train_dl, _, valid_dl, _ = get_binary_mnist_dataloaders(path, A, B)

    # Convenient method for turning a data loader into batch
    train_x, train_y = next(iter(train_dl))
    valid_x, valid_y = next(iter(valid_dl))

    if flatten:
        train_x = torch.flatten(train_x, start_dim=1)
        valid_x = torch.flatten(valid_x, start_dim=1)

    train_y = mnist_target_to_binary(train_y, B)
    valid_y = mnist_target_to_binary(valid_y, B)

    return train_x, train_y, valid_x, valid_y


def format_duration_with_prefix(duration, sig=2):
    # Round to significant digits
    duration = round(duration, -int(floor(log10(abs(duration)))) + (sig - 1))

    if duration < 1e-6:
        return f"{duration*1e9:.1f} ns"
    elif duration < 1e-3:
        return f"{duration*1e6:.1f} μs"
    elif duration < 1:
        return f"{duration*1e3:.1f} ms"
    else:
        return f"{duration:.1f} s"


@contextmanager
def stopwatch(label: str):
    start = timer()
    try:
        yield
    finally:
        print(f"{label}: {timer() - start:6.2f}s")
