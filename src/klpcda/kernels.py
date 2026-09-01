"""Kernel matrix construction and centering.

Implements the centered kernel matrix of Qu & Pei (2026), Eq. (2):

    K_tilde = K - (1/n) 1 K - (1/n) K 1 + (1/n^2) 1 K 1

and the corresponding out-of-sample centering used to project new points
without recomputing the training eigenproblem (the same convention used by
sklearn's KernelPCA; the papers' Eq. (9) states the raw-kernel projection
formula but omits this test-time centering, which matters once the training
points are not already zero-mean in feature space).
"""

from __future__ import annotations

import numpy as np


def pairwise_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    degree: int = 3,
    coef0: float = 1.0,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute k(x_i, y_j) for x_i in X, y_j in Y."""
    if kernel == "linear":
        return X @ Y.T
    if kernel == "poly":
        return (X @ Y.T + coef0) ** degree
    if kernel == "rbf":
        g = gamma if gamma is not None else 1.0 / X.shape[1]
        x_sq = np.sum(X * X, axis=1)[:, None]
        y_sq = np.sum(Y * Y, axis=1)[None, :]
        sq_dists = x_sq + y_sq - 2.0 * (X @ Y.T)
        np.maximum(sq_dists, 0.0, out=sq_dists)
        return np.exp(-g * sq_dists)
    raise ValueError(f"unknown kernel {kernel!r}; expected 'linear', 'poly', or 'rbf'")


def center_train_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a training kernel matrix (Eq. 2)."""
    row_mean = K.mean(axis=0, keepdims=True)
    col_mean = K.mean(axis=1, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def center_test_kernel(K_test: np.ndarray, K_train: np.ndarray) -> np.ndarray:
    """Center a (n_test, n_train) kernel block using training statistics.

    K_test[i, j] = k(x_test_i, x_train_j); K_train is the raw (uncentered)
    training kernel matrix used to derive the centering constants.
    """
    train_row_mean = K_train.mean(axis=0, keepdims=True)
    train_total_mean = K_train.mean()
    test_row_mean = K_test.mean(axis=1, keepdims=True)
    return K_test - test_row_mean - train_row_mean + train_total_mean
