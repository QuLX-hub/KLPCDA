"""Between-class and within-class scatter operators B and W.

Follows Qu & Pei, "Kernelized Linear Principal Component Discriminant
Analysis", Neural Networks 198 (2026) 108539, Eqs. (11) and (13):

    B = (1/n) * blockdiag(1/n_1 * ones(n_1,n_1), ..., 1/n_L * ones(n_L,n_L))
        - (1/n^2) * ones(n, n)

    W = blockdiag(P_1 I_{n_1}, ..., P_L I_{n_L})
        - blockdiag(1/n * ones(n_1,n_1), ..., 1/n * ones(n_L,n_L))

with P_i = n_i / n. These act on the (centered) kernel matrix in place of
the feature-space scatter matrices S_b = X^T B X, S_w = X^T W X.
"""

from __future__ import annotations

import numpy as np


def class_scatter_operators(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    n = y.shape[0]
    classes = np.unique(y)

    B = np.zeros((n, n))
    W = np.zeros((n, n))

    for c in classes:
        idx = np.where(y == c)[0]
        n_i = idx.shape[0]
        if n_i < 1:
            continue
        block = np.ix_(idx, idx)
        B[block] += 1.0 / (n * n_i)
        p_i = n_i / n
        W[block] += p_i * np.eye(n_i) - (1.0 / n) * np.ones((n_i, n_i))

    B -= 1.0 / (n**2)
    return B, W
