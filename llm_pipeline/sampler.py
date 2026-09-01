"""k-shot support-set sampling for few-shot text classification.

Standard stratified few-shot sampling convention: draw exactly `k` examples
per class, without replacement, for use as a small labeled support set (e.g.
to `fit` a `KLPCDA` model on frozen LLM sentence embeddings before
`transform`-ing a much larger unlabeled/test set).
"""

from __future__ import annotations

import numpy as np


def k_shot_indices(y: np.ndarray, k: int, seed: int, classes: np.ndarray | None = None) -> np.ndarray:
    """Return indices of a stratified k-shot support set: exactly `k`
    examples per class, sampled without replacement, order shuffled."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if classes is None:
        classes = np.unique(y)

    chosen = []
    for c in classes:
        class_idx = np.where(y == c)[0]
        if class_idx.shape[0] < k:
            raise ValueError(f"class {c!r} has only {class_idx.shape[0]} examples, need k={k}")
        chosen.append(rng.choice(class_idx, size=k, replace=False))

    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx
