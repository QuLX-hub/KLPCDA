import numpy as np
import pytest

from llm_pipeline.sampler import k_shot_indices


def test_k_shot_indices_exact_count_per_class():
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    idx = k_shot_indices(y, k=5, seed=0)
    assert idx.shape[0] == 15
    counts = {c: int((y[idx] == c).sum()) for c in (0, 1, 2)}
    assert counts == {0: 5, 1: 5, 2: 5}


def test_k_shot_indices_no_replacement():
    y = np.array([0] * 10 + [1] * 10)
    idx = k_shot_indices(y, k=5, seed=1)
    assert len(set(idx.tolist())) == len(idx)


def test_k_shot_indices_different_seeds_differ():
    y = np.array([0] * 50 + [1] * 50)
    idx_a = k_shot_indices(y, k=5, seed=1)
    idx_b = k_shot_indices(y, k=5, seed=2)
    assert set(idx_a.tolist()) != set(idx_b.tolist())


def test_k_shot_indices_raises_when_class_too_small():
    y = np.array([0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        k_shot_indices(y, k=3, seed=0)
