import numpy as np

from klpcda.scatter import class_scatter_operators


def test_B_W_hand_computed_two_class():
    # 2 classes, sizes n1=2, n2=1, n=3, matching Eqs. (11) and (13).
    y = np.array([0, 0, 1])
    B, W = class_scatter_operators(y)

    n, n1, n2 = 3, 2, 1
    expected_B = np.full((3, 3), -1.0 / n**2)
    expected_B[:2, :2] += 1.0 / (n * n1)
    expected_B[2, 2] += 1.0 / (n * n2)

    expected_W = np.zeros((3, 3))
    p1, p2 = n1 / n, n2 / n
    expected_W[:2, :2] = p1 * np.eye(2) - (1.0 / n) * np.ones((2, 2))
    expected_W[2, 2] = p2 * 1.0 - (1.0 / n) * 1.0

    np.testing.assert_allclose(B, expected_B)
    np.testing.assert_allclose(W, expected_W)


def test_W_cross_class_blocks_are_zero():
    y = np.array([0, 0, 1, 1, 1])
    _, W = class_scatter_operators(y)
    np.testing.assert_allclose(W[:2, 2:], 0.0)
    np.testing.assert_allclose(W[2:, :2], 0.0)


def test_B_full_matrix_gets_uniform_correction():
    # B's second term (-1/n^2) touches every entry, including cross-class
    # blocks -- unlike W's second term, which is block-diagonal only.
    y = np.array([0, 0, 1, 1, 1])
    B, _ = class_scatter_operators(y)
    n = 5
    assert B[0, 2] == -1.0 / n**2
