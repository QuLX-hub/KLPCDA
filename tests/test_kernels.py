import numpy as np

from klpcda.kernels import center_test_kernel, center_train_kernel, pairwise_kernel


def test_linear_kernel_matches_gram_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    K = pairwise_kernel(X, X, kernel="linear")
    np.testing.assert_allclose(K, X @ X.T)


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    K = pairwise_kernel(X, X, kernel="rbf")
    np.testing.assert_allclose(np.diag(K), np.ones(6), atol=1e-10)


def test_centered_kernel_rows_and_columns_sum_to_zero():
    # H @ 1 = 0 for the centering operator H = I - (1/n) ones(n,n); since
    # K_tilde = H K H, every row/column of K_tilde must sum to ~0.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 5))
    K = pairwise_kernel(X, X, kernel="poly", degree=2)
    K_tilde = center_train_kernel(K)
    np.testing.assert_allclose(K_tilde.sum(axis=0), np.zeros(8), atol=1e-8)
    np.testing.assert_allclose(K_tilde.sum(axis=1), np.zeros(8), atol=1e-8)


def test_out_of_sample_centering_matches_manual_formula():
    # Direct check against the textbook out-of-sample centering formula
    # (Schölkopf et al., 1998), computed entry-by-entry from RAW kernels.
    # `center_test_kernel` must use the raw training kernel for this, not an
    # already-centered one, which would silently drop two of the three
    # correction terms.
    rng = np.random.default_rng(2)
    X_train = rng.normal(loc=1.5, size=(6, 3))  # deliberately non-zero-mean
    X_test = rng.normal(loc=1.5, size=(2, 3))

    K_train_raw = pairwise_kernel(X_train, X_train, kernel="poly", degree=2)
    K_test_raw = pairwise_kernel(X_test, X_train, kernel="poly", degree=2)
    got = center_test_kernel(K_test_raw, K_train_raw)

    n = X_train.shape[0]
    grand_mean = K_train_raw.mean()
    train_col_mean = K_train_raw.mean(axis=0)  # (1/n) sum_l k(x_l, x_j), per j

    expected = np.zeros_like(K_test_raw)
    for i in range(X_test.shape[0]):
        test_row_mean = K_test_raw[i].mean()  # (1/n) sum_l k(x_test_i, x_l)
        for j in range(n):
            expected[i, j] = K_test_raw[i, j] - test_row_mean - train_col_mean[j] + grand_mean

    np.testing.assert_allclose(got, expected, atol=1e-10)
