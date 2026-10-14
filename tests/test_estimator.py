import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from klpcda import KLPCDA, VARIANTS
from klpcda.estimator import _break_ties


def _make_sss_blobs(n_per_class=8, n_classes=3, n_features=50, seed=0, centers=None):
    """Small-sample-size synthetic data: n = n_per_class*n_classes << n_features,
    mimicking the n << d regime this repo targets.

    Pass an explicit `centers` array to draw train/test splits from the same
    underlying classes; otherwise centers are freshly sampled from `seed`.
    """
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = rng.normal(scale=4.0, size=(n_classes, n_features))
    else:
        n_features = centers.shape[1]
    X, y = [], []
    for c in range(centers.shape[0]):
        X.append(centers[c] + rng.normal(scale=0.5, size=(n_per_class, n_features)))
        y.append(np.full(n_per_class, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


def test_all_variants_fit_transform_shapes():
    X, y = _make_sss_blobs()
    for method in VARIANTS:
        model = KLPCDA(method=method, n_components=2, kernel="linear")
        out = model.fit_transform(X, y)
        assert out.shape == (X.shape[0], 2)
        X_new, _ = _make_sss_blobs(n_per_class=2, seed=99)
        assert model.transform(X_new).shape == (X_new.shape[0], 2)


def test_no4_requires_no_labels_others_do():
    X, y = _make_sss_blobs()
    KLPCDA(method="no4", n_components=2).fit(X)  # must not raise
    for method in set(VARIANTS) - {"no4"}:
        with pytest.raises(ValueError):
            KLPCDA(method=method, n_components=2).fit(X)


def test_unknown_method_rejected():
    X, _ = _make_sss_blobs()
    with pytest.raises(ValueError):
        KLPCDA(method="no8").fit(X)


def test_no4_linear_kernel_matches_pca_subspace():
    # Method No.4 (variance-only, linear kernel) is centered KPCA, which is
    # mathematically PCA. Compare the two projections via the canonical
    # correlation between them -- should be ~1 for every retained component
    # if the subspaces agree (they can differ by rotation/sign).
    X, _ = _make_sss_blobs(n_per_class=10, n_classes=2, n_features=15, seed=3)
    k = 4

    klpcda_proj = KLPCDA(method="no4", n_components=k, kernel="linear").fit_transform(X)
    pca_proj = PCA(n_components=k).fit_transform(X)

    # Canonical correlations via SVD of the cross-covariance of the two
    # (each already effectively zero-mean since both come from centered data).
    A = klpcda_proj - klpcda_proj.mean(axis=0)
    Bp = pca_proj - pca_proj.mean(axis=0)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    Bp /= np.linalg.norm(Bp, axis=0, keepdims=True)
    cross = A.T @ Bp
    canonical_corrs = np.linalg.svd(cross, compute_uv=False)

    np.testing.assert_allclose(canonical_corrs, np.ones(k), atol=1e-6)


def test_no7_selects_smallest_eigenvalues():
    # Eq. (31): argmin v^T S_w v -- the stored eigenvalues must be the
    # smallest ones of W @ K_tilde, not the largest.
    X, y = _make_sss_blobs()
    model = KLPCDA(method="no7", n_components=3, kernel="linear").fit(X, y)
    # cross-check against full spectrum computed independently
    from klpcda.kernels import center_train_kernel, pairwise_kernel
    from klpcda.scatter import class_scatter_operators

    K_tilde = center_train_kernel(pairwise_kernel(X, X, kernel="linear"))
    _, W = class_scatter_operators(y)
    full_spectrum = np.sort(np.linalg.eig(W @ K_tilde)[0].real)
    np.testing.assert_allclose(np.sort(model.eigenvalues_), full_spectrum[:3], atol=1e-6)


def test_fusion_coefficients_change_subspace_for_no2():
    # (alpha, beta) must change the *learned subspace* for Method No.2, not
    # just eigenvalue scale -- because they weight two independently-summed
    # terms.
    #
    # Built so C-only (pure variance/PCA) and S_b-only (pure between-class)
    # are forced to disagree: dims 1-4 carry large shared ("nuisance")
    # variance with no class-mean difference (so C-only picks them), while
    # dim 0 is the only one with a nonzero class-mean gap (so S_b-only must
    # pick it instead). A dataset without a deliberate nuisance direction
    # (e.g. well-separated blobs, where between-class spread already
    # dominates total variance) would let both settings agree by accident.
    rng = np.random.default_rng(7)
    n_per_class = 10
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    X = rng.normal(scale=5.0, size=(2 * n_per_class, 5))  # nuisance dims 0-4
    X[:, 0] = rng.normal(scale=0.01, size=2 * n_per_class)  # near-zero noise
    X[n_per_class:, 0] += 5.0  # ...except a large class-mean gap on dim 0

    proj_c_only = KLPCDA(method="no2", n_components=1, fusion=(1.0, 0.0)).fit_transform(X, y)
    proj_b_only = KLPCDA(method="no2", n_components=1, fusion=(0.0, 1.0)).fit_transform(X, y)

    def normalized(P):
        P = P - P.mean(axis=0)
        return P / np.linalg.norm(P, axis=0, keepdims=True)

    corr = float((normalized(proj_c_only).T @ normalized(proj_b_only)).item())
    assert abs(corr) < 0.5, "fusion=(1,0) and fusion=(0,1) should span different subspaces"


def test_no1_no3_no5_do_not_collapse_when_W_is_rank_deficient():
    # Regression test for a real bug found running this estimator on
    # high-dimensional embeddings with many classes and few examples each:
    # W has exact rank n - L (each class block loses one dimension -- see
    # class_scatter_operators), so W @ K_tilde has L singular values that
    # are mathematically zero but land anywhere from ~1e-15 to ~1e-21 in
    # floating point. An earlier version added a tiny additive ridge
    # (`W @ K_tilde + 1e-8*I`) before calling pinv, which is exactly the
    # wrong fix: it turns an exact rank deficiency pinv would otherwise
    # truncate correctly into a full-rank matrix inverted exactly, blowing
    # the near-zero directions up to ~1e8-1e15 and making them dominate the
    # top eigenvectors regardless of which term (B or C) they're paired
    # with -- so No.1, No.3, and No.5 all collapsed to the *same* learned
    # subspace. This reproduces that regime (many classes, few examples
    # each, so W's null space is a large fraction of the sample count) and
    # checks the three variants now disagree, as they mathematically should.
    n_classes, k, n_features = 20, 3, 60
    rng = np.random.default_rng(42)
    centers = rng.normal(scale=3.0, size=(n_classes, n_features))
    X, y = _make_sss_blobs(n_per_class=k, seed=42, centers=centers)  # n=60, same as n_features

    def top_component(method):
        return KLPCDA(method=method, n_components=1, kernel="linear").fit_transform(X, y)

    def normalized(P):
        P = P - P.mean(axis=0)
        return P / np.linalg.norm(P, axis=0, keepdims=True)

    proj = {m: normalized(top_component(m)) for m in ("no1", "no3", "no5")}
    corr_13 = float((proj["no1"].T @ proj["no3"]).item())
    corr_15 = float((proj["no1"].T @ proj["no5"]).item())
    corr_35 = float((proj["no3"].T @ proj["no5"]).item())

    # Before the fix these were identical (|corr| ~= 1.0 for every pair).
    assert not (abs(corr_13) > 0.999 and abs(corr_15) > 0.999 and abs(corr_35) > 0.999), (
        "No.1/No.3/No.5 collapsed to the same subspace -- the W@K_tilde "
        "rank-deficiency regularization regressed"
    )


def test_break_ties_ranks_by_tie_break_ascending():
    # Regression test for the tie-break fix: within a (near-)degenerate
    # eigenspace, `_break_ties` must replace numpy.linalg.eig's arbitrary
    # basis with one ranked by the tie-break criterion *ascending* --
    # empirically the direction that dramatically improved No.1/No.5's
    # real-data accuracy (see the estimator.py docstring for the reasoning).
    rng = np.random.default_rng(0)
    n, r = 6, 4
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    P = Q[:, :r] @ Q[:, :r].T  # rank-r orthogonal projector: exactly r repeated eigenvalues
    c = 3.0
    E = c * P

    A = rng.normal(size=(n, n))
    tie_break = A @ A.T  # symmetric PSD, generically distinct eigenvalues on Range(P)

    eigvals, eigvecs = np.linalg.eig(E)
    eigvals, eigvecs = _break_ties(eigvals.real, eigvecs.real, tie_break, rtol=1e-3)

    is_degenerate = np.isclose(eigvals, c, atol=1e-6)
    assert is_degenerate.sum() == r, "the r repeated eigenvalues should form one cluster"

    scores = np.array([v @ tie_break @ v for v in eigvecs[:, is_degenerate].T])
    assert np.all(np.diff(scores) >= -1e-9), "cluster should be ranked by tie_break, ascending"


def test_no1_no5_use_own_c_term_as_tie_break():
    # No.1 and No.5 both include a C (variance/K_tilde) term in their own
    # formula, so tie-breaking with it stays within the paper's own
    # definition (see estimator.py's `fit`). Verify the wiring: components_
    # actually changes when the tie-break is disabled versus enabled, on a
    # balanced small-sample draw where No.5's signal subspace is exactly
    # degenerate (many classes, few examples each -- same regime as
    # `test_no1_no3_no5_do_not_collapse_when_W_is_rank_deficient`).
    from klpcda import estimator as est_mod

    n_classes, k, n_features = 20, 3, 60
    rng = np.random.default_rng(42)
    centers = rng.normal(scale=3.0, size=(n_classes, n_features))
    X, y = _make_sss_blobs(n_per_class=k, seed=42, centers=centers)

    model_tied = KLPCDA(method="no5", n_components=1, kernel="linear").fit(X, y)

    # Reproduce the untied (pre-fix) behavior directly for comparison.
    original_eig_topk = est_mod._eig_topk

    def untied_eig_topk(E, k_, largest, energy_ref, **kwargs):
        kwargs.pop("tie_break", None)
        return original_eig_topk(E, k_, largest, energy_ref)

    est_mod._eig_topk = untied_eig_topk
    try:
        model_untied = KLPCDA(method="no5", n_components=1, kernel="linear").fit(X, y)
    finally:
        est_mod._eig_topk = original_eig_topk

    corr = float(
        np.abs(model_tied.components_[:, 0] @ model_untied.components_[:, 0])
        / (np.linalg.norm(model_tied.components_[:, 0]) * np.linalg.norm(model_untied.components_[:, 0]))
    )
    assert corr < 0.999, "tie-breaking should select a different direction than the untied basis"


@pytest.mark.parametrize("method", ["no1", "no2", "no4", "no5", "no6"])
def test_sss_classification_beats_chance(method):
    # SSS-robust variants (S_w-free: No.2/4/6) on a small, separable,
    # n << d synthetic task, evaluated with the paper's 1-NN convention.
    # Train and test must share the same class centers, so we sample the
    # centers once and draw both splits from them.
    n_classes, n_features = 3, 60
    centers = np.random.default_rng(10).normal(scale=4.0, size=(n_classes, n_features))
    X_train, y_train = _make_sss_blobs(n_per_class=5, seed=11, centers=centers)
    X_test, y_test = _make_sss_blobs(n_per_class=20, seed=12, centers=centers)

    model = KLPCDA(method=method, n_components=2, kernel="linear")
    Z_train = model.fit_transform(X_train, y_train)
    Z_test = model.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=1).fit(Z_train, y_train)
    acc = clf.score(Z_test, y_test)
    assert acc > 1.0 / 3  # chance level for 3 balanced classes
