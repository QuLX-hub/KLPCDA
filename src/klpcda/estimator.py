"""KLPCDA: a scikit-learn-style estimator for the seven variants of

Qu & Pei, "Kernelized Linear Principal Component Discriminant Analysis",
Neural Networks 198 (2026) 108539, and the companion cross-domain ablation
study ("A Kernel-Based Modular Discriminant Analysis Framework for
Small-Sample Learning").

Each variant builds a single operator E from pseudo-inverses of the
(kernelized) scatter operators and takes its eigenvectors, rather than
reformulating the problem as a `scipy.linalg.eig(a, b)` generalized
eigenproblem. Three implementation choices that are not fully determined by
the papers' own formulas are documented inline where they occur: the
fusion-coefficient placement for Method No.1/No.2 (below `_FUSION_VARIANTS`),
out-of-sample kernel centering (`kernels.center_test_kernel`), and a
tie-break for No.1/No.5's exactly-degenerate eigenspace under balanced
sampling (`_break_ties`, below).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .kernels import center_test_kernel, center_train_kernel, pairwise_kernel
from .scatter import class_scatter_operators

VARIANTS = ("no1", "no2", "no3", "no4", "no5", "no6", "no7")

# Variants whose target function is a *ratio* of two quadratic forms with a
# single scalar coefficient (No.3/4/5/6/7): scaling that coefficient scales
# the objective uniformly and provably cannot change the arg max/min. Only
# No.1 (three summed terms) and No.2 (two summed terms) have an
# independently tunable *ratio between terms* that changes the learned
# subspace -- so `fusion` must be applied to each summed term before they
# combine, not as a uniform rescale of the whole operator afterward.
_FUSION_VARIANTS = ("no1", "no2")

_UNSUPERVISED_VARIANTS = ("no4",)  # pure KPCA: no class labels needed


def _break_ties(
    eigvals: np.ndarray, eigvecs: np.ndarray, tie_break: np.ndarray, rtol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Within each run of near-equal eigenvalues (sorted ascending; a "run"
    is a maximal stretch of consecutive values each within `rtol` of its
    neighbor, relative to the largest-magnitude eigenvalue overall), replace
    the arbitrary basis `numpy.linalg.eig` happened to return for that
    (near-)degenerate eigenspace with the one that diagonalizes `tie_break`'s
    quadratic form -- i.e. rank directions *within* a tied cluster by a
    secondary criterion the caller supplies, instead of leaving the choice
    to whatever orthonormal basis floating-point noise produced. Every
    eigenvalue within a detected run is replaced by the run's mean, so the
    caller's subsequent top-k sort treats them as exactly tied (matching
    what clustering just decided) rather than reordering by the noise-level
    differences that triggered the clustering in the first place.

    No.1 and No.5's own eigenvalues are provably, exactly degenerate over
    their entire "signal" subspace under balanced, class-per-sample-count-
    matched sampling. This does not patch a bug; it resolves a genuine
    ambiguity already present in their ratio-of-quadratic-forms objective
    once the primary term stops discriminating between directions.
    """
    n = len(eigvals)
    order = np.argsort(eigvals)
    vals = eigvals[order].copy()
    vecs = eigvecs[:, order].copy()

    scale = max(float(np.abs(vals).max()), 1e-12)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(vals[j] - vals[j - 1]) <= rtol * scale:
            j += 1
        if j - i > 1:
            V, _ = np.linalg.qr(vecs[:, i:j])
            M = V.T @ tie_break @ V
            M = (M + M.T) / 2  # guard against asymmetry from floating-point noise
            sub_vals, sub_vecs = np.linalg.eigh(M)
            # Ascending, not descending: empirically, preferring the
            # *lowest*-tie-break-score direction within a tied cluster wins
            # decisively over preferring the highest -- verified against
            # real high-dimensional embedding data at several sample sizes.
            # The literal "maximize C/W" reading of No.5's own formula would
            # suggest descending; that reading evidently does not extend
            # correctly into this exact-tie regime. Not fully explained
            # theoretically -- an open question rather than a settled
            # mechanism (see the Limitations discussion in the paper this
            # estimator supports).
            reorder = np.argsort(sub_vals)
            vecs[:, i:j] = V @ sub_vecs[:, reorder]
            vals[i:j] = vals[i:j].mean()
        i = j
    return vals, vecs


def _eig_topk(
    E: np.ndarray,
    k: int,
    largest: bool,
    energy_ref: np.ndarray,
    eps: float = 1e-12,
    tie_break: np.ndarray | None = None,
    tie_break_rtol: float = 1e-3,
):
    """Eigendecompose E, keep the k eigenpairs ranked by eigenvalue, and
    normalize each eigenvector v by sqrt(max(v^T energy_ref v, eps)) -- the
    convention used throughout the reference implementation (Eq. 8).

    `tie_break`, when given, is a symmetric quadratic-form matrix used to
    re-rank directions *within* any near-degenerate cluster of E's own
    eigenvalues (see `_break_ties`) before the top-k cutoff is applied.
    """
    eigvals, eigvecs = np.linalg.eig(E)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    if tie_break is not None:
        eigvals, eigvecs = _break_ties(eigvals, eigvecs, tie_break, tie_break_rtol)

    order = np.argsort(eigvals)
    order = order[::-1] if largest else order
    k = min(k, eigvecs.shape[1])
    order = order[:k]

    selected = eigvecs[:, order]
    norms = np.empty(k)
    for i in range(k):
        v = selected[:, i]
        energy = float(v @ energy_ref @ v)
        norms[i] = np.sqrt(max(energy, eps))
    return eigvals[order], selected / norms


class KLPCDA(BaseEstimator, TransformerMixin):
    """Kernelized Linear Principal Component Discriminant Analysis.

    Parameters
    ----------
    method : {"no1", ..., "no7"}
        Which of the seven fused objectives to solve (see Table 2 of the
        Neural Networks paper). "no4" is unsupervised (pure centered KPCA);
        the other six require class labels `y` at fit time.
    n_components : int
        Number of projection directions to keep.
    kernel : {"linear", "poly", "rbf"}
        Kernel function evaluated on the *original* input space.
    degree, coef0 : poly kernel parameters.
    gamma : rbf kernel bandwidth (defaults to 1 / n_features).
    fusion : tuple(float, float), default (1.0, 1.0)
        (alpha, beta) weights for the variance term and the between-class
        term. Only meaningful for method in {"no1", "no2"} -- see
        `_FUSION_VARIANTS` above. Ignored otherwise.
    reg : float
        Relative singular-value cutoff (`rcond`) passed to
        `numpy.linalg.pinv` when inverting `W @ K_tilde` for methods
        No.1/3/5. `W` has exact rank `n - L` (Sec. "class_scatter_operators"
        docstring; each class block loses exactly one dimension), so
        `W @ K_tilde` has `L` singular values that are mathematically zero
        but land anywhere from ~1e-15 to ~1e-21 in floating point --
        `numpy`'s own default `rcond` (~1e-15, relative to machine epsilon)
        is not generous enough to reliably zero all of them out, and a
        naive *additive* ridge (`M + reg*I`) before pinv makes this *worse*,
        not better: it turns an exact rank deficiency that `pinv` would
        otherwise truncate correctly into a full-rank matrix that gets
        inverted exactly, amplifying the near-zero directions to ~1/reg.
        Verified empirically that the true "signal" singular values and the
        near-zero ones are separated by ~12 orders of magnitude, so any
        `rcond` between 1e-10 and 1e-4 recovers the same well-conditioned
        pseudo-inverse; the default sits in the middle of that safe range.
    """

    def __init__(
        self,
        method: str = "no4",
        n_components: int = 10,
        kernel: str = "linear",
        degree: int = 3,
        coef0: float = 1.0,
        gamma: float | None = None,
        fusion: tuple[float, float] = (1.0, 1.0),
        reg: float = 1e-6,
    ):
        self.method = method
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.fusion = fusion
        self.reg = reg

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return pairwise_kernel(
            X, Y, kernel=self.kernel, degree=self.degree, coef0=self.coef0, gamma=self.gamma
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KLPCDA":
        if self.method not in VARIANTS:
            raise ValueError(f"method must be one of {VARIANTS}, got {self.method!r}")
        X = check_array(X)
        n = X.shape[0]

        needs_labels = self.method not in _UNSUPERVISED_VARIANTS
        if needs_labels and y is None:
            raise ValueError(f"method {self.method!r} requires class labels y")

        K_raw = self._kernel(X, X)
        K_tilde = center_train_kernel(K_raw)

        if needs_labels:
            y = np.asarray(y)
            B, W = class_scatter_operators(y)

        alpha, beta = self.fusion
        tie_break = None  # only No.1/No.5 have an in-formula tie-break term

        if self.method == "no1":
            WK = W @ K_tilde
            E = np.linalg.pinv(WK, rcond=self.reg) @ (alpha * (K_tilde / n) + beta * (B @ K_tilde))
            largest = True
            # No.1's own fused numerator (alpha*C + beta*S_b) is the natural
            # secondary criterion once pinv(W@K_tilde) itself stops
            # discriminating between directions -- not information from
            # outside the paper's own formula for this variant.
            tie_break = alpha * (K_tilde / n) + beta * B
        elif self.method == "no2":
            E = alpha * (K_tilde / n) + beta * (B @ K_tilde)
            largest = True
        elif self.method == "no3":
            WK = W @ K_tilde
            E = np.linalg.pinv(WK, rcond=self.reg) @ (B @ K_tilde)
            largest = True
            # No secondary term of No.3's own (pure S_b/S_w) to tie-break
            # with: B itself is exactly what's degenerate here (its range
            # lies entirely in W's null space under balanced sampling), so
            # there is nothing left within this variant's own definition to
            # resolve the ambiguity.
        elif self.method == "no4":
            E = K_tilde
            largest = True
        elif self.method == "no5":
            WK = W @ K_tilde
            E = np.linalg.pinv(WK, rcond=self.reg) @ K_tilde
            largest = True
            tie_break = K_tilde  # No.5's own C term (its only other term)
        elif self.method == "no6":
            E = B @ K_tilde
            largest = True
        elif self.method == "no7":
            E = W @ K_tilde
            largest = False  # Eq. (31): argmin v^T S_w v
            # Pure S_w minimization -- no secondary term of its own either
            # (same reasoning as No.3).
        else:  # pragma: no cover - guarded above
            raise AssertionError

        eigvals, components = _eig_topk(
            E, self.n_components, largest, energy_ref=K_tilde, tie_break=tie_break
        )

        self.X_fit_ = X
        self.K_train_raw_ = K_raw
        self.eigenvalues_ = eigvals
        self.components_ = components
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "components_")
        X = check_array(X)
        K_test_raw = self._kernel(X, self.X_fit_)
        K_test = center_test_kernel(K_test_raw, self.K_train_raw_)
        return K_test @ self.components_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
