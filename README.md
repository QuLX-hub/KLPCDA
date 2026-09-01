# KLPCDA

A scikit-learn-style estimator for **Kernelized Linear Principal Component
Discriminant Analysis (KLPCDA)** — seven variants of small-sample,
high-dimensional discriminant analysis (n << d), from

> Qu & Pei, "Kernelized Linear Principal Component Discriminant Analysis",
> *Neural Networks* 198 (2026) 108539.

The core idea: in the small-sample regime, feature-space scatter matrices
(`S_b`, `S_w`) become rank-deficient or unreliable to estimate directly.
KLPCDA works entirely in kernel space instead — the seven variants fuse a
variance term, a between-class term, and a within-class term in different
ratios, each solvable as a single generalized-eigenvalue-free eigenproblem
on an `n x n` kernel matrix rather than a `d x d` scatter matrix.

## Why this repo exists

This is a from-scratch Python port of the authors' reference MATLAB
implementation, built to be a normal `fit`/`transform` scikit-learn
estimator instead of a script. Porting it surfaced two implementation bugs
in the reference MATLAB code and one previously undocumented numerical
failure mode in the Python port itself — all three are written up in detail,
with the reasoning and the fix, in [`DECISIONS.md`](DECISIONS.md). The
reference MATLAB is kept alongside the Python port in
[`reference/matlab/`](reference/matlab/) specifically so the comparison is
auditable rather than asserted.

21 unit tests cover all seven variants' output shapes, the No.4-equals-PCA
identity, the fusion-coefficient behavior for No.1/No.2, and two regression
tests for the numerical failure modes documented in `DECISIONS.md`.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

```python
import numpy as np
from klpcda import KLPCDA

X_train, y_train = ...  # n_samples << n_features
X_test = ...

model = KLPCDA(method="no6", n_components=10, kernel="linear")
Z_train = model.fit_transform(X_train, y_train)
Z_test = model.transform(X_test)
```

## The seven variants

| Method | Objective | Labels needed |
|---|---|---|
| No.1 | `argmax (alpha*C + beta*S_b) / S_w` | yes |
| No.2 | `argmax alpha*C + beta*S_b` | yes |
| No.3 | `argmax S_b / S_w` | yes |
| No.4 | `argmax C` (= kernel PCA) | no |
| No.5 | `argmax C / S_w` | yes |
| No.6 | `argmax S_b` | yes |
| No.7 | `argmin S_w` | yes |

`C` is the (centered, kernelized) total-variance term; `S_b`/`S_w` are the
kernelized between-/within-class scatter terms (`B`, `W` in
[`src/klpcda/scatter.py`](src/klpcda/scatter.py)). `method="no1"` and
`"no2"` additionally expose a `fusion=(alpha, beta)` parameter that changes
the learned subspace, not just the reported eigenvalues — see
`DECISIONS.md` item 1 for why that distinction isn't automatic in a naive
port.

## Structure

```
src/klpcda/
    estimator.py   # KLPCDA (BaseEstimator, TransformerMixin)
    scatter.py      # between-/within-class scatter operators B, W
    kernels.py       # linear/poly/rbf kernels + double-centering
tests/               # 21 unit tests, incl. regressions for DECISIONS.md items 4-5
reference/matlab/    # the authors' original MATLAB implementation (ground truth)
DECISIONS.md         # every place this port diverges from the MATLAB reference, and why
```

## Citation

```bibtex
@article{qu2026klpcda,
  title={Kernelized Linear Principal Component Discriminant Analysis},
  author={Qu, Lingxiao and Pei, Yan},
  journal={Neural Networks},
  volume={198},
  pages={108539},
  year={2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
