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
ratios, each solvable as a single eigenproblem on an `n x n` kernel matrix
rather than a `d x d` scatter matrix.

## Applying it to frozen LLM embeddings

`llm_pipeline/` connects `KLPCDA` to sentence-embedding models: extract and
cache embeddings for a text classification dataset
(`llm_pipeline/extract_embeddings.py`, via `sentence-transformers`), then
draw a stratified k-shot support set (`llm_pipeline/sampler.py`) to `fit` a
`KLPCDA` model directly on the cached embedding vectors. This is the
direction the project is actively moving in: whether small-sample
discriminant analysis, originally developed for pixels, vibration signals,
and gene expression, transfers to the `n << d` regime that few-shot text
classification on frozen LLM embeddings sits in by construction.

```bash
pip install -e ".[embeddings]"
python -m llm_pipeline.extract_embeddings \
    --dataset mteb/banking77 --model sentence-transformers/all-MiniLM-L6-v2 \
    --out cache/banking77_minilm.npz
```

```python
import numpy as np
from klpcda import KLPCDA
from llm_pipeline.sampler import k_shot_indices

cached = np.load("cache/banking77_minilm.npz", allow_pickle=True)
X_train, y_train = cached["X_train"], cached["y_train"]

support_idx = k_shot_indices(y_train, k=5, seed=0)
model = KLPCDA(method="no6", n_components=10, kernel="linear")
Z_support = model.fit_transform(X_train[support_idx], y_train[support_idx])
```

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

25 unit tests cover all seven variants' output shapes, the No.4-equals-PCA
identity, the fusion-coefficient behavior for No.1/No.2, the k-shot sampler,
and two regression tests for numerical failure modes of the estimator
itself (documented inline in `estimator.py`): a rank-deficiency edge case in
the pseudo-inverse used by No.1/3/5, and a tie-break for No.1/No.5's
exactly-degenerate eigenspace under balanced sampling.

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
the learned subspace, not just the reported eigenvalues (see the
`estimator.py` docstring).

## Structure

```
src/klpcda/
    estimator.py   # KLPCDA (BaseEstimator, TransformerMixin)
    scatter.py      # between-/within-class scatter operators B, W
    kernels.py       # linear/poly/rbf kernels + double-centering
llm_pipeline/        # extraction + k-shot sampling for frozen LLM embeddings
tests/               # 25 unit tests
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
