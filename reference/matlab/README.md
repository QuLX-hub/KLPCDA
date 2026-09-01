# Reference MATLAB implementation

This directory holds the MATLAB code the author shared alongside the
Neural Networks submission — kept verbatim for reproducibility and as the
ground truth `DECISIONS.md` (at the repo root) compares the Python port
against. Do not "fix" the bugs noted there in place; if you need a working
MATLAB baseline, patch a copy and note it separately.

- `KLPCDA_main.m` — two-stage grid search (kernel degree + subspace
  dimension, then fusion coefficients) and 10-fold evaluation for all seven
  methods, driven by pre-split `.mat` files.
- `poly_kernel.m` — polynomial kernel `(u v^T + 1)^d`.
- `IndianPines_preprocess.m` — example dataset preprocessing / fold
  generation script.

See the repo-root `DECISIONS.md` for the two implementation issues found
while porting this to Python (`src/klpcda/estimator.py`):

1. `method_E_builder`'s fusion-coefficient rescaling (`E = p1*E + p2*E`)
   happens *after* the operator is already assembled, which cannot change
   eigenvectors — only eigenvalues — making the Stage 2 coefficient search
   a no-op for Methods No.1/No.2 as written.
2. `center_kernel_cross` is passed the already-centered training kernel
   instead of the raw one, so two of the three out-of-sample centering
   correction terms are (numerically) zero regardless of the actual data.

Original MATLAB `README.md` (citation, usage) is reproduced below.

---

## KLPCDA MATLAB Implementation

### Kernelized Linear Principal Component Discriminant Analysis (KLPCDA)

**Author:** Lingxiao Qu
**Affiliation:** The University of Aizu, Japan
**Date:** December 2025

### Overview

This repository provides the MATLAB implementation of the KLPCDA
framework, including seven KLPCDA variants (Method 1–7) designed for
small-sample and high-dimensional learning problems.

### Usage

**Step 1 — Dataset Preprocessing.** Run the preprocessing script for the
target dataset, e.g. `IndianPines_preprocess.m`, which generates
`traindata01.mat` ... `testlabels10.mat` per fold, stored under
`~/IndianPines/`.

**Step 2 — Run KLPCDA.**

```matlab
KLPCDA_main(1, 1);   % datasetID=1 (Indian Pines), methodID=1
```

`methodID` ranges 1–7 for the seven KLPCDA variants.

### Experimental settings

- Kernel: polynomial only, `k(x,z) = (x^T z + 1)^d`.
- Classifier: 1-nearest-neighbor.
- Parameter optimization: validation-driven two-stage grid search.

### Citation

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

### License

This project is intended for academic and research use.
