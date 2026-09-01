# Design log: Python port vs. the reference MATLAB implementation

This file records every place where `src/klpcda/` deliberately diverges from
the MATLAB reference (`reference/matlab/KLPCDA_main.m`,
`reference/matlab/poly_kernel.m`) that accompanied the submission of
Qu & Pei (2026), *Kernelized Linear Principal Component Discriminant
Analysis*, Neural Networks 198, 108539. The MATLAB file's own docstring
calls it a "minimal runnable implementation," so some divergence is
expected; each entry below states what was found, why it matters, and what
this repo does instead. Nothing here is a criticism of the paper's math —
both issues are implementation-level, in a script the authors labelled as a
simplified demo, not the published tables themselves.

## 1. Fusion-coefficient placement for Method No.1 / No.2

**What the MATLAB does.** `method_E_builder` first assembles the operator
`E` with unit weights (e.g., for Method No.2: `E = (1/n)*Ktr + B*Ktr`), and
*then* rescales the whole thing: `E = p1*E + p2*E`, i.e. `E <- (p1+p2)*E`.
The stage-2 grid search in `KLPCDA_main.m` searches over `p1`, `p2` under
this rule.

**Why it matters.** For any square matrix `E` and eigenpair `E v = lambda v`,
scaling `E` by a positive constant `c` gives `(cE) v = (c*lambda) v` — the
eigen*vectors* are unchanged, only the eigenvalues scale. Since both the
grid-search ranking heuristic (`u' * Ktr * u`, which depends only on `u`)
and the final evaluation's eigenvalue-based ranking use only the
eigenvectors (or a monotone function of a scaled eigenvalue, which preserves
ordering), a uniform rescale of `E` cannot change which projection
directions are picked. As coded, tuning `(p1, p2)` for Methods No.1/No.2 is
a no-op for the learned subspace — it can only rescale reported eigenvalues.

This is specific to No.1/No.2. For No.3/No.5/No.6/No.7, the paper's own
target functions are a *single-coefficient ratio* applied to the whole
objective (Table 2: e.g. No.6 is `argmax beta * v^T S_b v`) — for those,
scaling really is mathematically inert by construction, not a bug. It only
becomes a bug for No.1/No.2, where the intent (Eq. 19–24) is to weight two
*independently summed* terms (`alpha * v^T C v + beta * v^T S_b v`) before
they interact with the rest of the eigenproblem.

**What this repo does instead.** `KLPCDA.fit` applies `fusion = (alpha,
beta)` to the variance term and the between-class term *before* combining
them (and, for No.1, before the pseudo-inverse of `W @ K_tilde`):

```python
# No.2
E = alpha * (K_tilde / n) + beta * (B @ K_tilde)
# No.1
E = pinv(W @ K_tilde + reg*I) @ (alpha * (K_tilde / n) + beta * (B @ K_tilde))
```

This makes `fusion` actually change the learned subspace for No.1/No.2, per
Eqs. (19)–(24). No `fusion`-like parameter is exposed for No.3–No.7, since a
single coefficient there cannot affect direction selection either way.

**Status:** not yet re-validated against the paper's Table 3/4 numbers
(those describe MNIST-scale experiments this repo hasn't run). If you rerun
that comparison and it doesn't reproduce, look here first.

## 2. Out-of-sample kernel centering

**What the MATLAB does.** `center_kernel_cross(Kx_raw, Ktr_cent)` computes
correction terms from `Ktr_cent`, the *already double-centered* training
kernel, not the raw one:

```matlab
mean_Ktr_rows = mean(Ktr_cent, 2);   % ~= 0 by construction, see below
grand = mean(Ktr_cent(:));           % ~= 0 by construction, see below
Kx = Kx_raw - repmat(mean_Ktr_rows', m, 1) - repmat(mean_Kx_cols, 1, n) + grand;
```

**Why it matters.** Double-centering is `K_tilde = H K H` with `H = I -
(1/n) * ones(n,n)`, and `H` annihilates the all-ones vector: `H @ 1 = 0`.
So `K_tilde @ 1 = 0` exactly (up to floating point) — every row/column of
`Ktr_cent` sums to zero, and its grand mean is zero. Passing `Ktr_cent`
into a function that computes "the mean of this matrix's rows/entries"
therefore always returns (numerically) zero, silently dropping two of the
three correction terms in the standard out-of-sample centering formula
(Schölkopf et al., 1998):

```
K_tilde(x', x_j) = k(x', x_j) - (1/n) sum_l k(x', x_l)
                              - (1/n) sum_l k(x_l, x_j)      <- needs RAW K_train
                              + (1/n^2) sum_l sum_m k(x_l, x_m)  <- needs RAW K_train
```

The dropped terms are only *actually* zero if the raw kernel's own row/
column means are zero — true for a linear kernel on already mean-centered
input `X` (which paper 2, Sec. 2.2, explicitly assumes: "given zero-centered
input samples"), but not generally true for the polynomial kernel that every
reported experiment in papers 1/2/3 uses, nor for arbitrary embeddings this
repo may be fed (real-world feature vectors are not guaranteed zero-mean).
So this under-centers test points whenever the assumption doesn't hold,
biasing projections of new data (training-set fit is unaffected, since
`Ktr_cent` there is centered directly, not via this cross-kernel path).

**What this repo does instead.** `kernels.center_test_kernel` (written
before this MATLAB file was shared, and kept as-is after the comparison)
takes the *raw* training kernel and computes the correction terms from it
directly:

```python
def center_test_kernel(K_test, K_train):  # K_train is RAW, uncentered
    train_row_mean = K_train.mean(axis=0, keepdims=True)
    train_total_mean = K_train.mean()
    test_row_mean = K_test.mean(axis=1, keepdims=True)
    return K_test - test_row_mean - train_row_mean + train_total_mean
```

`KLPCDA` stores `self.K_train_raw_` (uncentered) at fit time specifically so
`transform()` can use this correct version.

## 3. Eigenvector ranking: eigenvalue-based, not energy-based

The MATLAB script uses two *different* ranking rules in two places: during
the stage-1/stage-2 grid search it sorts eigenvectors by projected energy
`diag(U' * Ktr * U)` (descending, regardless of method), but in the final
per-fold evaluation it sorts by the eigenvalue itself (`ascending` for
Method No.7, `descending` otherwise). This repo only implements the second
rule (eigenvalue-based, matching the paper's stated `argmax`/`argmin
lambda`), applied consistently everywhere — there is no separate
grid-search-time heuristic here since this repo does not yet reimplement
the two-stage (kernel/dimension, then fusion-coefficient) grid search from
`KLPCDA_main.m` at all.

## 4. `reg`: additive ridge before `pinv` vs. `rcond` inside it

**Found:** running this estimator on a real high-dimensional dataset (many
classes, few examples per class — the small-sample-size regime this repo
targets), Methods No.1, No.3, and No.5 (all three built via `pinv(W @
K_tilde) @ something`) produced *identical* accuracy to six decimal places.
Since their target functions genuinely differ (No.1 fuses all three terms,
No.3 is `S_b/S_w`, No.5 is `C/S_w`), this could only mean the three
operators were collapsing to the same dominant eigenvectors regardless of
what multiplied `pinv(W @ K_tilde)` on the right.

**Root cause.** `W` has *exact* rank `n - L` (`class_scatter_operators`:
each class block loses exactly one dimension along its "all-ones" direction
-- see the docstring there). With many classes and few examples each, that
null space is a large fraction of `n`. `np.linalg.svd(W @ K_tilde)` on a
real fold showed a striking spectrum: hundreds of "real" singular values
decaying smoothly down to a small floor, then an *immediate*
~12-order-of-magnitude drop to a cluster of near-zero values -- i.e. the
rank deficiency is numerically about as clean as floating point allows,
`numpy`'s own default `rcond` for `pinv` (relative to machine epsilon,
effectively an absolute cutoff around `1e-15` here) sat *inside* that noise
cluster rather than safely above it, and the earlier code additionally
added `W @ K_tilde + 1e-8 * I` before calling `pinv` at all.

That additive ridge is precisely backwards: it turns an *exactly*
rank-deficient matrix -- which `pinv` would otherwise truncate correctly --
into a numerically full-rank one, so every one of those near-zero
directions gets inverted almost exactly instead of zeroed out, landing
around `1/1e-8 = 1e8` (empirically the resulting `pinv` had spectral norm
`~1.7e9`; with no ridge at all it was `~5.8e15`, i.e. *worse*, since
`numpy`'s default `rcond` didn't reliably clear the noise cluster either).
Either way, the pseudo-inverse ends up dominated by directions that live
entirely inside `W`'s null space -- which has nothing to do with `B` or
`C` -- so No.1/No.3/No.5 all end up picking (approximately) the same
leading eigenvectors.

**Fix.** Stop adding a ridge before `pinv`; instead pass an explicit,
generous `rcond` *into* `pinv` (`np.linalg.pinv(W @ K_tilde, rcond=self.reg)`),
so the known-exact rank deficiency is truncated rather than inverted.
Checked empirically that any `rcond` between `1e-10` and `1e-4` recovers the
identical, well-conditioned pseudo-inverse (spectral norm `~6.5e4`) on the
fold above -- the gap between "real" and "noise" singular values is wide
enough that the exact choice doesn't matter within that range. Default
changed to `reg=1e-6`. See `tests/test_estimator.py::
test_no1_no3_no5_do_not_collapse_when_W_is_rank_deficient` for a synthetic
reproduction (20 classes, 3 examples each) that would have caught this
before it reached real data.

**Not present in the MATLAB reference** for a different reason than items
1–2 above: `KLPCDA_main.m`'s `pinv_safe` calls plain `pinv(M)` on the
*happy path* (MATLAB's own rcond-based truncation), and only falls back to
an additive `pinv(M + 1e-8*eye(...))` if `pinv(M)` throws an exception --
which it rarely does for a well-formed matrix, since `pinv` doesn't error
on rank deficiency, it handles it. This bug was introduced by this Python
port's `reg` parameter defaulting to "always add ridge," not inherited from
MATLAB.

## 5. Tie-breaking No.1/No.5's exactly-degenerate eigenspace under balanced sampling

**Found.** Under *balanced* sampling (every class contributing the same
number of examples), `W`'s within-class scatter operator is exactly `(1/L)
* Pi` for an orthogonal projector `Pi` (`L` = number of classes) -- provable
in closed form and verified against real embedding data to float64
precision. This makes No.5's entire `pinv(W@K_tilde)@K_tilde` "signal"
eigenspace (all `n-L` dimensions) share the *exact same* eigenvalue (`L`,
to ~1e-10 precision), and No.1 (which shares the same `pinv(W@K_tilde)`
factor) collapses to `1/k` similarly, since its `B`-term contributes exactly
zero within that subspace (`B`'s range lies entirely in `Pi`'s null space,
provable the same way). `E`'s own eigenvalue-based ranking therefore
carries *zero* information about which directions within that subspace to
prefer -- `numpy.linalg.eig` returns *some* orthonormal basis of the tied
eigenspace, but which one is essentially arbitrary, sensitive to
floating-point noise rather than any real signal.

**Why it matters.** No.5 and No.1 fold their entire `n_components`
selection into this arbitrary choice under exactly the balanced-sampling
condition this repo's small-sample-size regime is built around -- a real
failure mode unrelated to embedding quality or `n/d`, and easy to miss
without a closed-form argument for why it happens.

**What this repo does.** `_eig_topk` (in `estimator.py`) accepts an
optional `tie_break` matrix: after the usual eigendecomposition, any run
of near-equal eigenvalues (relative tolerance `1e-3`, see `_break_ties`)
gets its arbitrary basis replaced by the one that diagonalizes
`tie_break`'s own quadratic form. For No.5, `tie_break = K_tilde` --
literally the *other* term already in its own `Cv/S_w v` definition, used
to resolve the ambiguity `pinv(W@K_tilde)` leaves behind rather than
introducing information from outside the variant's own formula. For No.1,
`tie_break = alpha*(K_tilde/n) + beta*B` -- its own fused numerator, for
the same reason. No.3 (`S_b/S_w`, no `C` term of its own, and `B` is
exactly the thing proven void in this subspace) and No.7 (pure `argmin
S_w`, no second term at all) have no comparable in-formula fallback and are
deliberately left untouched -- their degeneracy is characterized, not
patched.

**A genuine open question, not swept under the rug: the *direction* that
works is not the one the literal formula would suggest.** No.5's formula
reads `argmax Cv/S_w v`; on the degenerate subspace `S_w v` is *constant*
(`= 1/L` for every unit `v`, since `W = (1/L)*Pi` there), so "faithfully"
extending "maximize" would mean preferring the *highest*-`C` direction
within the tied cluster. Tried that first -- it made things dramatically
*worse* than the pre-fix arbitrary baseline. The *opposite* choice --
lowest-`C`-first -- instead worked decisively better, and this pattern
replicated cleanly across multiple sample sizes and two independent
real-world datasets. `_break_ties` therefore ranks ascending, not
descending, despite that being the empirically-validated choice rather than
the theoretically "obvious" one. A working hypothesis -- `Pi`'s range is
exactly the "within-class-only-varies" subspace (orthogonal complement of
the class-mean directions), so *low* variance there plausibly picks the
most internally-consistent, lowest-noise directions for a class, which a
Fisher-LDA-style intuition would favor even though it doesn't match a
literal "maximize `C`" reading -- is offered as a hypothesis, not a proven
mechanism. Flagging this precisely so a future reader (or a closer read of
the source papers' actual derivation, not available here) can either
confirm or overturn it, rather than silently treating "ascending won
empirically" as if it were understood.

**Numerical detail that mattered:** an initial `rtol=1e-4` cluster-detection
tolerance produced one catastrophic per-run outlier, traced to that run's
degenerate cluster having a couple of eigenvalues sitting just outside the
tolerance, splitting one true cluster into an inconsistent mix of tied and
untied directions. `rtol=1e-3` (verified identical to `1e-2`, so not a
knife-edge choice) resolved it completely.

**Regression tests:** `tests/test_estimator.py::
test_break_ties_ranks_by_tie_break_ascending` (unit-level: a synthetic
exactly-degenerate matrix, checks the ascending-by-tie-break property
directly) and `::test_no1_no5_use_own_c_term_as_tie_break` (checks the
tie-break actually changes the selected direction versus the untied
baseline, in the same rank-deficient regime as item 4's regression test).

## 6. Hyperparameter search is not ported

`KLPCDA_main.m`'s two-stage grid search (kernel parameter + subspace
dimension, then fusion coefficients) and the 10-fold cross-domain evaluation
harness are not reimplemented here. This repo's estimator is fit/transform
only; hyperparameter search is left to the caller (e.g., `sklearn`'s
`GridSearchCV`).
