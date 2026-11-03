# paper_alpha_recon re-run: fixed `get_S`, `fit_abstraction_K`, two blockiness levels

*2026-09-24. Re-asks the `paper_alpha_recon` figure after two changes to the metric: the
`get_S` fix (local inverse) and the switch to `fit_abstraction_K`. Adds a blockiness sweep.*

**Script** `scripts/alpha_recon_K.py` · **Plot** `scripts/plot_alpha_recon_K.py` ·
**Data** `data/alpha_recon_K.csv` (2520 rows, 0 NaN, 0 errors, 185 min) ·
**Figures** `figures/paper_alpha_recon_K_blk03.{png,pdf}`, `..._blk10.{png,pdf}`

180 jobs = 30 seeds x {d=0.1, 0.5, 0.9} x {blockiness 0.3, 1.0}, heterogeneity=0, p=100,
n_high=10. Each job scores 7 partitions (ground truth analytic, GT partition, AOC Closure,
OC Closure, Greedy All-In, GradCAP at lambda_oc in {0, 100, 1000}) under **both** fitters.
Usable seeds: 19 at (0.3, d=0.1) and 29 at (1.0, d=0.1) — the generator fails `connected=True`
at low density; 30 everywhere else.

## Method: the comparison is exactly paired

HailMary reads only `A_low` and its logits — **the fitter is not in the training loop**. Each
arm is trained ONCE and the resulting partition is scored under both fitters, so `k`, the
partition, and all training noise are identical by construction. Every K-vs-alt difference
below is the fitter alone. This is the design of FITTER_SWITCH.md, reused.

`_K` at `n_restarts=10`; `_alt` at `process_output`'s defaults (`n_restarts=5, maxiter=500,
faithful=True`) so the `_alt` row reproduces the published figure's fit.

---

## Answers at a glance

| question | answer |
|---|---|
| Does the `get_S` fix hold up? | **Yes.** The generating abstraction now scores RAE 2e-32 to 1e-30 against itself at every cell; exactness residual ~9e-16. |
| Does it change which arms move? | **Only the cyclic-quotient ones.** GradCAP 45-52% cyclic, OC Closure 38-53%. AOC Closure, Greedy All-In and GT partition are 0-6% cyclic and are essentially untouched. |
| Is `fit_abstraction_K` more exact? | **Yes, by 7 orders.** Median RAE 8.9e-14 vs 5.1e-07 at blockiness=1. |
| Can you build this figure on `_K`? | **No.** It cannot reconstruct even the ground-truth partition (below). |
| Does blockiness change the shapes? | **Yes, materially** — but mostly by moving the floor, not by re-ranking methods. |

---

## 1. The `get_S` fix is confirmed, and it matters for half the learned partitions

The generating abstraction scored against **itself**, which is the calibration the fix was
for:

| blockiness | d=0.1 | d=0.5 | d=0.9 |
|---|---|---|---|
| 0.3 | 2.8e-32 | 3.1e-31 | 6.2e-30 |
| 1.0 | 2.1e-32 | 1.4e-31 | 1.3e-30 |

Exactness residual `||(I-A)S - T(I-A_high)||_max ~ 9e-16`, and the recomputed `S` matches the
generator's own `S_gt` to **0.0 exactly**. The 0.02-0.21 self-RAE the old submatrix form
produced is gone.

**Which arms the fix actually touches** — fraction of fits whose quotient is cyclic:

| method | blk=0.3 | blk=1.0 |
|---|---|---|
| GradCAP | 0.52 | 0.45 |
| OC Closure | 0.53 | 0.38 |
| Greedy All-In | 0.06 | 0.00 |
| AOC Closure | 0.00 | 0.00 |
| GT partition | 0.00 | 0.00 |

The local inverse and the `E` submatrix agree exactly on an acyclic quotient, so the fix is a
no-op for AOC Closure, Greedy All-In and the ground truth, and changes roughly **half** of all
GradCAP and OC Closure fits. Any earlier comparison between GradCAP/OC Closure and the closure
baselines was therefore scoring the two sides under different effective metrics.

---

## 2. ⚠ `fit_abstraction_K` cannot reconstruct the ground-truth partition

At blockiness=1, `A` is exactly `T A_high T^+`, so recon=0 is attainable and the analytic
ground truth hits it. Handing each fitter the **ground-truth partition**:

| d | `_K` recon | `_alt` recon | `_K` RAE | `_alt` RAE | `_K` \|\|A_high\|\| | `_alt` \|\|A_high\|\| |
|---|---|---|---|---|---|---|
| 0.1 | **0.908** | 1.8e-12 | 6.4e-14 | 1.6e-12 | 1.45 | 5.45 |
| 0.5 | **0.330** | 2.6e-09 | 4.9e-14 | 4.0e-09 | 7.49 | 9.95 |
| 0.9 | **0.248** | 4.4e-08 | 1.3e-13 | 5.5e-09 | 12.4 | 13.9 |

`_K` is the **more exact** fitter — RAE 1e-13 against `_alt`'s 1e-9 — and still reconstructs
the correct partition at 0.25-0.91 where `_alt` reaches the floor. It marginalises 0-2 nodes
doing it, so this is not variables being dropped: it is **pure attenuation of edge scale**.
`_K` satisfies the abstraction equation by shrinking `A_high` toward zero, which the equation
permits and which RAE cannot see.

The two fitters fail in opposite directions, which is worth stating plainly:

* **`_K` attenuates weights but keeps variables.** GradCAP `||A_high||` 0.36-0.72 against
  `_alt`'s 2.5-4.3, with 3-8 nodes marginalised against `_alt`'s 8-10.
* **`_alt` drops variables but keeps the surviving weights right.** On AOC Closure it
  marginalises 14-43 nodes against `_K`'s 4-6.5 — far more — yet reconstructs better
  (0.75 vs 0.99 at d=0.1). Its `maximise ||A_high||^2 s.t. residual` constraint buys correct
  scale at the cost of discarding variables wholesale.

Root cause is the one FITTER_SWITCH.md already named: `_K` has no constrained-faithfulness
formulation. Its tie-break is `tie_tol = best*1e-3 + 1e-8`, then max `||A_high||` among tied
restarts — when the best residual is ~1e-13, `tie_tol` collapses to the additive 1e-8 and all
10 restarts sit inside it, so the selection is effectively arbitrary within a basin that is
already attenuated. **If you want this figure on `_K`, port `_alternative`'s constrained
`maximise ||A_high||^2 subject to residual <= tol` across.** Do not soften it into the
objective — `scripts/lambda_norm_test.py` already rejected that.

Paired over all fits, `_alt` beats `_K` on reconstruction in **62%** of 623 instances at
blockiness=1 (median 0.846 vs 0.992).

---

## 3. Blockiness changes the shapes, mostly by moving the floor

`recon = ||A - A_hat||_F^2 / ||A||_F^2`; 1.0 = edgeless high-level graph.

**The attainable floor**, i.e. the analytic ground truth:

| blockiness | d=0.1 | d=0.5 | d=0.9 |
|---|---|---|---|
| 0.3 | 0.574 | 0.542 | 0.508 |
| 1.0 | 0.000 | 0.000 | 0.000 |

At blockiness=0.3 a *perfect* answer still leaves 51-57% of the error, so the whole figure is
squeezed into roughly [0.5, 1.3] and every curve sits in the top half of its own range. At
blockiness=1.0 the full range is genuinely available. **This is the dominant shape effect and
it is a property of the metric's headroom, not of the methods.**

Two consequences that matter for how the figure is read:

**a) At low blockiness the fitter pathology is invisible.** `_alt` beats `_K` in only **46%**
of 553 paired fits at blk=0.3 — a coin flip — and on the GT partition the two agree to within
0.005-0.047 (0.592 vs 0.545, 0.544 vs 0.541, 0.514 vs 0.508). With the ceiling at ~0.54 there
is no room for the attenuation to show. A blockiness-0.3-only study would have concluded the
two fitters were interchangeable.

**b) The method ordering only becomes legible at blockiness=1 under `_alt`.** Values at
alpha=0.01:

| | d=0.1 | d=0.5 | d=0.9 |
|---|---|---|---|
| Greedy All-In | **0.127** | **0.173** | **0.686** |
| GradCAP lam=0 | 0.180 | 0.934 | 1.000 |
| GradCAP lam=100 | 0.247 | 0.996 | 1.000 |
| OC Closure | 0.336 | 0.985 | 2.004 |
| GradCAP lam=1000 | 0.437 | 0.994 | 1.237 |
| AOC Closure (fallback) | 0.753 | 2.063 | 7.156 |

Greedy All-In wins at every density, and at d=0.1 **one-child hurts monotonically**
(lam=0: 0.180 -> lam=100: 0.247 -> lam=1000: 0.437). At blockiness=0.3 this ordering is
compressed into a band around 0.7-1.0 and largely disappears.

---

## Caveats

* **`recon > 1` is common and is not a plotting artefact.** AOC Closure under `_alt` reaches
  2.06 (d=0.5) and 7.16 (d=0.9) — far worse than predicting an edgeless graph. It appears in
  the published `alpha_recon_b08_n15.csv` too, so it is not new, but it means the fallback
  ceiling can sit well off the top of the natural [0,1] range and the y-axes are not
  comparable across panels.
* **The alpha-curves are largely flat**, because both fitters clear even alpha=1e-4 on most
  partitions (median RAE 1e-13 / 1e-7). The tolerance axis has little left to do once the
  fitter is this exact — most of the visible alpha-dependence at blk=0.3 under `_alt` is the
  *fallback* being escaped, not methods trading exactness for reconstruction.
* **d=0.1 is under-sampled**: 19 seeds at blk=0.3, 29 at blk=1.0, against 30 elsewhere.
* Densities are `sample_low_level_graph(density=...)` on the high-level graph, not the
  low-level edge density.
