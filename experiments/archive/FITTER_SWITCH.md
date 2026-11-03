# Does switching to `fit_abstraction_alternative` change the loss conclusions?

*2026-09-15. `process_output` (src/utils/helper.py:483) now calls `fit_abstraction_alternative`
instead of `fit_abstraction`. Every loss conclusion in [REPORT.md](REPORT.md) was measured with
the old fitter. This re-asks each one of both fitters.*

**Scripts:** `scripts/lossfit.py`, `scripts/nochild_restarts.py`, `scripts/graph_recon.py`,
`scripts/altbudget.py` — analysed by `scripts/analyse_lossfit.py`, `scripts/analyse_nochild.py`.
**Data:** `data/lossfit.csv` (58 instances × 11 arms × 2 fitters),
`data/nochild_restarts.csv` (18 × 5 arms × 8 restarts × 2 fitters).

## Method: the comparison is exactly paired

HailMary training reads only `A_low` and its logits — **the fitter is not in the training
loop**. So each arm is trained ONCE and the resulting partition is scored under BOTH fitters.
`allin`, `one_child`, `valid`, `k_used` and `info` are therefore *identical* across fitters by
construction (asserted in the analysis); every difference reported below is the fitter alone,
with no training noise mixed in.

---

## Arm names (the same config has more than one name across scripts)

`lossfit.py`, `nochild_restarts.py`, `graph_recon.py` and `ratio.py` were written for different questions
and their arm names drifted. Duplicates are genuine independent replications -- different
instances and seeds, same lambdas.

| lambdas (graph / one_child / info / recon) | names used |
|---|---|
| 10 / 0 / 0 / 0 | **`graph_only`** = **`no_oc_graph`** |
| 0 / 0 / 10 / 0 | **`info_only`** = **`no_oc_info`** |
| 10 / 0 / 10 / 0 | `no_oc_g_info` |
| 10 / 0 / 0 / 10 | **`no_oc_g_recon`** = **`g10_recon`** |
| 0 / 0 / 0 / 10 | `recon_only` |
| 0 / 100 / 10 / 0 | `oc_info` (the current recommendation) |
| 1 / 100 / 10 / 0 | `oc_info_g1` |
| 10 / 100 / 10 / 0 | `oc_info_g10` |
| 100 / 100 / 10 / 0 | `oc_info_g100` |
| 0 / 100 / 1 / 0 | `oc_info_u1` |
| 0 / 100 / 100 / 0 | `oc_info_u100` |
| 0 / 100 / 0 / 10 | `oc_recon` |
| 1 / 100 / 0 / 10 | **`oc_recon_g1`** = **`g1_recon_oc`** |
| 10 / 100 / 0 / 10 | `g10_recon_oc` |
| 10 / 100 / 10 / 10 | `g10_rec_inf_oc` |

`no_oc_*` and `*_only` are the one-child-OFF arms; `oc_*` and `*_oc` keep
`lambda_one_child=100`. `lambda_surj=0` and 6000 epochs throughout.

---

## Answers at a glance

| claim in REPORT.md | `fit_abstraction` | `fit_abstraction_alternative` | verdict |
|---|---|---|---|
| `info` (intra_mass) beats `recon` | 52/58 paired wins; hybrid 0.353 vs 0.485 | 55/58; 0.338 vs 0.468 | **unchanged** |
| `lambda_info = 10` optimal | 0.353 vs 0.371 (λ=1) / 0.435 (λ=100) | 0.338 vs 0.371 / 0.371 | **unchanged** |
| `lambda_graph = 0` recommended | 0 ≈ 1, a coin flip | **λ=1 wins** at every tight α | **CHANGED** |
| `lambda_one_child` load-bearing | All-In 0/58 without it | All-In 0/58 without it | **unchanged (structurally)** |
| one-child needed at all | assumed | **only below α ≈ 0.05** | **CHANGED** |

Two measurement bugs surfaced on the way, both of which would have corrupted the comparison
had they gone unnoticed. They are the most important findings here.

---

## ⚠ 0. The switch made the pipeline NON-DETERMINISTIC  [fixed]

`fit_abstraction` defaults to `seed=0`. `fit_abstraction_alternative` defaults to
**`seed=None`** (helper.py:633), so `rng = np.random.RandomState(None)` drew fresh SLSQP
starting points on every call, and `process_output` passed no seed. Re-scoring the SAME
partitions across two runs moved exactness by **+-0.04** (0.431 vs 0.444 at (3,100); 0.569 vs
0.528 at (10,500)) -- the same magnitude as the differences between the loss arms this report
is trying to resolve. On a single OCC partition, seeds 0/1/2 gave `A_high` matrices differing
by up to **22.2** in absolute entry magnitude, so this was never cosmetic.

**Fixed:** `process_output` now forwards `seed` (default 0). Three identical calls now return
bit-identical `A_high`. Any single-run comparison of two loss configurations made before this
fix carries this noise.

---

## ⚠ 1. `info_eff` silently stops charging for discarded nodes

`info_eff` counts mass incident to *marginalised* nodes as lost, testing marginalisation at
**exact zero**. That test is fitter-specific:

* `fit_abstraction` propagates `t_X ∝ W_Xc (I − W_cc)^{-1} t_c`, so a member of X with no edge
  into the child c gets an **exact structural zero**. This is the mechanism REPORT.md calls
  "marginalising the offenders". Measured: 5.7 of 50–100 nodes on average.
* `fit_abstraction_alternative` optimises every entry freely under a per-cluster unit-norm
  constraint, so those same nodes come out at **1e-4 … 1e-2 instead of 0**. The exact test
  counts **none** of them — mean `marginalised` is 0.00 across all 638 rows.

So under the alternative `info_eff` collapses to plain intra-cluster `info` for every arm, and
reads a partition as losslessly retaining mass it has in fact all but discarded:

| arm | `info` (intra) | `info_eff` exact-zero | `info_eff` @1e-2 | understated by |
|---|---|---|---|---|
| `oc_info` | 0.301 | **0.301** | 0.416 | 0.115 |
| `info_only` | 0.000 | **0.000** | 0.293 | 0.293 |
| `graph_only` | 0.030 | 0.030 | 0.101 | 0.071 |

**It also reverses which fitter looks better.** For `oc_info`: under exact-zero `info_eff` the
alternative wins (0.301 vs 0.367); under a 1e-2 threshold it *loses* (0.416 vs 0.432). Both
fitters normalise each cluster's t-vector to unit norm, so a row's largest magnitude is
scale-free and comparable — every number in this report is quoted at both definitions, and the
1e-2 threshold is the honest one for the alternative.

## ⚠ 2. `is_dag` fabricates cycles from numerical dust

`is_dag` (helper.py:92) tests `A != 0` exactly. The alternative leaves off-mask entries at
**1e-16 … 1e-19**, so every `A_high` it returns registers as cyclic — including on OCC
partitions whose quotient is acyclic *by construction*. Thresholded at 1e-9, `A_high` recovers
exactly the structural mask and **is** a DAG:

    d=0.1  k=27  mask_edges=26   alt off-mask nonzero: 71 entries, median 2.1e-19, max 5.5e-17
           thresholded at 1e-9 -> 26 edges, dag True     (reg: 26 edges, dag True)

**This also affects the existing `data/fitcmp.csv`:** its `alt_dag = 0.242` against
`reg_dag = 1.000` is measuring dust, not cycles. `fitcmp.py` has the same un-thresholded call.
`lossfit.py` and `nochild_restarts.py` are patched; `lossfit.csv`'s `dag` column predates the
fix and should be ignored.


## ⚠ 3. The lambda_graph : lambda_recon RATIO, and why my first graph+recon arm was wrong

*`graph_recon.py`'s `g10_recon` (lambda_graph=10, lambda_recon=10) reported All-In 0.077 and
0/39 exact. That arm was mis-specified and the number is not a fact about graph+recon.*

**The loss terms are not on comparable scales.** Raw magnitudes at near-hard T_soft (p=50,
d=0.5):

| partition | k | graph | recon | info | recon/graph |
|---|---|---|---|---|---|
| singletons | 50 | 0.00134 | 0.0014 | 0.0006 | 1 |
| occ | 11 | 0.00077 | **32.04** | 6.10 | **41534** |
| one blob | 2 | 0.00013 | 4.13 | 2.00 | 31227 |
| random 5-way | 5 | 3.84565 | 5.70 | 0.98 | 1 |

`graph_loss` only becomes large (~3.8) at badly mismatched partitions; near a good one it is
~4e4 times smaller than `recon_loss`. So at 1:1 the recon term dominates completely, and
`g10_recon` was effectively `recon_only` with a decorative graph term -- which the partitions
confirm: k_used 45.6 vs 44.3 at d=0.1 and 14.4 vs 15.1 at d=0.9, All-In 0.077 vs 0.000.

**The ratio sweep** (`ratio.csv`, 39 instances paired against `lossfit.csv` and
`graph_recon.csv`; `lambda_one_child = 0` in every row):

| lambda_graph : lambda_recon | All-In | exact | ie_t2 | k | rel_err med |
|---|---|---|---|---|---|
| 0 : 1 (recon alone) | 0.000 | 0.000 | 0.076 | 30.3 | 0.325 |
| **1 : 1 (`g10_recon`)** | **0.077** | 0.000 | 0.047 | 31.8 | 0.201 |
| 100 : 1 | 0.359 | 0.205 | **0.043** | 32.6 | 0.031 |
| **1000 : 1** | **0.385** | 0.308 | 0.073 | 32.3 | **0.012** |
| 10000 : 1 | 0.333 | 0.333 | 0.089 | 32.6 | 0.025 |
| 1000 : 0 | 0.359 | 0.359 | 0.082 | 33.1 | 0.033 |
| 10 : 0 (`graph_only`) | **0.410** | 0.333 | 0.081 | 32.7 | 0.037 |
| 1000 : 0 + info=10 | 0.410 | 0.333 | 0.067 | 33.2 | 0.020 |

1. **1000:1 gives All-In 0.385 against 1:1's 0.077** -- a 5x swing from the ratio alone,
   matching the reported experience that lambda_graph=1000 / lambda_recon=1 satisfies All-In
   fairly often without necessarily reaching exactness (All-In 0.385, exact 0.308). It also has
   the lowest median rel_err of the family.
2. **The gain comes from recon no longer dominating, not from recon contributing.** Everything
   at 100:1 and above sits in the same 0.33-0.41 band as `graph_loss` ALONE (0.410 at
   lambda_graph=10). `graph + recon` ~ `graph`; recon adds nothing to All-In at any ratio.
3. **lambda_graph magnitude barely matters** -- 10 / 1000 / 10000 give 0.410 / 0.359 / 0.333,
   mildly DECREASING. The report's lambda_graph range (0-100) was not too low after all, at
   least with one-child off.

By density, 1000:1 gives All-In **0.57 at d=0.1 and 0.75 at d=0.3**, then 0.12 / 0.38 / 0.12 at
d = 0.5 / 0.7 / 0.9 -- a sparse-graph phenomenon, consistent with the restart study.

**New warning: never combine high lambda_graph with lambda_one_child=100.** It is the worst
configuration tested:

| arm | All-In | one_child | ie_t2 | rel_err med |
|---|---|---|---|---|
| `g1000_r1_oc` | 0.179 | 0.077 | 0.347 | 0.345 |
| `g1000_i10_oc` | 0.231 | 0.077 | 0.327 | 0.393 |

The terms pull in opposite directions -- graph_loss separates, one_child_loss merges -- and at
1000:100 neither wins cleanly, so the result satisfies neither condition and the abstraction
error blows up. Go high on lambda_graph only with one-child off.

---

## The one loss conclusion that changes: `lambda_graph`

`rerun_fixed.csv` put λ=1 marginally ahead of the recommended λ=0 after the edge_weight fix but
flagged it as "NOT yet established". The fitter switch settles it — the two changes compound.
(Hybrid retention, lower is better; OCC = 0.539 / 0.553.)

| α | reg: λ=0 | reg: λ=1 | alt: λ=0 | alt: λ=1 |
|---|---|---|---|---|
| 0.001 | **0.436** | 0.448 | 0.450 | **0.430** |
| 0.01 | **0.405** | 0.421 | 0.396 | **0.370** |
| 0.05 | 0.353 | 0.384 | 0.338 | 0.334 |

λ=1 is also the **more robust** arm under the conservative worst-of-three noise weighting that
`relerr_sensitivity.csv` argues for: acceptance 0.64 vs 0.59, and at p=50 its hybrid was
*completely unmoved* by the reweighting (0.289 → 0.289) where λ=0 degraded 0.287 → 0.323.

**Recommendation: `lambda_graph = 1`** (not 0, not 10) when shipping the alternative.

## The new frontier: α decides whether one-child is needed at all

Best arm by tolerance, alternative fitter, `info_eff_t2`:

| α | best arm | retention | vs OCC |
|---|---|---|---|
| 0.001 | `oc_info_g1` — one-child ON | 0.481 | −14% |
| 0.01 | `oc_info_g1` — one-child ON | 0.445 | −21% |
| **0.05** | **`graph_only` — one-child OFF** | 0.362 | −35% |
| 0.10 | `graph_only` — one-child OFF | 0.200 | −64% |

Below α ≈ 0.05 the one-child arms win; above it, dropping one-child wins decisively. And the
crossover is **density-dependent** — `graph_only` accepts 0.83–0.90 at d=0.1–0.3 but only
0.25–0.33 at d ≥ 0.5. It is a sparse-graph strategy.

---

## Why `graph_only` does not collapse into a blob

Loss terms at near-hard `T_soft`, same graph (p=50, d=0.5):

| partition | k | `graph` | `one_child` | `info` |
|---|---|---|---|---|
| singletons | 50 | **0.0013** | **69.52** | 0.0006 |
| occ | 11 | 0.0008 | 0.0000 | 6.10 |
| one blob | 2 | **0.0001** | 0.0000 | 2.00 |
| random 5-way | 5 | 3.8457 | 5.9598 | 0.98 |

`graph_loss` is multiplied by `co_assign`, so it charges only **co-assigned pairs with
mismatched child profiles** — it is ~0 at singletons *and* ~0 at the blob, making it a
**separating** pressure, never a merging one. `one_child_loss` is the sole merging term, and at
singletons it is 69.5 against graph's 0.0013. Set `lambda_one_child = 0` and nothing in the
objective wants a blob. The blob tracks one-child *enforcement* exactly:

| arm | one-child satisfied | largest cluster |
|---|---|---|
| `occ` | 1.00 | 74% |
| `oc_recon_g1` | 1.00 | 70% |
| `oc_info` | 0.57 | 56% |
| `oc_info_u100` (λ_info swamps it) | 0.07 | 35% |
| `info_only` (λ_oc = 0) | 0.00 | 30% |
| `graph_only` (λ_oc = 0) | 0.00 | **17%** |

This also explains why `graph_only` reaches All-In 40% of the time without one-child: **All-In
is vacuous at fine granularity** (singletons satisfy it trivially). `graph_only` lands at
k = 33.7, so All-In is nearly free there. It is not that `graph_loss` finds All-In; it is that
it lives where All-In is cheap. The price is compression — 97% of edge mass retained, but only
a 1.5:1 reduction in variable count, against OCC's k = 20.2.

## How often does `graph_only` find a partition admitting an exact abstraction?

Exactness scored as "either fitter found one" (a fitter failing is not proof none exists).

| arm | one-child | P(exact \| All-In holds) | P(exact \| All-In violated) |
|---|---|---|---|
| `graph_only` | 0.00 always | **0.84** (36/43) | 0.05 |
| `oc_info` | 0.55 | **1.00** (43/43) | 0.26 |
| `no_oc_g_info` | 0.00 always | 0.53 (38/72) | 0.00 |
| `info_only` | 0.00 always | — never holds | 0.02 |

* **One-child is not necessary** — 36 `graph_only` partitions had All-In, no one-child, and an
  exact abstraction.
* **All-In alone is not sufficient** — 7 of 43 All-In-holding `graph_only` partitions admitted
  no exact abstraction. Drop one-child and you lose the guarantee, not necessarily the object.
* At **coarse** granularity All-In looks nearly sufficient on its own (43/43 for `oc_info`); the
  gap against `graph_only`'s 0.84 fits the vacuity story — All-In on a k=33 partition is weak
  evidence, on a k=6 partition it is strong.

---

## "Drop one-child and restart until a run works" — tested

18 instances × 5 arms × 8 restarts (`nochild_restarts.csv`). Criterion `allin_exact` = All-In
holds AND an exact abstraction was found.

| arm | per draw | any-of-8 | if independent | never / mixed / always |
|---|---|---|---|---|
| `no_oc_graph` | 0.250 | **0.333** | 0.900 | **12 / 4 / 2** |
| `no_oc_g_info` | 0.264 | 0.389 | 0.914 | 11 / 3 / 4 |
| `oc_info` | 0.299 | 0.444 | 0.941 | 10 / 5 / 3 |
| `no_oc_g_recon` | 0.007 | 0.056 | 0.054 | 17 / 1 / 0 |

**Restarts do not behave like independent tries.** 8 draws bought +0.08, against the +0.65
independence predicts; 12 of 18 instances failed all 8 draws. This reproduces the earlier
finding (`restarts.csv`: per-draw 0.184, any-of-5 0.211, and 60 of 76 instances valid 0 times
out of 5) in the no-one-child regime and under the looser exactness criterion.

**The reason is density, not the optimiser.** Feasibility is a property of the instance:

| `allin_exact` per draw | d=0.1 | d=0.5 | d=0.9 |
|---|---|---|---|
| `no_oc_graph` | **0.75** | 0.00 | 0.00 |
| `no_oc_g_info` | **0.75** | 0.00 | 0.04 |
| `oc_info` | 0.00 | **0.46** | **0.44** |

At d=0.1 dropping one-child succeeds ~75% per draw, so restarts are barely needed; at d ≥ 0.5
no number of restarts helps. That is why there is almost no mixing — the instances partition by
density, not by luck. Init noise (0.1 vs 1.0) made no difference, matching REPORT.md's init
sweep.

### But the arms are complementary, and a PORTFOLIO works

`oc_info` and `no_oc_g_info` succeed on disjoint density regimes, so running both and keeping
the best qualifying draw nearly doubles coverage:

| strategy | coverage | best-draw retention | hybrid | vs OCC (0.629) | k |
|---|---|---|---|---|---|
| `oc_info` alone | 8/18 | 0.544 | 0.528 | −16% | 3.0 |
| `no_oc_g_info` alone | 7/18 | 0.035 | 0.471 | −25% | 43.6 |
| **`oc_info` + `no_oc_g_info`** | **14/18** | 0.266 | **0.373** | **−41%** | 23.2 |

Portfolio coverage by density: **1.00** at d=0.1, 0.50 at d=0.5, 0.83 at d=0.9.

Relaxing to the α=0.05 criterion instead of exactness, restarts *do* pay (9 of 18 instances
mixed for `no_oc_graph`) and the portfolio is dramatic:

| strategy | coverage | hybrid | vs OCC |
|---|---|---|---|
| `oc_info` alone | 18/18 | 0.278 | −56% |
| `oc_info` + `no_oc_g_info` | 18/18 | 0.087 | **−86%** |
| all three arms | 18/18 | 0.064 | **−90%** |

far above the ~28% REPORT.md currently claims for the refinement.

### `graph + recon` is the worst no-one-child arm, not the retention-preserving one

| arm | All-In | allin ∧ exact | instances with a hit | retention when accepted |
|---|---|---|---|---|
| `no_oc_g_recon` | **0.028** | **0.007** | **1/18** | 0.086 |
| `no_oc_graph` | 0.299 | 0.250 | 6/18 | 0.056 |
| `no_oc_g_info` | **0.500** | 0.264 | 7/18 | **0.001** |

Adding `recon` collapses All-In from 0.30 to 0.03 and does not even buy retention —
`graph + info` retains *more*. `no_oc_g_info` has the **highest All-In rate of any arm tested**,
one-child arms included. Use `graph + info`, not `graph + recon`.

---

## A practical warning about the switch

At the 1e-8 exactness bar, on **identical partitions**, the alternative finds far fewer exact
abstractions:

| arm | reg | alt |
|---|---|---|
| `occ` | 1.000 | 0.948 |
| `oc_info` | **0.621** | **0.155** |
| `oc_info_g10` | 0.759 | 0.414 |

Yet its α=0.05 acceptance *rises* (0.64 → 0.71) and its median `rel_err` is 6e-4. That pattern
points at SLSQP stopping on budget rather than a real modelling gap — `process_output` ships
`n_restarts=3, maxiter=100`, and the alternative optimises a residual where `fit_abstraction`
solves its equations analytically to machine precision.

**It is mostly budget, and `maxiter` is the binding knob** (`altbudget.csv`, 18 instances x 4
partitions, identical partitions at every budget):

| fitter | n_restarts | maxiter | exact | median rel_err | p90 rel_err | marg@1e-2 | secs |
|---|---|---|---|---|---|---|---|
| alt | 3 | 100 | 0.431 | 9.3e-5 | 0.102 | 3.83 | 0.98 |
| alt | 10 | 100 | 0.444 | 2.2e-5 | 0.076 | 4.01 | 3.16 |
| **alt** | **10** | **500** | **0.569** | **0.000000** | **0.019** | 5.93 | 10.2 |
| alt | 30 | 500 | 0.556 | 0.000000 | 0.018 | 6.19 | 28.5 |
| reg | 3 | 100 | 0.653 | 0.000000 | 0.138 | 6.51 | 1.35 |

Raising `maxiter` 100 -> 500 closes **62% of the exactness gap** (0.431 -> 0.569 against reg's
0.653), drives median `rel_err` to exactly zero, and cuts p90 by 5x. Extra *restarts* barely
help (0.431 -> 0.444 at maxiter=100) and 30 adds nothing over 10. Per arm, `oc_info`'s median
`rel_err` goes 0.0022 -> 0.000000 and `graph_only`'s 0.035 -> 0.0099.

**This corroborates the metric bug:** marginalisation at 1e-2 *rises* with budget, 3.83 ->
5.93, converging on `fit_abstraction`'s 6.51. The alternative approaches the same solution; at
low budget it has simply not pushed the offenders down yet -- which is exactly why it looked
"lossless" under the exact-zero test.

**Recommendation: `n_restarts=10, maxiter=500`** -- 10x the fit cost, and the residual 0.084
gap against `fit_abstraction` is real rather than budget.

## `graph + recon`, with and without one-child

`graph_recon.csv`, 39 instances. `lossfit.py` carried recon only at lambda_graph in {0,1} and
always WITH one-child, so these cells were missing.

| arm | one_child | All-In | valid | exact (reg) | k | intra | hyb@0.05 (alt) |
|---|---|---|---|---|---|---|---|
| `g1_recon_oc` | 100 | **0.872** | **0.872** | 0.897 | 3.2 | 0.549 | 0.547 |
| `g10_recon_oc` | 100 | 0.821 | 0.821 | **0.897** | 3.7 | 0.494 | 0.533 |
| `g10_rec_inf_oc` | 100 | 0.795 | 0.744 | 0.821 | 5.6 | 0.439 | 0.515 |
| `g10_recon` | **0** | 0.077 | 0.000 | **0.000** | 31.8 | 0.024 | 0.459 |
| `recon_only` | **0** | 0.000 | 0.000 | **0.000** | 30.3 | 0.031 | 0.510 |

* **`graph + recon + one_child` has the highest validity of any arm tested** -- 0.82-0.87 valid,
  0.90 exact, far above `oc_info`'s 0.41/0.64. But it collapses to k~3 and its retention is no
  better than OCC (0.533-0.547 against OCC's 0.590), so the validity buys nothing usable.
* **`graph + recon` without one-child at lambda_graph=10 is never exact: 0/39**, All-In 0.077.
  But this is a SCALE artifact, not a property of graph+recon -- at lambda_graph=1000 the same
  combination reaches All-In 0.385 and exact 0.308. See "The lambda_graph : lambda_recon RATIO"
  above. `no_oc_g_recon` in the restart study (All-In 0.028, 1/18 hits) used the same 1:1
  scaling and carries the same caveat.
* Adding `info` on top (`g10_rec_inf_oc`) is the best of the recon family -- validity 0.74,
  retention 0.515, and the highest acceptance of any one-child arm under the alternative (0.62
  at alpha=0.05) -- but `oc_info_g1` still retains more (0.434).

### The synthesis: validity and retention trade off almost monotonically

Alternative fitter, `info_eff_t2`, 39 shared instances:

| arm | validity | retention (hyb@0.05) |
|---|---|---|
| `g1_recon_oc` | 0.87 | 0.547 |
| `g10_recon_oc` | 0.82 | 0.533 |
| `g10_rec_inf_oc` | 0.74 | 0.515 |
| `oc_info_g1` | 0.49 | **0.434** |
| `oc_info` | 0.41 | 0.437 |
| `graph_only` | 0.00 | **0.360** |

**`recon` buys validity and costs retention; `info` buys retention and costs validity;
`graph_loss` buys retention without buying validity at all.** This resolves the contradiction
REPORT.md carries between `marginalisation.csv` ("recon is the better objective") and
`slack_nograph_robust.csv` ("info beats recon at every alpha") -- they sit at different points
on one frontier, and which wins depends on whether the scoring rule rewards validity (via a
fallback) or retention conditional on acceptance.

**Caveat on the tight-alpha winner.** On the 58-instance grid `oc_info_g1` wins at alpha <=
0.01; on this 39-instance subset `graph_only` wins there instead (0.509 vs 0.520). That
ordering is NOT stable between instance subsets. The lambda_graph = 1 vs 0 comparison **is**
stable -- g1 wins in both (0.481 vs 0.495 on 58 instances; 0.520 vs 0.533 on 39).

Note also that the alternative failed on 1 of 58 **OCC** partitions, where all three conditions
hold and an abstraction provably exists. It also finds abstractions `fit_abstraction` misses on
3–10% of partitions, so the two fitters are complementary rather than ordered — scoring
existence as "either fitter" is the right call.

Cost: the alternative is **1.8× slower on average** (2.36 s vs 1.27 s per fit) but degrades
badly with `k`: at p=100, d=0.1 with one-child off (k ≈ 100) a single restart study instance
took ~3 hours against ~6 minutes for the dense cells.

---

## What I would change in the recommendation

1. **`lambda_graph = 1`**, not 0 — the only conclusion that actually flips.
2. **Keep `lambda_info = 10` and `info` over `recon`** — both survive untouched.
3. **Score `info_eff` at a 1e-2 tolerance**, never exact zero, or the alternative's retention
   numbers are meaningless.
4. **Threshold `A_high` before `is_dag`** and re-run `fitcmp.py`'s dag column.
5. **DONE: `process_output` now defaults to `n_restarts=5, maxiter=500, seed=0`** and
   forwards all three (helper.py:471). `maxiter` is the binding knob -- 100 -> 500 drives
   median `rel_err` to exactly 0 and cuts p90 from 0.080 to 0.018 at every restart count --
   while restarts are inert: 3/5/10 give exact 0.514/0.458/0.528, non-monotonic, at
   3.9/6.2/12.1 s (`altbudget2.csv`). `n_restarts=5` verified safe against the spurious
   all-zero `A_high` optimum: 0 lost-edge failures over 39 OCC instances at 3, 5 and 10
   restarts, identical edge recovery, and exactness equal to 10 restarts
   (`restartcheck.csv`). Fit cost is ~4x the old (3, 100) default.
6. **Scale lambda_graph against lambda_recon by ~1e3-1e4, never 1:1** -- at 1:1 recon
   dominates by 4e4 and All-In collapses to 0.077. At 1000:1 it recovers to 0.385, which is
   what `graph_loss` alone already delivers, so `graph + info` remains the better pairing.
   Never combine high lambda_graph with one-child.
7. **Run a portfolio, not restarts** — `oc_info` + `no_oc_g_info`, keeping the best qualifying
   draw. Coverage 8/18 → 14/18, retention −16% → −41%.

## Caveats

* 58 instances for the loss grid; **18** for the restart study, so ~6 per density cell — the
  density split is the load-bearing claim there and rests on small n.
* `graph_type='random'`, n_high=10, slack 2 throughout; p ∈ {50, 100} only.
* Acceptance rates use the generator's noise weighting except where the worst-of-three is
  named; REPORT.md's warning that this weighting is the most favourable still applies.
* 2 of 60 loss-grid instances failed generation at `connected=True` and were dropped.
* `graph_recon.csv` covers 39 instances (seeds 0-3), a subset of the loss grid, so its OCC
  baseline (0.582-0.590) differs slightly from `lossfit.csv`'s (0.553-0.560).
* `altbudget.csv` / `altbudget2.csv` are 18 instances x 4 partitions; their exactness rates
  compare only across budgets, not against `lossfit.csv`.
* **Everything measured before the seed fix carries +-0.04 of fit noise** (see section 0). The
  paired design protects the FITTER comparison (both fitters saw identical partitions) but not
  single-run comparisons between budgets or arms.
* `restartcheck.csv` tests the all-zero `A_high` risk on OCC partitions only; the original
  3/120 failures in `fitcmp.csv` were on trained partitions, so it bounds rather than settles
  that risk.
* The tight-alpha ranking between `graph_only` and `oc_info_g1` is not stable across instance
  subsets (see above).

## Random-init restarts: do independent restarts buy anything?

`scripts/restarts_randinit.py` -> `data/restarts_randinit.csv` (240 runs, 19.8 min).
3 densities x 11 instances x {lam_oc=0, 1000} x 10 **independent random-init** restarts.
The 11 instances are exactly the seeds/k of the first 3-4 per density in `frontier2.csv`,
so the comparison against AOC-Closure init is **paired on the same instances at the same
budget** (verified: k matches instance-for-instance).

Scored the way the paper scores: accept a run iff RAE <= alpha, then minimise IEM, else
fall back to closure. (The log's own summary picks best-RAE and best-IEM *independently*,
which can be different restarts and is not an attainable operating point -- ignore it.)

Matched, alpha = 0.05, median over instances:

| lam_oc | d | OCC | AOC init, 1 run | random init, best-of-10 |
|---|---|---|---|---|
| 0 | 0.1 | 0.381 | acc 1.00, IEM 0.366 | acc 1.00, IEM **0.020** |
| 0 | 0.5 | 0.651 | acc 0.00, IEM 0.651 | acc 0.00, IEM 0.651 |
| 0 | 0.9 | 0.849 | acc 0.00, IEM 0.849 | acc 0.00, IEM 0.849 |
| 1000 | 0.1 | 0.381 | acc 1.00, IEM 0.274 | acc 1.00, IEM 0.272 |
| 1000 | 0.5 | 0.651 | acc 1.00, IEM 0.636 | acc 0.75, IEM 0.634 |
| 1000 | 0.9 | 0.849 | acc 1.00, IEM 0.783 | acc 1.00, IEM 0.748 |

**Restart variance under random init is large, and best-of-n does exploit it.** Median
accepted restart -> best accepted restart: 0.159 -> 0.020 (d=0.1, lam=0), 0.775 -> 0.748
(d=0.9, lam=1000). So restarts are not wasted motion; they are recovering from bad draws.

**But they do not beat AOC-Closure init.** Per-instance at lam_oc=1000, AOC-init-1-run vs
random-init-best-of-10 is 4 wins / 3 wins / 4 ties -- a wash at 10x the compute. The one
decisive win for random init is lam_oc=0 at d=0.1 (2 of 3 instances: 0.366->0.020,
0.381->0.040), where AOC init is visibly trapped near its own partition.

**Acceptance is where init matters most.** Per-run P(RAE <= 0.05):

| lam_oc | d | AOC init | random init |
|---|---|---|---|
| 0 | 0.1 | 0.95 | 0.67 |
| 0 | 0.5 | 0.25 | **0.00** |
| 0 | 0.9 | 0.00 | 0.00 |
| 1000 | 0.1 | 1.00 | 1.00 |
| 1000 | 0.5 | 0.95 | **0.30** |
| 1000 | 0.9 | 1.00 | 0.75 |

At lam_oc=0, random init reaches RAE <= 0.05 in **0 of 80 runs** at d >= 0.5. A single
AOC-init run beats random init's best-of-10 on acceptance at d=0.5. Closure init is not
just a speedup, it is what makes the exactness-seeking arm feasible off low density.

**Conclusion: keep AOC-Closure init and do not sell restarts as the lever.** lam_oc is the
lever. The honest caveat for the paper is the d=0.1 / lam_oc=0 trapping: closure init costs
information loss there, which is exactly the regime where lam_oc=0 is otherwise attractive.

---

## 2026-09-23 — T-faithfulness: `faithful=True` is now the DEFAULT

**BREAKING for reproduction.** `process_output`, `fit_abstraction` and
`fit_abstraction_alternative` now default to `faithful=True`. Every CSV generated before
this date used the old objective. Pass `faithful=False` to reproduce them.

### What we found

On the **ground-truth partition**, where an exact abstraction provably exists, all three of
(generating abstraction, `fit_abstraction`, `fit_abstraction_alternative`) achieve
**RAE = 0.000** — yet they reconstruct A at 0.000, 0.08-0.39 and 0.42-0.96 relative error
respectively. Same partition, same zero abstraction error, completely different abstractions.

The cause is stated in the code itself (helper.py, in the constraint block): the residual
has a SOLUTION SET, and it "cannot tell a T-faithful abstraction from a T-unfaithful one --
both are exact zeros of this objective (e.g. Y2 = X2 - X3 cancelling X1's effect is just as
valid a root as the faithful completion)". `fit_abstraction_alternative` was landing on the
cancelling roots: measured against the quotient of A_low it **dropped 62-89% of the true
cross-cluster edges**, and at d=0.1 returned ||A_high|| = 0.99 against a true 3.82 -- a
nearly edgeless high-level model that is still a perfectly valid exact abstraction.

The old `||A_high||` tie-break could not fix this because it is **selection among the random
restarts that happened to be drawn, not search**. If no restart lands faithful, no tie-break
helps.

### The change

Swap the roles of objective and constraint:

| | old (`faithful=False`) | new (`faithful=True`) |
|---|---|---|
| objective | minimise residual | **maximise \|\|A_high\|\|^2** |
| constraint | per-cluster unit norms | residual <= rtol, **plus** per-cluster unit norms |
| selection | largest \|\|A_high\|\| among tied restarts | (not needed) |

The residual enters as an INEQUALITY, not an equality: a sum of squares has vanishing
gradient on its own solution manifold, which SLSQP handles badly as an equality.
`fit_abstraction` gets the same treatment, maximising \|\|A_high\|\|^2 subject to its
multi-child penalty; there only the SINK t-vectors are free, so the search is over
sum(\|sink clusters\|) parameters rather than p. Both fall back to the old path if no restart
is feasible, so the new mode should never return worse than the old one.

### Measured (ground-truth partition, blockiness=1, heterogeneity=0, p=100)

Reconstruction \|\|A - A_hat\|\|^2 / \|\|A\|\|^2, and seconds:

| d | prop default | prop faithful | alt default | alt faithful |
|---|---|---|---|---|
| 0.1 | 4.8e-01 (0.0s) | **4.4e-11** (0.9s) | 1.0e+00 (0.9s) | **3.6e-11** (0.7s) |
| 0.5 | 5.1e-01 (0.0s) | **5.9e-12** (1.6s) | 9.9e-01 (1.4s) | **1.8e-12** (3.1s) |
| 0.9 | 1.1e-01 (0.5s) | **4.3e-12** (2.8s) | 9.9e-01 (1.5s) | 4.6e-08 (9.1s) |

8-11 orders of magnitude better reconstruction, at 2-6x the cost. RAE stays 0 throughout, so
this is a pure gain, not a trade-off. **Propagation + faithful is the better of the two** --
more accurate at high density and ~3x faster -- but it needs an acyclic quotient
(`reverse_topo_order`), which is why the alternative exists at all.

### Caveats worth carrying

* **The binary T-faithfulness criterion is necessary but not sufficient.** `fit_abstraction`
  satisfies "every cross-cluster edge survives into A_high" with zero violations and still
  reconstructs at 0.06-0.51 -- its edges are present but wrongly scaled. Reconstruction is
  zero only when (T, A_high) matches the generating pair up to a per-cluster rescaling.
* **The definition in the paper is broader than its prose gloss.** It quantifies over all
  (r, s), so 22-45% of the pairs it constrains are WITHIN-cluster and 3-32% are cross-cluster
  but reachable only indirectly -- only 34-74% are the cross-cluster direct edges the
  intuition describes.
* **Formal "!= 0" vs numerics is unresolved.** The unfaithful fits leave \|(ET)_rj\| at
  ~1e-8, which does not shrink when maxiter goes 100 -> 10000. That is consistent both with
  a true zero floored by a squared objective (sqrt(eps) ~ 1.5e-8) and with genuine
  attenuation; these runs cannot separate them. A relative condition
  \|(ET)_rj\| >= c * max_s \|e_rs t_sj\| would sidestep the question entirely.
* **The faithful fit is not bulletproof**: one instance (d=0.9, seed 2) reconstructed at
  0.347 with 20/38 near-empty edges.
* Scripts: `scratchpad/faithful_fit.py`, `prop_faithful.py`, `tfaith.py` (session scratch).

---

## 2026-09-23 (later) — three-way comparison: `fit_abstraction` vs `_alternative` vs `_K`

The section above compared two fitters on ground-truth partitions and concluded
"propagation + faithful is the better of the two". `fit_abstraction_K` did not exist then.
With it in the picture, and with the **cyclic** case added, that recommendation changes.

Script `scripts/fit3cmp.py`, data `data/fit3cmp.csv`, figure `figures/fit3cmp.png`.
120 instances: {acyclic, cyclic} x {p=24 k=4, p=40 k=6} x {ground-truth partition, random
partition} x 15 trials, `n_restarts=10`, all three fitters on identical inputs. No crashes.

### Two things had to be fixed before the comparison could be run

**1. `scripts/cyclic.py`'s generator cannot answer this question.** It adds reversed edges to
a finished `A_low`, which destroys the generating abstraction — so there is no ground truth
to hit and "did it reach RAE 0" is unanswerable. The new generator instead makes `A_high`
cyclic **first** (rho = 0.7) and runs `T_concretisation_sampling`, which is exact by
construction regardless of cycles: verified at `||E_low T - S E_high||_max ~ 3e-15`.

The algebra is worth stating, because it shows acyclicity was never needed. Exactness is
`(I - A_low) S = T (I - A_high)`. Block `(i, j)` of the left side is
`-A_low[pi_i, pi_j] S_jj` for `i != j`, so the requirement is
`A_low[pi_i, pi_j] E_jj t_j = A_high[i,j] t_i`, and the generator satisfies it by
construction via `c . s_j = 1`. Nothing in that argument uses a topological order.

**2. `get_S` is wrong on cyclic quotients** — see the new REPORT.md section. It was
contaminating the cyclic cells of this very comparison, so everything below is scored with
the corrected S.

### Results

Median RAE / fraction reaching RAE < 1e-6 / fraction fully T-faithful / median seconds.
"T-faithful" = every cross-cluster block of `A_low` appears as a nonzero `A_high` edge.

| cell | quotient cyclic | `fit_abstraction` | `_alternative` | `_K` |
|---|---|---|---|---|
| acyclic, GT, p=24 | 0% | **1e-30** / 1.00 / 1.00 / 0.2s | 4e-08 / 0.73 / 1.00 / 4.2s | 4e-12 / 1.00 / 1.00 / 0.9s |
| acyclic, GT, p=40 | 0% | **3e-10** / 1.00 / 1.00 / 3.2s | 3e-05 / 0.27 / 1.00 / 4.3s | 5e-12 / 1.00 / 1.00 / 1.6s |
| cyclic, GT, p=24 | 100% | 2e-01 / 0.00 / **0.00** / 0.3s | 2e-07 / 0.80 / 1.00 / 3.6s | **9e-12** / 0.93 / 1.00 / 0.8s |
| cyclic, GT, p=40 | 100% | 3e-01 / 0.00 / **0.00** / 0.9s | 4e-05 / 0.27 / 1.00 / 5.1s | **4e-11** / 0.93 / 1.00 / 1.6s |
| acyclic, random | 87-100% | 2e-01 / 0.13 / 0.00 / 5.5s | 7e-05 / 0.33 / 0.93 / 7.8s | **2e-12** / 1.00 / 0.33 / 1.7s |
| cyclic, random | 100% | 8e-01 / 0.00 / 0.00 / 7.1s | 1e-01 / 0.00 / 1.00 / 7.0s | 1e-01 / 0.00 / 1.00 / **1.4s** |

The oracle (generating abstraction) scores RAE ~1e-30 in every GT cell, so the ceiling is
real and the metric is calibrated.

### What changes

**`fit_abstraction_alternative` does not reach zero even on the ground-truth acyclic
partition.** Median RAE 4e-08 at p=24 and 3e-05 at p=40 — only 27% of p=40 runs clear 1e-6.
The `faithful=True` formulation stops as soon as the residual is inside `rtol`; it does not
drive it to zero. The earlier section reported "RAE stays 0 throughout", measured at a
coarser tolerance. The **reconstruction** claim from that section is unaffected; the
abstraction error itself is not actually zero.

**`fit_abstraction` cannot be T-faithful on a cyclic quotient, structurally.**
`high_level_mask` runs `enforce_dag` before the fit begins, so cycle edges are deleted a
priori: recall 0.63-0.68, **0/30 faithful**, RAE pinned at ~0.25. This is not a tolerance or
restart issue and no amount of budget fixes it. The same mechanism explains the random-
partition column, where the quotient is cyclic 87-100% of the time and recall sits at a
chance-like 0.44-0.50. The previous recommendation ("propagation + faithful is the better of
the two") holds **only** for acyclic quotients — where it is still the most accurate and the
fastest at p=24.

**`fit_abstraction_K` is the new default recommendation.** It is the only fitter that is both
exact and fully T-faithful on both ground-truth families, and it is the fastest in every cell
except small acyclic: 0.3-1.7s median, 3-5x faster than `_alternative`. On the cyclic GT
partition it reaches 4e-11 where `fit_abstraction` sits at 3e-01 — twelve orders of
magnitude, from a structural limitation rather than a numerical one. Its docstring's claim
that measuring the residual in un-inverted coordinates avoids the `cond(E_ii)^2` distortion
is borne out.

### Caveats

* **`_alternative` inflates `A_high` on large cyclic instances.** At cyclic p=40 GT it
  returns 29 edges against a 17.3-edge target (11.8 spurious even at a relative threshold);
  `_K` gives 2.3. This is the direct cost of `maximise ||A_high||^2` — it buys faithfulness
  (no dropped edge, ever) by pushing mass into edges that should not exist.
* **`_K`'s faithfulness on random partitions is threshold-sensitive.** Recall 0.89 at
  tol=1e-8 but 0.49 at tol=1e-6*max on acyclic/random: it keeps real edges at magnitudes
  near the numerical floor. This is the "formal != 0 vs numerics" caveat from the previous
  section, reappearing. It does **not** occur on ground-truth partitions.
* **Dense `A_high` from `_K` on cyclic GT is numerical dust, not an extraction error.** Checked
  directly: its `T` matches `T_gt` at per-cluster |cos| = 1.0000 and its `A_high` matches the
  diagonally-similar implied matrix to 2.8e-6. The 1e-8 absolute cut counts that dust as
  edges; the relative cut does not.
* Both `_alternative` and `_K` are scored here at `n_restarts=10`. `process_output` defaults
  to 5, and `cyclic.py` uses 3.

---

## 2026-09-23 — TESTED AND REJECTED: a soft `lambda * ||E_high||^2` term in the objective

**Question.** `fit_abstraction_K` returns an exact abstraction for essentially *any* partition
on an acyclic graph, including random ones, by marginalising the cross-cluster emitters (see
REPORT.md). Would adding a soft norm reward to the residual — `residual - lambda*||offdiag(M)||^2`,
where `M = D^-1 E_high`, so this is `||E_high||` in per-cluster-normalised form — stop bad
partitions from scoring RAE = 0, while keeping `A_high` reflective of the low-level structure?

**Answer: no. Do not add it.** `scripts/lambda_norm_test.py`, `data/lambda_norm_test.csv`,
acyclic graphs, 10 trials per cell, `n_restarts=10`.

Fraction reaching RAE < 1e-6 / median RAE / marginalisation rate / cross-cluster recall:

| p | partition | lambda=0 | 0.01 | 0.1 | 1.0 | 10.0 |
|---|---|---|---|---|---|---|
| 24 | **GT** | **1.0** / 0e0 | 0.4 / 0e0 | 0.3 / 3e-4 | 0.3 / 1e-2 | 0.3 / 4e-2 |
| 24 | random | 1.0 / 0e0 | 0.3 / 1e-4 | 0.2 / 0.27 | 0.2 / 0.69 | 0.2 / 0.71 |
| 40 | **GT** | **1.0** / 0e0 | **0.0** / 5e-2 | 0.0 / 0.17 | 0.0 / 0.35 | 0.0 / 0.35 |
| 40 | random | 1.0 / 0e0 | 0.1 / 8e-3 | 0.1 / 10.0 | 0.0 / 87.5 | 0.0 / 27.6 |

**1. It destroys exactness on GOOD partitions, at any lambda worth having.** At lambda = 0.01
— the smallest value tested — the ground-truth exact rate falls from 1.00 to 0.40 at p=24 and
to **0.00** at p=40. The residual near its solution manifold is ~1e-12 while the norm reward
is O(1), so the optimiser will always sell a "free" residual increase for norm. That is
intrinsic to a *soft* penalty here, not a tuning failure.

**2. It does not separate good partitions from bad ones — and at p=40 it INVERTS the
ordering.** At lambda=0.01, p=40: GT median RAE **0.047**, random median RAE **0.008**. The
ground-truth partition scores *worse* than the random control. At p=24 the exact rates are
0.4 (GT) vs 0.3 (random) — noise. There is no lambda at which RAE becomes a partition-quality
signal.

**3. It does fix the structural pathology — but the hard-constraint version already does that,
better.** Marginalisation drops 0.413 -> 0.018 and recall rises 0.666 -> 0.971 (p=40 random,
lambda=0.01), and GT reconstruction improves 0.760 -> 0.627 at p=24. But
`fit_abstraction_alternative`'s `faithful=True` — *maximise* the norm **subject to** the
residual, the constrained form — gets the same structural benefit (marginalisation 0.032,
recall 0.99) while retaining far more exactness. If faithfulness is what you want, port that
constrained formulation to `_K`; do not soften it into the objective.

**4. At larger lambda the fits become garbage.** Reconstruction on random partitions reaches
4.6, 17.4, 2.6 against a "predict no edges" baseline of 1.0 — an order of magnitude *worse*
than predicting nothing.

### Consequence for the framing

RAE cannot be made into a partition-quality signal by reweighting the objective. The exact-
abstraction solution set is a property of the **partition**, and both the soft term (tested
here) and the hard constraint (`_alternative`, measured earlier: still RAE < 1e-6 on 87% of
random partitions at p=24) merely pick a different point inside it. They do not shrink it.

So an "exactness vs information" trade-off is not the right frame for the acyclic case: on
acyclic low-level graphs exactness is close to free, and the real problem is
**minimise information loss subject to exactness**. Report `info_loss` / `info_effective` and
T-faithfulness as the quality measures, with RAE as a feasibility check rather than a score.

One caveat before adopting that framing wholesale: exactness is *not* free in general. On
**cyclic** random partitions `_K`'s exact rate is 0.40 (p=24) and 0.00 (p=40) — marginalisation
is unavailable there (emitter-marginalisation rate 0.000-0.065) and exactness fails with it.
The "fitter always returns an exact abstraction" premise holds for acyclic low-level graphs,
not universally.

---

## 2026-09-24 — ⚠ blockiness=0.5 / heterogeneity=1 CANNOT DISCRIMINATE THESE FITTERS

**Read this before trusting any reconstruction number in the 2026-09-23 three-way comparison
above.** That sweep ran at `blockiness=0.5, heterogeneity=1.0`. In that setting the ORACLE —
the generating `(A_high, T)` pair itself — reconstructs at **0.87-0.95**, against a
"predict no edges" baseline of 1.0. There is no headroom, so every fitter scores ~0.9 and the
comparison measures the ceiling, not the fitters.

`reconstruction_loss` builds `A_hat = T A_high pinv(T.T)^T`, whose `(i,j)` block is
`A_high[i,j] t_i t_j^T / ||t_j||^2`. With `heterogeneity=0, internal_density=0, blockiness=1`
the concretisation produces exactly that (`s_j = t_j`, `c = t_j/||t_j||^2`), so the oracle hits
~1e-16 and the fitters spread over six orders of magnitude. **That is the regime to compare
in** — and it is the one the earlier 2026-09-23 T-faithfulness entry used.

Measured (`scripts/recon_blockiness.py`, GT partitions, medians over 10 trials):

| kind | p | blockiness / heterogeneity | oracle | `_K` | `_alternative` | `fit_abstraction` |
|---|---|---|---|---|---|---|
| acyclic | 24 | 0.5 / 1.0 | 0.870 | 0.830 | 0.723 | 0.723 |
| acyclic | 24 | **1.0 / 0.0** | **1.1e-16** | **0.496** | 4.3e-07 | 3.3e-08 |
| acyclic | 40 | 0.5 / 1.0 | 0.940 | 0.892 | 0.902 | 0.859 |
| acyclic | 40 | **1.0 / 0.0** | **2.0e-16** | **0.667** | 4.4e-07 | 6.9e-07 |
| cyclic | 40 | **1.0 / 0.0** | 1.7e-16 | 8.5e-07 | 1.1e-06 | 0.282 |

Only RAE and runtime from the earlier sweep are unaffected; its reconstruction and
faithfulness columns are weak evidence and should not be cited.

### Consequence: `fit_abstraction_K` has a real defect on acyclic ground-truth partitions

At `blockiness=1, heterogeneity=0` it reconstructs at **0.496 (p=24)** and **0.667 (p=40)**
while reaching RAE 7e-15 and recall 1.000 — every edge present, all of them wrongly scaled.
At p=40 it also drops to **recall 0.850**, so it does lose ground-truth edges after all; the
"recall 1.000 on 60/60 GT instances" reported on 2026-09-23 holds only in the forgiving
setting.

Cause (`scripts/probe_K_restarts.py`): `_K` has no search for a faithful solution, only the
tie-break `tie_tol = best*1e-3 + 1e-8` then max-norm among tied restarts. The window is not
too narrow — with `best_res ~ 1e-13` it is ~1e-8 absolute and 7-10 of 10 restarts qualify —
but **selection among the restarts that happened to be drawn is not search**, the same
failure the 2026-09-23 entry identified for `_alternative`. Measured headroom: a higher-norm
restart existed outside the window in 29/120 instances, and the chosen `||A_high||` fell short
of the pool's best by up to 4.3.

### The fix: `faithful=True` for `_K`, same constrained form as `_alternative`

Maximise `||A_high||^2` subject to per-cluster unit norms (equality) and residual <= rtol
(inequality). `scripts/kfaith.py`, results `data/kfaith_b1.csv`, GT partitions,
`blockiness=1, heterogeneity=0`:

| kind | p | `_K` recon | **`_K` faithful** | `_alternative` | `_K` recall | **`_Kf` recall** |
|---|---|---|---|---|---|---|
| acyclic | 24 | 0.496 | **1.0e-06** | 4.3e-07 | 1.000 | 1.000 |
| acyclic | 40 | 0.667 | **1.2e-06** | 4.4e-07 | 0.850 | **1.000** |
| cyclic | 24 | 9.7e-07 | 1.5e-05 | 1.3e-04 | 1.000 | 1.000 |
| cyclic | 40 | 8.5e-07 | 9.5e-06 | 1.1e-06 | 1.000 | 1.000 |

Five to six orders on acyclic, and recall restored at p=40. RAE degrades from 1e-15/1e-13 to
1e-12/1e-10 — still far below any threshold. Runtime 1.1-3.1s against `_K`'s 0.21-0.41s
(5-7x) and `_alternative`'s 0.73-1.42s.

`_alternative` is already comparable on acyclic (4.4e-7) and faster, so faithful-`_K` is
"best of both" — `_K`'s cyclic robustness and exactness plus correct scaling — rather than
uniquely necessary.

CAVEAT measured at `blockiness=0.5`: there the faithful branch fails feasibility at p=40 and
falls through to plain `_K`, returning bit-identical output for 6-9x the runtime. Normalising
the residual (`.mean()` instead of `.sum()`) does NOT fix that — the residual AT THE OPTIMUM is
4e-3 to 75, three to five orders above any usable tolerance, i.e. SLSQP walks off the feasible
manifold chasing norm. So the constrained form is regime-dependent, and a silent fallback
should be logged rather than hidden.

### MERGED 2026-09-24 — `fit_abstraction_K(..., faithful=True)` is now the DEFAULT

**BREAKING for reproduction**, the same way the 2026-09-23 switch was for the other two
fitters. Every CSV generated before today used the old path, including this session's
`fit3cmp.csv`, `randpart_probe.csv`, `emit_probe.csv` and `probeK.csv`. Pass `faithful=False`
to reproduce them.

    fit_abstraction_K(A_low, E_low, T_hard, n_restarts=10, seed=None, maxiter=100,
                      faithful=True, rtol=1e-9, verbose=False)

* `faithful=True` maximises `||A_high||^2` subject to residual <= `rtol` and the per-cluster
  unit norms. It reuses the existing `objective_and_grad` as the inequality constraint, and
  the torch objective reuses the projection coefficients the residual already forms, so the
  extra cost per call is one `k x k` inverse.
* Feasibility is checked on **both** constraints. The residual divides by `||E_ii t_i||^2` and
  is therefore scale-invariant, so an inflated `T` passes it alone -- the same guard
  `_alternative` needed.
* Falls back to the residual-minimising path when no restart is feasible. `verbose=True`
  reports either `n/N feasible restarts` or the fallback, so the regime-dependent no-op
  described above is visible rather than silent.
* `faithful=False` is the old path, unchanged.

Verified against `scripts/kfaith.py`: identical to 1.6e-17, and `faithful=False` reproduces
the pre-merge numbers exactly (0.496 / 0.667 / 9.73e-07 / 8.51e-07).

---

## 2026-09-24 — `fit_abstraction_auto`: branch on the quotient  [RECOMMENDED ENTRY POINT]

No single fitter works everywhere, and the property that decides which one to use is
**whether the quotient (cluster graph) is acyclic**. Measured reconstruction on ground-truth
partitions at `blockiness=1, heterogeneity=0` (`scripts/fastfix.py`, oracle ~1e-16):

| quotient | p | `fit_abstraction` | `_alternative` faithful | `_K` faithful | `_K` default |
|---|---|---|---|---|---|
| acyclic | 24 | **1.9e-07** | 6.5e-07 | 2.6e-06 | 4.6e-01 |
| acyclic | 40 | **6.9e-07** | 6.4e-07 | 1.2e-06 | 6.7e-01 |
| acyclic | 60 | **1.3e-07** | 1.7e-05 | 1.4e-06 | 6.0e-01 |
| acyclic | 100 | **2.6e-07** | 7.6e-01 | 6.6e-01 | 6.6e-01 |
| cyclic | 24 | 2.1e-01 | 3.6e-06 | **9.4e-06** | 9.7e-07 |
| cyclic | 40 | 2.6e-01 | 1.1e-06 | **7.4e-06** | 8.5e-07 |
| cyclic | 60 | 2.4e-01 | 2.5e-06 | **1.2e-06** | 3.8e-01 |
| cyclic | 100 | 2.6e-01 | 1.1e-03 | **1.2e-06** | 1.3e-06 |

`fit_abstraction` is the best solver on acyclic quotients at every size and the **only** one
still working at p=100 -- both `_alternative` (0.76) and `_K` faithful (0.66) collapse there.
It is also the cheapest, because only the SINK t-vectors are free. It cannot touch cyclic
quotients: `high_level_mask` runs `enforce_dag`, deleting the cycle edges before the fit
begins (0.21-0.26 everywhere).

### The API

    quotient_mask(A_low, T_hard)       # raw (k,k) cluster graph -- NOT dag-enforced
    is_quotient_acyclic(A_low, T_hard) # is_dag(quotient_mask(...)), O(p^2 + k^3)
    fit_abstraction_auto(A_low, T_hard, E_low=None, n_restarts=10, seed=None,
                         maxiter=100, verbose=False, return_method=False)

`quotient_mask` has to exist separately because **`high_level_mask` cannot detect a cycle** --
it runs `enforce_dag`, so it always reports a DAG. Measured on one cyclic instance:
`quotient_mask` 6 edges / `is_dag=False`, `high_level_mask` 5 edges / `is_dag=True`.

The dispatcher also guards against a silently degenerate propagation result: an edgeless
`A_high` on a quotient that demonstrably has edges is the annihilating solution, not an
abstraction, so it falls through to `_K` rather than returning it. `return_method=True`
reports which solver ran, worth recording per instance in an experiment loop.

### Result: oracle-matching at every size, both regimes

Ground-truth partitions, `blockiness=1`, reconstruction (oracle = 0.000):

| | p=24 | p=40 | p=100 |
|---|---|---|---|
| acyclic GT | **0.000** (propagation) | **0.000** (propagation) | **0.000** (propagation) |
| cyclic GT | **0.000** (K faithful) | **0.000** (K faithful) | **0.000** (K faithful) |

Branch selection matched `is_quotient_acyclic` on 72/72 instances.

CAVEAT: on bad/random partitions the quotient is cyclic 87-100% of the time, so the dispatcher
routes to `_K` faithful, which is the slow path AND cannot succeed there (reconstruction
0.90-1.00 -- correctly so, the partition is bad). Measured 15-75s per fit at p=40-100 under
6-way contention. If you are SCANNING many candidate partitions rather than scoring a final
one, call `fit_abstraction_K(..., faithful=False)` directly for the scan and reserve the
dispatcher for the partitions you keep.

### Rejected on the way here

* **`faithful=False` + more restarts.** 10 -> 100 restarts moves recovery from 0.25 to 0.30 at
  p=24 and 0.00 to 0.00 at p=40 (20 seeds, GT partitions, `blockiness=1`).
* **Selecting the restart with the lowest `reconstruction_loss`** (legal -- it needs no ground
  truth). Returns EXACTLY what the `||A_high||` tie-break already returns, to four decimals, in
  every cell: the pool contains no good solution to select. At p=40, 100 restarts costs 9.3s
  and recovers 0%; `faithful=True` with 10 restarts costs 8.2s and recovers 100%.
* **Emission-maximising initialisation** (start `t_i` at the top right-singular vector of
  cluster i's stacked outgoing blocks, the direction of maximal emission, one small SVD per
  cluster). Reconstruction 1.0 -- fully vacuous -- in almost every cell. The optimiser walks
  straight back out to an annihilating solution, so a better starting point does not help.
