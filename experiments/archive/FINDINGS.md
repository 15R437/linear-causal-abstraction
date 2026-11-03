# How we got here: partitions and T-fitting for linear causal abstraction

*2026-09-09 → 09-30. This file is the narrative: what we tried, what each step taught us, and
why the final pipeline looks the way it does. The evidence lives in the detailed reports,
in `scripts/` (one file per question) and in `data/`. See [README.md](README.md) for the index.*

## Where we ended up

```
A_low ─┬─ OC Closure     (one-child; best of d draws on information loss)
       ├─ AOC Closure    (one-child + All-In by construction → RAE = 0 guaranteed)
       └─ Greedy All-In  (merge to budget k, minimising All-In violations)
                │  each candidate partition π
                ▼
   fit_abstraction_eigen  (src/algos/T_fitter.py)
     cyclic SCCs of the quotient → t pinned as the top eigenvector of the cycle product
     sinks → optimised (SLSQP);   everything else → propagated from its children
                │
                ▼
   accept  iff  RAE ≤ α  and  ReconErr < ReconErr(AOC Closure);  otherwise fall back to AOC Closure
```

Partition learners: `src/algos/greedy.py`. Paper: `paper/alg_partition.tex`,
`paper/alg_eigenfit.tex`, `paper/pipeline_figure.tex`.

---

## 1. The starting point: a learned partition (HailMary, later renamed GradCAP)

We started with HailMary, which learns a soft partition T by gradient descent on a sum of
losses (reconstruction, graph/All-In, one-child, surjectivity, information). A partition
admits an exact abstraction when two **sufficient** conditions hold:

* **All-In**: for every high-level edge X→Y, every member of X has an edge into Y.
* **One-child**: every cluster has at most one child cluster.

The opening question was narrow: would a surjectivity loss or a better initialisation help?
(`smoke_test_surj_init.py`, `analyse_*.py`). The answer was no to both. But it
established the success criterion: over 2000 runs, P(exact | both conditions) = **1.0000**.
The raw notes from that phase are in
[results/…/FINDINGS.md](../../results/smoke_tests/20260911-235555_FINAL/FINDINGS.md).

## 2. The structure of the problem matters more than the optimiser  (09-11 → 09-13)

* **One-child is what makes All-In satisfiable.** Without it, All-In held in 0/80 runs at
  any loss weight. All-In alone is vacuous, since singletons satisfy it.
* **On DAGs, an acyclic quotient follows from the two conditions** (155/155). In the cyclic
  case it has to be imposed separately.
* **Utility is information loss**, the fraction of |A| edge mass inside clusters.
  Reconstruction error cannot measure quality under one-child, because the quotient has
  ≤ k−1 edges. Information loss must be read *conditional on validity*: failed runs score
  better on it, which invalidated four of our own early claims.

## 3. A combinatorial method beats the learned one  (09-13 → 09-14)

Merging children until every cluster has ≤ 1 child (the **one-child closure**), plus a
targeted-merge fallback, gave **100% exact recovery** across density 0.1–0.9 and p ≤ 400.
Existing baselines reached 0–12.5%. It runs in 0.14 s at p=200.

HailMary on top of the closure was a strictly non-negative refinement, worth only
~0.02–0.03 information at ~200× the runtime. It never rescued an instance the closure
failed on (0/35). On random DAGs the partitions are **one blob plus singletons**. Brute
force at p ≤ 10 shows this is largely forced by the conditions, not an artefact of the
method. → [REPORT.md](REPORT.md) Acts I–VI.

## 4. Relaxing exactness: the α-dial  (09-14 → 09-17)

* The conditions are sufficient, **not necessary**. About 44% of All-In-violating
  partitions are still exact, because the fitter *marginalises* the offending nodes.
* Tolerating a relative error α opens a smooth trade-off. At α = 0.05, retention is 34%
  better than the closure, with the closure as a safe fallback. This produced the
  **accept-if-RAE ≤ α-else-fall-back** rule that the final pipeline still uses.
* **`graph_loss` had a bug** (the `edge_weight` term, fixed 09-15): its correlation with
  All-In violations was ~0. Every loss-weight conclusion before that date was re-run
  (`rerun_fixed.py`, `lossfit.py`).
* The pipeline moved to `fit_abstraction_alternative`, which surfaced three measurement
  bugs: unseeded restarts, `info_eff` not charging for discarded nodes, and `is_dag`
  fabricating cycles from 1e-16 dust. → [FITTER_SWITCH.md](FITTER_SWITCH.md).

## 5. Partition baselines: three learners, not one  (09-21 → 09-22)

All-In is a structural-equivalence condition. **Community detection optimises the wrong
sign**, since it maximises intra-cluster mass. Out of the baselines came the three learners
we kept:

* **OC Closure**: selects on minimum information loss rather than the widest draw.
* **AOC Closure**: one-child closure plus All-In repair, so it is exact by construction.
* **Greedy All-In**: lexicographic (violations, Δintra) merges to a budget k.

The key finding is that **the conditions are a cliff at α = 0 and a dial above it**. At
α = 1e-6, satisfying both conditions gives 0.87 acceptance, but All-In alone gives 0.04.
At α = 0.05, the All-In-only partitions lose far less information (0.19 vs 0.59). Every
method must be held to the same k budget, otherwise singletons win trivially.
→ [PARTITION_BASELINES.md](PARTITION_BASELINES.md).

## 6. The T-fitting reckoning  (09-23 → 09-24)

As the partitions improved, the fitter turned out to be the weak link:

1. **`get_S` was wrong on cyclic quotients.** It used E_low's block instead of the local
   inverse. Fixed; the oracle now scores ~1e-30 against itself.
2. **The P(valid | cyclic) = 0 result was a generator artefact.** Adding back-edges to a
   finished A_low leaves no abstraction to find. With a proper cyclic ground truth
   (`cyclic_gen.py`), exact abstractions exist on fully cyclic quotients.
3. **Random partitions also reached RAE = 0**, because the fitter marginalised exactly the
   All-In violators. So RAE alone cannot rank partitions, and **ReconErr became the second
   metric** in the acceptance gate.
4. **The fitters were T-unfaithful.** They reached RAE = 0 while dropping 62–89% of the
   true cross-cluster edges. Swapping objective and constraint fixed this: *maximise*
   ‖A_high‖² subject to residual ≤ tol. That improved reconstruction by 8–11 orders. A
   soft norm penalty was tested and rejected (it destroys exactness).
5. **Blockiness 0.5 cannot discriminate fitters**, because even the oracle reconstructs at
   ~0.9. Fitters are compared at blockiness 1, where the oracle reaches 1e-16.
6. No single fitter won everywhere. Propagation was best on acyclic quotients and `_K` on
   cyclic ones, so `fit_abstraction_auto` routed between them on quotient acyclicity.
   → [FITTER_SWITCH.md](FITTER_SWITCH.md), [ALPHA_RECON_K.md](ALPHA_RECON_K.md).

## 7. The final pieces  (09-25 → 09-26)

**GradCAP was dropped.** It was 63% of the runtime, it never expanded coverage over the
closures, and it is excluded from the paper's figures. What remains is three sub-second
combinatorial learners.

**The eigen fitter replaced the router.** Composing the collinearity constraint
M_ij t_j = m_ij t_i around a cycle gives Π t = (∏m) t. So a cluster on a cycle is an
eigenvector of the cycle product and has *no* free parameters. On acyclic quotients the
fitter reduces to propagation with optimised sinks. Results (`eigen_bench.py`,
`tk_sweep.py`, medians):

| | eigen | `_K` | `auto` |
|---|---|---|---|
| cyclic GT: RAE / ReconErr | **5e-16 / 4e-16** | 8e-7 / 3e-6 | 6e-10 / 1e-9 |
| cyclic GT: seconds | **0.008** | 0.93 | 3.5 |
| acyclic GT: RAE / ReconErr | **3e-16 / 0.38** | 4e-7 / 0.71 | 6e-16 / 0.42 |
| random partition: RAE | **0.35** | 1e-5 | 2e-8 |

The last row is a feature. Eigen is the first fitter that does **not** certify random
partitions as exact, so RAE is meaningful again. `_K` annihilated outright on cyclic
quotients at small k (edge recall 0); eigen is exact there and 40–140× faster.

## Open issues

* **The largest-|λ| rule is measured, not proved, below blockiness 1.** It was right in
  every instance tested (blockiness 0.5–0.9), but n is small. Overdetermined clusters
  (on several cycles) use one cycle each, and any disagreement is left as residual.
* **Acyclic ReconErr is ~0.3 against an oracle of ~1e-16.** The sinks are not pinned by
  the abstraction condition, so this gap is sink freedom, not fitter error.
* **Not every paper output uses eigen yet.** `supplementary/figures/alpha_b08_k10*.pdf`
  are the `auto` versions (the `_eigen` re-runs exist in `figures/`), and
  `fitter_gt_random.csv` was produced with `auto`.
* **RAE acceptance depends on the noise weighting.** It varies up to 30× on inexact
  instances, so acceptance rates are optimistic.
* The synthetic generator never produces non-negative cyclic models. Only the Leontief data
  (EXIOBASE3, GB) tests that class.
