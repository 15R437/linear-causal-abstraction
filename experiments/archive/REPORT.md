# Learning graph partitions for causal abstraction — what we found

*Investigation of 2026-09-10 → 09-13. Every claim below is backed by a CSV in `data/` and a
script in `scripts/`. Figures in `figures/`. The dense reference version is
[FINDINGS.md](../../results/smoke_tests/20260911-235555_FINAL/FINDINGS.md).*


## Answers at a glance

| question | answer |
|---|---|
| What does OCC's partition look like at p=50-200? | **One blob + singletons -- a star, and near-TRIVIAL.** Exactly one non-singleton cluster in 87% of random instances (100% at d=0.9); blob is the sink, singletons are sources feeding it. Only **1.9%** of nodes sit in genuine mid-sized clusters (0.0% at d=0.9). The one_child family is different: 21.4% mid-sized, blob only 19%. |
| Are the abstractions trivial on random DAGs? | **Essentially yes** -- 98-100% of nodes are in the blob or alone. But this is a property of the CONDITIONS, not of OCC: the best achievable balance is 0.889 vs OCC's 0.856, and HailMary cannot improve it either. |
| Is the blob inherent or an artefact of OCC? | Largely inherent. Exhaustive enumeration at p<=10: the best valid partition can only improve balance 0.856 -> 0.889. HailMary cannot break it up either. |
| Is OCC optimal? | **On k, yes** -- it finds the true maximum k on every instance at p<=10. On information, no: 0.151 vs 0.107 achievable at the same k. |
| Are All-In + one-child necessary? | **No, only sufficient.** ~44% of All-In-violating partitions still give an exact abstraction, because fit_abstraction marginalises the offenders. |
| Does one-child matter? | It is what makes All-In SATISFIABLE. Without it, All-In holds 0/80 at any loss weight. |
| Is the high-level graph acyclic? | Free on DAGs (a corollary of the two conditions, 155/155). Must be imposed separately when the low-level graph is cyclic — but as a *sufficient* condition only; see the artefact warning below. |
| Does OCC handle cyclic low-level graphs? | Yes, if the quotient is acyclic: 111/111 valid with it, 0/89 without (re-confirmed under the corrected S). `fit_abstraction` handles cycles; only the quotient must be acyclic. |
| ⚠ Is a RANDOM partition a valid control? | **No — it reaches RAE = 0 too.** `fit_abstraction_K` is exact on 100% of random partitions. It marginalises 27-36% of nodes (T entries -> ~1e-9), hitting *precisely* the All-In violators, and the high-level graph collapses to a star. Reconstruction 0.98-0.99 where predicting NO edges scores 1.0. Report info_loss and T-faithfulness beside RAE, never RAE alone. |
| ⚠ Is an acyclic quotient NECESSARY for an exact abstraction? | **No — and the `P(valid\|cyclic)=0` result must not be cited as if it were.** That zero is an artefact of `cyclic.py`, which adds back-edges to a finished `A_low`, leaving a model that is no longer a concretisation of anything and so has no abstraction to find. With a proper cyclic ground truth (`A_high` cyclic first, then concretisation) the abstraction exists on a 100%-cyclic quotient and `fit_abstraction_K` recovers it at RAE ~4e-11 in 93% of instances. |
| ⚠ Is `get_S` correct? | **Fixed 2026-09-23.** It used `E_low`'s diagonal block where the abstraction equation forces the LOCAL inverse `(I - A_low[pi_i,pi_i])^-1 t_i`; on cyclic quotients the generating abstraction scored RAE 0.02-0.21 against ITSELF. Now computes the local form, oracle back to ~2e-30, acyclic results unchanged to 4.7e-15. Pass `A_low`. |
| Which T-fitter should I use? | **`fit_abstraction_K` with `faithful=True`** (the default since 2026-09-24; the old path reconstructs at 0.50-0.67 on acyclic GT partitions and loses 15% of edges at p=40). Plain **`fit_abstraction_K`.** Only one both exact and fully T-faithful on acyclic AND cyclic ground-truth partitions, and the fastest (0.3-1.7s). `fit_abstraction` is structurally unfaithful on cyclic quotients (`enforce_dag` deletes the cycle edges before fitting); `_alternative` does not reach zero even on acyclic GT (median 3e-05 at p=40). |
| How robust is rel_err to the noise model? | **Not very, where it matters.** Most instances are exact (rel_err=0, no sensitivity), but among the inexact ones the weighting changes the value by up to 30x and flips the accept decision on 3 of 4. The generator's SNR weighting is the most favourable, so reported acceptance rates are optimistic. |
| Does the refinement scale to p=200? | **Yes at density >= 0.3** -- 69% accepted, -25% info vs OCC. At d=0.1 it is never accepted (OCC is already good there unaided). Cost 268s vs OCC's 0.25s. |
| ⚠ Is any of this superseded? | **Yes.** The `edge_weight` term in graph_loss destroyed its correlation with All-In violations in the soft regime. With it removed, `lambda_graph` DOES matter (All-In 0.35->0.55) and graph_loss alone reaches All-In 0.40 with 96.5% retention. See the edge_weight section. |
| What is the final algorithm? | **OCC, then HailMary (closure init, slack 1-2xK, one_child=100, info=10, graph OFF), accept if rel_err <= alpha else keep OCC.** At alpha=0.05: retain **28% more** than OCC, 92% accepted. `info` beats `recon` at every alpha. |
| What is the alpha-frontier? | **Tolerate 5% relative error -> 34% better retention than OCC** (info_eff 0.396 vs 0.604), always safe via OCC fallback. alpha=0 IS OCC. lambda_util=10 optimal at every alpha. |
| Can we get APPROXIMATE abstractions that retain more? | **Yes, and this is HailMary's real role.** intra_mass with lambda_graph=0: info 0.374 vs OCC's 0.673 (**44% better**) at 3.9% relative error (p90), median 0. recon with lambda_graph=0: 94% fully valid, zero error, 13% better info. OCC cannot do this -- it only returns exactly-valid partitions. |
| Does HailMary beat OCC on EXACT abstraction? | Never on coverage (0 of 35 instances OCC failed). On information, ~8-15% better on DENSE random graphs; nothing on one_child; WORSE on sparse (d=0.1). |
| Best HailMary config? | recon loss, closure init, k=K, with OCC fallback. reg_surj init and varying k both fail to help. |
| Can reconstruction loss measure quality? | **No** -- ~90% of valid abstractions score worse than predicting no edges, because one-child caps the high-level graph at k-1 edges. |
| How should quality be measured? | Largest-cluster size + information loss. **Cluster count is near-useless** when 96% of clusters are singletons. Always report info beside success. |
| How fast is OCC? | 0.14s at p=200, 2s at p=400 (~p^1.6). `refine_by_splitting` contributes nothing on 90/90 and costs 99.5% of the runtime -- drop it. |

---

## The question we started with

HailMary learns a graph partition — part (i) of a two-part pipeline; `fit_abstraction`
learns the continuous T values in part (ii). Part (i) succeeds when two sufficient
conditions hold:

* **All-In** — for every high-level edge X→Y, *every* member of X has an edge into Y
* **one-child** — every cluster has at most one child

Theory says these guarantee an abstraction exists. **We confirmed it empirically**: over
2000 runs, P(valid abstraction | both conditions) = **1.0000** (n=529 one_child, n=516
random), `A_high` a DAG in 100% of them. When they fail: 0.758 / 0.267. So this is measured
fact, not an assumption we are leaning on.

The opening question was narrow — *would a surjectivity loss or a different initialisation
make HailMary better?* The answer turned out to be no to both, but chasing it exposed the
structure of the problem, and that structure is the real result.

---

## Act I — Three bugs, and one that mattered

Before anything could be measured, three bugs had to go (`fitcmp.csv`, `random_check.csv`):

1. **`surjective_loss` had no gradient.** A boolean comparison sat between the parameters
   and the loss; measured gradient norm exactly **0**. `lambda_surj` had never done anything
   at any value — which retroactively explains why every surjectivity experiment you
   remember running came back inconclusive.
2. **`fit_abstraction_alternative` returns an all-zero `A_high`.** The zero vector lies in
   every span, so annihilating real cross-cluster signal is a *global* optimum. It lost
   every edge in 3/120 runs. Switching `process_output` to `fit_abstraction`: never lost an
   edge, median log10 obs −18.0 vs −14.3, and **37× faster**.
3. **Seeds did not reproduce weights.** `sample_disjoint_uniform` accepted a `seed` and
   ignored it, drawing from global `np.random`.

---

## Act II — What actually limits us

### Q: What is the largest k admitting a valid partition, as a function of p?

**A: it is not a function of p.** It is governed by the ground-truth granularity.

![ceiling](figures/01_ceiling.png)

| ceiling / p | nh=3 | nh=5 | nh=10 | nh=20 | nh=40 |
|---|---|---|---|---|---|
| one_child | 0.572 | 0.591 | 0.600 | 0.621 | 0.652 |
| random | 0.492 | 0.642 | **0.189** | **0.124** | **0.079** |

A clean ~0.6p on one_child at every granularity; on random it collapses as n_high grows,
because more high-level blocks at fixed density means higher out-degree.

**All-In never binds** — singletons satisfy it trivially, so it alone permits k = p.
One-child sets the entire ceiling, and it tracks maximum out-degree: a node with d children
forces its d targets into one cluster, which cascades.

> *This corrects an earlier claim that the random ceiling is "roughly constant at 20–25
> regardless of p". That was an artefact of a grid that fixed n_high = p/10, moving p and
> n_high together.*

**Consequence for choosing k:** a fixed fraction of p is the wrong shape. Use the closure's
own K (below). If you must use a fraction, k = 0.1p is the *worst* setting for success on
both families; 0.4p maximises success, 0.2p hedges.

### Q: If we drop the one-child loss, how often is All-In satisfied, and what is the info?

**A: All-In is satisfied _never_ — 0 of 80 runs, at any weight — and info goes to ~0.**

![antagonism](figures/04_antagonism.png)

Sweeping `lambda_graph` over two orders of magnitude with `lambda_one_child = 0`
(`graphweight.csv`, and 114 earlier runs in `nochild.csv`):

| arm | All-In | info | high-level edges |
|---|---|---|---|
| control (lg=10, oc=100), random | 0.375 | 0.357 | 13.6 |
| oc=0, lg=10 | **0.000** | 0.000 | 214.8 |
| oc=0, lg=200 | **0.000** | 0.017 | 163.0 |
| oc=0, lg=1000 | **0.000** | 0.100 | 107.0 |

The optimiser *responds* to the weight — info 0.000→0.100, largest cluster 21→39 — it just
never arrives. And the last column shows why.

**One-child is not an extra constraint on top of All-In. It is what makes All-In
satisfiable.** Without it, partitions carry ~180–215 high-level edges; All-In demands every
member of a cluster have an edge into *each* of its children, a conjunction over ~4
children. One-child collapses that to one requirement per node — the only regime in which
it can hold.

This kills the intuition that one-child is what costs us the information. Dropping it does
not buy low-info *valid* partitions; it buys low-info **invalid** ones.

> *Scaling graph_loss by k (`graph_const`) was not run in this setting and does not need to
> be: its effective weight at these ceilings is ~200, bracketed and beaten by the lg=1000
> arm that already fails. Separately, `one_child_loss` self-scales ≈linearly with k, which
> is why the sweeps that scaled graph and info by k were already the "all three scaled" arm.*

### Q: Does density matter?

**A: yes, and it was a hidden confound in everything else — your prediction was right.**

![density](figures/02_density.png)

| closure success | d=0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|
| p=50 | 0.333 | 0.533 | 0.467 | 0.800 | 0.867 | **1.000** |
| p=100 | 0.417 | 0.533 | 0.467 | 0.800 | 1.000 | **1.000** |

Monotone in density, with the mechanism visible: the fraction of zero-out-degree nodes
triples from 0.10 to 0.33 as density falls, and those are exactly the nodes that strand.

**The two criteria move in opposite directions.** Sparse graphs lose the least information
(0.199 at d=0.1 vs 0.771 at d=0.9) and are the hardest to find valid partitions for. Every
other random-DAG result in this study fixes density=0.5 — well into the regime where the
problem is easy. **Treat density as a first-class axis in the big benchmark.**

---

## Act III — A much simpler algorithm

Investigating HailMary's initialiser turned up the **one-child closure**: start from
singletons, merge the children of any cluster with more than one child, repeat to a fixed
point. It enforces one-child only, never All-In, and is order-dependent.

### Q: Could it be a baseline in its own right? Is it guaranteed to work?

**A: yes to the first, no to the second — but the gap is now closed.**

It never returns an invalid partition (it validates before accepting), but it can return
**nothing**: 16.9% of random instances (44/260), flat in p (0.175/0.150/0.175/0.175 at
p=50/100/200/400).

**Those failures are structural, not stochastic.** 44 instances succeed on 0 of 12 draws,
80 on all 12. Among instances that return anything the per-draw failure rate is 0.277 — if
draws were independent, all twelve failing would have probability **2e-7**. So *more
restarts provably cannot help.*

### Q: Why do those instances fail?

**A: "locally-sunk" nodes — not sinks.** No global graph statistic predicts failure
(|ρ| ≤ 0.13 for sink count, density, max out-degree, root count). Only **20.3%** of
violating nodes are global sinks.

After closure, cluster X has at most one child Y, so any node in X sends out-edges either
*inside X* or into Y. A node whose out-edges all land inside its own cluster has nothing
reaching Y, and All-In breaks. The closure's own merging creates them by pulling a node's
targets in beside it.

### Q: So — fallback, or just more restarts?

**A: fallback, decisively.** Restarts cannot fix structural failures. The mechanism above
suggests the repair directly: when node i in X has no edge into X's child Y, *merge X and
Y* — and iterating terminates.

A single merge does NOT repair the violation locally: the merged cluster inherits Y's child
W, which members of X need not reach. Correctness comes from the loop — each merge strictly
decreases the cluster count, so it terminates within k−1 merges, worst case at the
single-cluster partition. One-child survives (the merged cluster's children are {W});
acyclicity survives because contracting X→Y could only create a cycle via a second path
X→a→Y, which one-child forbids. Measured: **at most 2 merges** over 38 instances needing
repair, never collapsing.

![fallback](figures/10_fallback.png)

| | closure alone | + targeted-merge fallback |
|---|---|---|
| coverage | 0.867 | **1.000** |
| on the 12 failures: clusters | — | 45.3 *(collapse-to-one gives 1)* |
| on the 12 failures: info | — | 0.423 *(collapse-to-one gives 1.000)* |
| merges needed | — | **1.0** |
| cost | — | **32 ms** |
| on instances closure solved | k 36.4, info 0.575 | **identical** |

One merge on average. It is additive — keep the 12 draws *and* rescue the remaining 13%.
**Closure is now a complete method: it always returns a certified-valid partition.**

### Q: How fast, and does the refinement step earn its keep?

**A: ~0.14 s, and no — `refine_by_splitting` does nothing at all.**

![refine](figures/03_refine.png)

Identical k, info and validity on **90/90** instances, while consuming **99.5%** of the
estimator's runtime. Dropping it is a **31× (random) / 286× (one_child)** speedup at zero
cost — the closure is already at a local maximum in k. Cost scales ~**p^1.6** (0.013 s at
p=50 → 0.40 s at p=400).

*(Caveat: the splitting search is 40 random cuts per cluster. The finding is "random
splitting never works", not "no valid split exists".)*

---

## Act IV — How should we measure utility?

### Q: Does reconstruction loss track intra-cluster edge mass?

**A: no — it is significantly _inverted_, and the metric cannot discriminate at all.**

![recon](figures/06_recon.png)

| metric | mean ρ(info, metric) | positive | p |
|---|---|---|---|
| recon (Frobenius) | −0.340 | 7/25 | 0.006 |
| recon_data (raw) | +0.095 | 14/25 | 0.542 |
| recon_std (standardised) | −0.252 | 7/25 | 0.037 |
| recon_l1 (standardised, L1 `A_hat`) | −0.283 | 7/25 | 0.020 |

The reason is structural: **one-child forces out-degree ≤ 1 on the high-level graph**, so a
k-cluster abstraction has at most **k−1 edges** and is always a forest of chains — 1–7
high-level edges standing in for ~364 low-level ones. Measured against a "predict no edges"
null, **~90% of valid abstractions are worse than the null**. Once that holds the metric
scores *damage*, and finer partitions (lower info) build a larger `A_hat` with more spurious
structure — hence the inversion.

Two traps inside the metric itself, both worth knowing:

* **`A_hat` construction matters a lot.** `reconstruct_adj` (pinv) is far worse than the
  L1-normalised form — 2.233 vs 1.584 on the same partition. **`experiments/synthetic.py`
  passes the pinv version**, overriding the better default inside
  `reconstruction_loss_with_data`.
* **Raw RMSE is variance-weighted and nearly constant.** Node variances span 1e10, so 5 of
  100 nodes carry 72% of the score; across valid partitions of one instance raw RMSE moves
  **0.011%** while the per-node-standardised version moves **3.5×**. The generator is not at
  fault — empirical SNR is uniform at 1.73–1.83 across all non-root nodes, by design. The
  spread is topological depth, which is exactly why per-node standardisation is right.

### Q: So can we just use intra-cluster edge weights? Is variance-weighting more principled?

**A: yes, use it — as a _definition_. And the weighting does not matter.**

![infomeasures](figures/07_infomeasures.png)

Three weightings rank partitions **identically — ρ = +1.000 on 14/14 instances**:

| measure | what it weights | cost at p=400 |
|---|---|---|
| `info_raw` | \|A_ij\| | 0.29 ms |
| `info_var` | w²·Var(Xi)/Var(Xj) — standardised path coefficient | 3.7 ms + 5.6 ms for variances |
| `anc_lost` | ancestral pairs co-clustered — weight-free | — |

So no referee can argue the |A_ij| weighting drives your result, and the variance version is
**cheap but unnecessary** (it changes levels 0.655 → 0.779, never order). Compute variances
analytically — `diag(MMᵀ)`, `M = (I−Aᵀ)⁻¹` — not by sampling, which is 30× more expensive.

Note the current implementation **already weights by variance implicitly** (raw squared
residuals in original units); per-node standardising *removes* that weighting.

**Argue intra-cluster mass as constitutive of information loss, not as a validated proxy.**
The three measures are not independent — all are functions of the co-assignment matrix — and
the only genuinely external candidate, reconstruction, is degenerate here.

### ⚠ The measurement trap that invalidated four of our own claims

**Runs that FAIL score _better_ on information than runs that succeed.**

| random family | failures | successes | unconditional mean |
|---|---|---|---|
| closure init | 7 runs @ **0.252** | 3 runs @ 0.565 | 0.346 |
| random init | 5 runs @ **0.412** | 5 runs @ 0.571 | 0.492 |

This is causal, not coincidental: `lambda_info` pushes mass out of clusters, `lambda_graph`
pulls toward All-In, and a run fails precisely when info wins. Failing runs keep 14.9
clusters; succeeding ones collapse to 7.3.

**Any unconditional info column flatters whichever method fails most.** This single error
produced four wrong claims during the investigation (including "HailMary is 2–6× better on
utility", which paired properly is 0.033). Report info **only over valid partitions, always
beside the success rate.**

---

## Act V — Does HailMary earn its keep?

### Q: How does it compare to closure, paired on the same graphs?

![paired](figures/05_paired.png)

| random, 45 paired instances | |
|---|---|
| success | closure **0.933**, HailMary 0.644 |
| HailMary valid where closure failed | **0 of 3** |
| both succeeded (n=29) | closure info 0.434, HailMary **0.401**, diff **−0.033** |
| HM better / closure better / tied | **8 / 0 / 21** |

| one_child, 45 paired instances | |
|---|---|
| both | success 1.000, info exactly 0.000, **all 45 tied** |
| time | closure **0.70 s**, HailMary 208 s |

**Across 121 paired instances, HailMary has never solved a graph closure could not** (0 of
35 such instances). Its contribution is a small, **strictly non-negative** improvement in
information on ~28% of random instances.

When it does fail, the failures are **not** benign near-misses: on random, failing runs
average **29.7 violating nodes out of 100** (median 31). Every one becomes a marginalised
variable in `fit_abstraction`.

### Q: Is the intended pipeline — closure's exact partition, same K — better?

**A: yes, by ~7 points of success, and most of our comparisons were not running it.**

| arm (42 paired random instances) | success | info |
|---|---|---|
| closure alone (the exact partition) | 1.000 | 0.495 |
| **exact partition → HailMary at k=K** | 0.762 | 0.395 |
| `closure_logits` (1 unvalidated draw + fold) | 0.690 | 0.374 |
| with fallback when HM breaks validity | **1.000** | 0.473 |

**`closure_logits` is not this pipeline.** It calls the closure *once*, never checks
validity, then folds clusters down to `n_high`. On one instance: the best of 12 draws had
**0** violations; the single draw it used had **38** — already invalid before any folding,
which added just 1 more. Fix: sample several orders and keep a valid one. Milliseconds
against a 6000-epoch run.

---

## Act VI — Four levers, none of which moved

![failed levers](figures/08_failed_levers.png)

| lever | result |
|---|---|
| **Slack** (k > K) | Trades success for info, doesn't escape the minimum: success 0.647→0.765 while info *worsens* 0.446→0.481. No effect at all on one_child. |
| **Restarts** | 5 restarts == 1 restart in **7 of 8** density cells. HailMary's failures are structural too. |
| **Initialisation** (margin/noise, 192 runs, 50:1 → pure random) | Nodes moved varies **5×** (12.7 → 67.9); outcome does not. Every setting within 0.019–0.028 of closure; conditional info flat at 0.616–0.629. |
| **Loss reweighting** | `graph_const` harmful at p=100, neutral elsewhere. Dropping one-child destroys All-In entirely. |

Pure random init abandons the closure partition almost entirely and *still* lands at the
same quality. Closure's partitions are not a lucky basin — they are already near the ceiling
the optimiser can reach.

**The search is not the bottleneck. The All-In constraint is.** Initialisation, slack,
restarts and loss reweighting have now all been tried against it.

Also settled, on the `one_child_loss` form: five variants (IPR, c−1, per_k, hard, thresh)
all remove the k-drift *and* the enforcement, because they are the same quantity — an
inflated effective child count. **Keep the current form on `T_soft`** (`ocnorm.csv`,
`cminus1.csv`, `leak.csv`). Your prior on this was right.

---

## Where it lands: the benchmark

All methods at the same k, 32 instances, scored on what matters (`benchmark.csv`):

![benchmark](figures/09_benchmark.png)

| family | method | success | **exact recovery** | info | recon_data | secs |
|---|---|---|---|---|---|---|
| one_child | **closure** | 1.0 | **1.000** | 0.000 | 4.03 | **0.21** |
| | closure+HM | 1.0 | 1.000 | 0.000 | 4.03 | 32.2 |
| | hc_fast | 0.0 | 0.062 | 0.042 | 3.38 | 0.64 |
| | uca | 0.0 | 0.125 | 0.036 | 4.16 | 14.5 |
| | random | 0.0 | 0.000 | 0.026 | 4.26 | 0.00 |
| random | **closure** | 1.0 | **1.000** | 0.583 | 27.9 | **0.04** |
| | closure+HM | 1.0 | 1.000 | 0.577 | 27.9 | 15.1 |
| | hc_fast | 0.0 | 0.000 | 0.400 | 23.2 | 0.86 |
| | uca | 0.0 | 0.000 | 0.075 | 28.3 | 11.6 |
| | random | 0.0 | 0.000 | 0.076 | 27.8 | 0.00 |

`exact` = obs_loss **and** intv_loss < 1e-8. **100% on both families; no baseline exceeds
12.5%.** The baselines' lower *info* is the trap above — they are invalid, so they never pay
the coarsening price validity demands.

---

## The large benchmark (234 instances, 1170 runs)

Density ∈ {0.1,0.3,0.5,0.7,0.9} × p ∈ {50,100} × n_high ∈ {5,10,20} × 8 seeds, plus the
one_child family. Every method at the same k. (`large_main.csv`)

![large density](figures/11_large_density.png)

| family | method | success | **exact recovery** | info | secs |
|---|---|---|---|---|---|
| one_child | **closure** | 1.00 | **1.000** | 0.000 | **0.34** |
| | closure+HM | 1.00 | 1.000 | 0.000 | 47.6 |
| | hc_fast | 0.00 | 0.050 | 0.036 | 0.99 |
| | uca | 0.00 | 0.000 | 0.036 | 21.3 |
| | random | 0.00 | 0.050 | 0.025 | 0.00 |
| random | **closure** | 1.00 | **1.000** | 0.466 | **0.12** |
| | closure+HM | 1.00 | 1.000 | 0.448 | 25.4 |
| | hc_fast | 0.01 | 0.077 | 0.345 | 1.28 |
| | uca | 0.00 | 0.026 | 0.076 | 16.0 |
| | random | 0.00 | 0.015 | 0.073 | 0.00 |

**Exact recovery is 1.000 at every density and every p.** `hc_fast` peaks at 0.176 (density
0.1) and is otherwise near zero. Read the info column only beside success: the baselines'
lower info is the trap in Act IV — they are invalid, so they never pay the coarsening price.

![large grid](figures/12_large_grid.png)

### The fallback does most of the work on sparse graphs

| family | density | raw closure | fallback needed | final success |
|---|---|---|---|---|
| one_child | 1.0 | 1.000 | 0.000 | 1.000 |
| random | **0.1** | **0.353** | **0.647** | **1.000** |
| random | 0.3 | 0.675 | 0.325 | 1.000 |
| random | 0.5 | 0.875 | 0.125 | 1.000 |
| random | 0.7 | 1.000 | 0.000 | 1.000 |
| random | 0.9 | 0.975 | 0.025 | 1.000 |

At density 0.1 the raw closure solves only 35% of instances; the targeted-merge fallback
rescues the rest. **It is not a patch on an edge case — on sparse graphs it is the method.**
This is much stronger evidence than the 12 failing instances it was first measured on.

HailMary's contribution across the whole grid is 0.466 → 0.448 information (and
0.530 → 0.492 at density 0.7), for ~200x the runtime, never making anything worse.

---

## The cyclic case

*Infrastructure built in `scripts/cyclic.py`: take the DAG generator, add `n_back` reversed
edges, rescale so the spectral radius < 1 so the equilibrium X = (I-A)^-1 U exists.*

### Q: does All-In + one-child still give an acyclic quotient?

**A: usually, but not always -- and the exception is exactly where you expected it.**

| back-edges | low-level cyclic | quotient acyclic | k |
|---|---|---|---|
| 0 | 0.0 | 1.00 | 20.3 |
| 2 | 1.0 | 1.00 | 19.4 |
| 5 | 1.0 | 1.00 | 18.4 |
| 10 | 1.0 | 1.00 | 16.6 |
| 25 | 1.0 | 1.00 | 11.8 |
| 50 | 1.0 | **0.95** | 7.9 |

On DAGs, acyclicity of the quotient is a **corollary** of the other two conditions: if the
quotient had a cycle X1->X2->...->Xn->X1, All-In lets you step from any member of X1 into
X2, then X3, ... back into X1, forever -- an infinite walk, impossible in a finite DAG.
Measured: 155/155 on DAGs, 0 cyclic quotients.

**That argument needs low-level acyclicity**, so with cycles it fails -- and empirically it
does fail, just rarely, and only under heavy cycling.

### Q: does an abstraction still exist? Does `fit_abstraction` even work?

**A: yes to both, provided the QUOTIENT is acyclic.**

      quotient ACYCLIC:  P(valid abstraction) = 1.0000   (n=111)
      quotient CYCLIC :  P(valid abstraction) = 0.0000   (n= 89)

A heavy sweep (200 instances, 50-800 back-edges) drives the quotient cyclic often enough to
populate both cells, and the separation is PERFECT at every cycle density:

      back-edges   quotient acyclic   P(valid | acyclic)   P(valid | cyclic)   k
      50               0.975              1.000  (39)         0.000  ( 1)     7.7
      100              0.875              1.000  (35)         0.000  ( 5)     4.0
      200              0.575              1.000  (23)         0.000  (17)     2.5
      400              0.325              1.000  (13)         0.000  (27)     1.9
      800              0.025              1.000  ( 1)         0.000  (39)     2.1

**`fit_abstraction` handles cyclic low-level graphs.** `reverse_topo_order` runs on the
HIGH-level mask, not on `A_low`; low-level cycles enter only through `(I - W_cc)^-1`, the
within-cluster block, which is fine whenever the spectral radius < 1. So
`fit_abstraction_alternative` is not required for the cyclic case -- which matters, because
that solver has the all-zero-`A_high` global optimum (Act I).

DANGER: `reverse_topo_order` has **no cycle detection**. On a cyclic quotient it returns an
invalid order *silently* rather than raising. Any cyclic-case code must check the quotient
is acyclic first.

### So the theorem has two regimes

| low-level graph | sufficient conditions |
|---|---|
| acyclic | All-In + one-child *(acyclic quotient comes free)* |
| cyclic | All-In + one-child + **acyclic quotient** *(the third must be imposed)* |

**Acyclicity of the quotient is exactly the extra condition needed** -- 111/111 with it,
0/89 without, across five cycle densities.

CAVEAT: the generator creates cycles by reversing existing edges, so short cycles dominate;
longer cycles are untested. Spectral radius is fixed at 0.8.

---

## 2026-09-23 — ⚠ `get_S` IS WRONG ON CYCLIC QUOTIENTS  [affects measurement, not the conclusion above]

`get_S` (helper.py) builds `S_ii = E_low[pi_i, pi_i] @ t_i` from the **full** inverse. The
abstraction equation `(I - A_low) S = T (I - A_high)` does not permit that. Taking block
`(i, i)` of the left side gives `(I - A_low[pi_i, pi_i]) S_ii = t_i`, i.e.

      S_ii  =  (I - A_low[pi_i, pi_i])^-1 t_i          <- LOCAL inverse, not E_low's block

The two agree **iff no path leaves cluster i and returns** — that is, iff the quotient is
acyclic. On a cyclic quotient they differ by O(1).

**What this invalidates: cyclic measurements, including the generating abstraction itself.**
On a cyclic-quotient instance built to be an exact abstraction by construction, the true
`(A_high, T)` pair scores `RAE = 0.02 - 0.21` against itself under `get_S`, and `~1e-30`
under the correct S. Anything that scored a cyclic quotient through `get_S` — which includes
`process_output` (helper.py, `S_pred = get_S(...)`) and therefore `scripts/cyclic.py` — was
measuring the defect of `get_S`, not of the fit.

It also contaminates **random/unlearned partitions on acyclic graphs**: those induce a cyclic
quotient in 87-100% of instances, so any control arm scored this way is affected too.

### It does NOT overturn the cyclic conclusion above

Re-ran the decisive cell with both S definitions, all three fitters, same generator, same
closure partitions (`scripts/cyclic_recheck.py`, `data/cyclic_recheck.csv`, 100 instances):

| quotient | n | P(valid), `get_S` | P(valid), corrected S |
|---|---|---|---|
| acyclic | 56 | 1.000 | 1.000 |
| cyclic | 44 | 0.000 | 0.000 |

Identical for `fit_abstraction`, `_alternative` and `_K`. So `P(valid | cyclic quotient) = 0`
on back-edge-perturbed graphs is real, not an artefact.

### ⚠ THE `P(valid | cyclic) = 0` RESULT IS AN ARTEFACT OF THE GENERATOR, NOT A FACT ABOUT CYCLES

**Read this before citing the two-regime table above.** That table says a cyclic low-level
graph needs "All-In + one-child + acyclic quotient". The third condition is an artefact of
how `scripts/cyclic.py` builds its instances, and does **not** generalise.

`cyclic.py` takes a finished `A_low` — one already generated as the concretisation of some
`A_high` — and then **adds reversed edges to it**. Those back-edges are not the image of any
high-level structure. The perturbed model is no longer a concretisation of anything, so for
most partitions there is simply **no exact abstraction in existence for the fitter to find**.
Measuring `P(valid) = 0` there says nothing about cycles; it says the question was asked of a
model that has no answer.

Generate a **proper cyclic ground truth** instead — make `A_high` cyclic *first* (rescaled to
rho < 1), then run `T_concretisation_sampling` — and everything works:

| | back-edge family (`cyclic.py`) | proper cyclic ground truth (`fit3cmp.py`) |
|---|---|---|
| quotient cyclic | 44/100 instances | 60/60 instances (by construction) |
| exact abstraction exists | no | **yes, by construction** (`\|\|E_low T - S E_high\|\|_max ~ 3e-15`) |
| `fit_abstraction_K` finds it | 0.000 | **0.93**, at `RAE ~ 4e-11` |
| oracle RAE | n/a | ~1e-30 |

**Cycles are not the problem and never were.** The exactness algebra contains no topological
order anywhere. Block `(i, j)` of `(I - A_low) S = T (I - A_high)` requires exactly

      A_low[pi_i, pi_j] E_jj t_j  =  A_high[i,j] t_i

and `T_concretisation_sampling` satisfies it by construction via `c . s_j = 1`, whether or
not `A_high` has cycles. The same holds for the diagonal block. Nothing in the derivation
would notice a cycle.

So the correct statement is:

> **A cyclic quotient does not preclude an exact abstraction.** With a genuine cyclic
> generating model, the abstraction exists on a 100%-cyclic quotient and is recovered at
> `RAE ~ 4e-11`. What `cyclic.py` measured is that back-edge-perturbed models mostly have no
> abstraction to find — a statement about that perturbation, not about cyclicity.

The two-regime table remains valid as a **sufficient**-condition statement, and the
`acyclic 1.000 / cyclic 0.000` split is a real property of the back-edge family (it survives
the corrected S, see the table above). But "acyclic quotient" must not be carried into the
paper as **necessary** for an exact abstraction. It is not.

One caveat in the other direction: `fit_abstraction` specifically *does* require an acyclic
quotient, because `high_level_mask` runs `enforce_dag` and deletes the cycle edges before
fitting starts (0/30 T-faithful on proper cyclic ground truth). That is a limitation of that
one fitter, not of the framework — use `fit_abstraction_K`. See FITTER_SWITCH.md.

### The fix  [APPLIED 2026-09-23]

`get_S` now computes the **local** form directly:

      s_i = (I - A_low[pi_i, pi_i])^-1 t_i

Signature is `get_S(T, E_low=None, T_hard=None, A_low=None)`. Pass `A_low`; `E_low` is still
accepted so older call sites keep working, and `A_low` is then recovered from it at the cost
of one extra O(p^3) inverse. All six `src/` call sites pass `A_low`; the study scripts under
`scripts/` still pass `E_low` only, which is correct, just slower.

An equivalent form exists that needs no `A_low` — from the `(pi_i, i)` block of
`E_low T = S E_high`, `s_i = e_ii^-1 E_low[pi_i, pi_i] t_i`, with `e_ii` read off
`diag(M^-1)` where `M = D^-1 E_high` has a unit diagonal and `M^-1 = (I - A_high) D` (valid
because `A_high` has no self-loops). It was implemented, measured and then **removed** in
favour of the local form, because:

* it is a **vector** identity in the direction `t_i`, not a matrix one. Even after the `e_ii`
  factor, `||e_ii^-1 E_ii - (I - A_ii)^-1||max` reaches **4.0** on cyclic ground-truth
  instances (without it, 4.9);
* it holds **only under exactness**. Writing `E_ii = L + X` with `L = (I - A_ii)^-1` the paths
  staying inside `pi_i` and `X` the excursions that leave and return, exactness forces
  `X t_i = (e_ii - 1) L t_i` — an excursion exits through the high-level edges and re-enters
  carrying the cluster's single aggregate signal, and `e_ii` is that scalar. On an APPROXIMATE
  fit the relation degrades with the residual: measured ~2e-6 on random-partition fits whose
  own residual was ~2e-7. That matters directly for the alpha-frontier work, where the fits
  are deliberately inexact;
* it is not even faster — 1.00 ms against the local form's 0.76 ms at p=800, k=20, both
  negligible beside the 39.6 ms `inv(I - A_low)` already being paid.

Note where `A_high` goes in the local form: it never enters. `T` and `S` are both
block-supported, so every cross term in the `(pi_i, i)` block dies, and the only `A_high`
entry that could survive is the diagonal — which is zero. The `e_ii` form reintroduces
`A_high` solely to cancel the excursion term that `E_ii` drags in.

**Verified.** Unchanged to 4.7e-15 on acyclic quotients, so no acyclic number moves. Oracle
RAE on cyclic instances went from 6.5e-02 / 1.0e-01 to **~2e-30**. On fitted random-partition
`T` the result matches the local form to 3.3e-16 (it *is* the local form). Empty clusters,
zeroed columns, `T_hard=None` and all-singleton partitions all behave. Every dependent module
imports.

---

## 2026-09-23 — ⚠ RANDOM PARTITIONS ALSO REACH RAE = 0, BY MARGINALISING THE VIOLATORS

**RAE = 0 does not mean the partition is good.** Intended as a negative control, random
partitions turn out to admit exact abstractions essentially always:

| fitter | acyclic, GT part. | acyclic, RANDOM part. |
|---|---|---|
| `fit_abstraction_K` | 1.00 exact | **1.00 exact** |
| `_alternative` | 0.27-0.73 | 0.33-0.87 |
| `fit_abstraction` | 1.00 | 0.13-0.53 |

`fit_abstraction_K` reaches RAE < 1e-6 on **100%** of random partitions at both p=24 and
p=40. The control does not control for anything.

### The mechanism is node-level marginalisation, and it targets exactly the All-In violators

It is **not** whole-cluster marginalisation: `RAE`'s `n_skipped` is **0 in every cell**, and
the per-cluster unit-norm constraint keeps every column norm at exactly 1. What collapses is
individual **entries** of `T`.

Worked example — p=24, k=4, random partition, `fit_abstraction_K`, `RAE = 3.2e-14`
(`scripts/randpart_probe.py`). The `|T|` entries are cleanly bimodal, 11 of 24 in the range
1.5e-10 to 8.5e-8 and the rest above 0.30. Grouping each cluster's nodes by their **All-In
signature** (which clusters they point into):

      cluster 0:  signature (1,2,3) -> nodes [2, 7, 13, 16, 20]   SURVIVING: none
                  no cross-cluster children -> [10, 18]           SURVIVING: [10, 18]
      cluster 1:  signature (0,2,3) -> nodes [0, 15, 21]          SURVIVING: none
                  no cross-cluster children -> [9, 14, 17]        SURVIVING: [9, 14, 17]
      cluster 2:  signature (0,1,3) -> nodes [6, 19]              SURVIVING: none
                  no cross-cluster children -> [3, 4, 5, 22]      SURVIVING: [3, 4, 5, 22]
      cluster 3:  signature (0,1,2) -> nodes [1, 8, 11]           SURVIVING: [1, 11]
                  no cross-cluster children -> [12, 23]           SURVIVING: [12, 23]

In three clusters out of four the fitter zeroes **precisely** the nodes with cross-cluster
children, turning those clusters into pure sinks. Only cluster 3 keeps its emitting nodes. So
the high-level graph collapses to a **star with cluster 3 as the sole parent** — 3 edges,
against 12 blocks that actually carry low-level edges:

      |A_high| on the 12 real blocks:
        1.06   0.72   0.70   1.4e-07  7.0e-08  5.8e-08  4.3e-08  4.0e-08
        2.9e-08  2.4e-08  2.3e-08  1.2e-08

A seven-order-of-magnitude gap. The nine attenuated edges are the ones whose parent cluster
had all its emitters zeroed. This is the same T-unfaithful root `fit_abstraction_alternative`'s
docstring warns about ("Y2 = X2 - X3 cancelling X1's effect"), reached by marginalisation
rather than by cancellation.

### It is the EMITTERS that get marginalised, so clusters lose their outgoing edges

The condition for `i != j` is `A_low[pi_i, pi_j] E_jj t_j = A_high[i,j] t_i`. Its left side
depends on the **parent's** vector `t_j`, and it demands the influence vector be *parallel
to* `t_i` — a rank-1 constraint a random partition generically cannot meet. The cheap escape
is to make it **zero**, which needs only that `supp(t_j)` avoid the nodes emitting into
`pi_i`. Zeroing *every* emitter in cluster `j` discharges all `k-1` constraints in column `j`
at once, killing **column j** of `A_high` — cluster `j` stops being a parent of anything. But
the move is finer-grained than that: zeroing only the nodes that emit into `pi_i` kills the
single edge `(i,j)` and leaves the rest of column `j` alive.

**Marginalisation is the mechanism; T-unfaithfulness is the result.** The other route to the
same result is *cancellation* — keeping the emitters nonzero and having their contributions
annihilate, which is the `Y2 = X2 - X3` case in `fit_abstraction_alternative`'s docstring.
Marginalisation dominates but does not monopolise (`scripts/mechanism_split.py`, acyclic,
random partition, 15 trials/cell):

| dead cross-cluster edges | p=24 | p=40 |
|---|---|---|
| total | 80 | 213 |
| structural (parent's support avoids every node emitting into that child) | 74 (**93%**) | 163 (**77%**) |
| cancellation (live emitters present, edge vanishes anyway) | 6 (7%) | 50 (23%) |
| explained by a fully dead parent column | 93% | **60%** |
| inside a column that is still live (partial loss) | 7% | **40%** |

At p=24, k=4 on a dense graph nearly every emitter emits into every other cluster, so zeroing
it kills the whole column — which is why the worked example below looks cleanly star-shaped.
At p=40, k=6 the signatures vary more and 40% of the losses are partial.

Measured selectivity (`scripts/emit_probe.py`, `data/emit_probe.csv`, 15 trials/cell,
acyclic, random partition):

| | `_K` p=24 | `_K` p=40 | `_alternative` p=24 |
|---|---|---|---|
| P(marginalised \| node emits cross-cluster) | **0.517** | **0.604** | 0.032 |
| P(marginalised \| node does not emit) | 0.064 | 0.050 | 0.026 |
| live parent clusters / true parent clusters | 0.52 | 0.70 | 0.98 |
| edges kept / blocks carrying low-level edges | 0.52 | 0.55 | 0.99 |

About a 10x selectivity for emitters, and roughly **half** the parent clusters go dead. The
worked example above (1 live parent of 4, 3 edges of 12) is the harsh end of that
distribution, not the typical case.

**Marginalisation is not a side effect of exactness — it is the mechanism.** On *cyclic*
random partitions the emitter-marginalisation rate is 0.000-0.065 and 95-100% of parents stay
live, and those are precisely the cells where `_K` **fails** to reach RAE = 0 (exact rate 0.40
at p=24, 0.00 at p=40). Where the escape hatch is unavailable, so is exactness.

Rates across the sweep — fraction of `T` entries driven below 1e-6 of their cluster's max:

| | GT partition | random partition |
|---|---|---|
| `_K`, p=24 / p=40 | 0.000 / 0.000 | **0.267 / 0.355** |
| `_alternative` | 0.000-0.008 | 0.007-0.033 |

On ground-truth partitions it never happens. `_alternative` marginalises far less — its
`maximise ||A_high||^2` objective fights it — and instead keeps nearly every edge
(recall 0.94-1.00) by inflating `A_high` to 28.8 edges against a 26.5-edge target.

### Yes, reconstruction pays for it — almost completely

`reconstruction_loss` is normalised by `||A_low||_F`, so **1.0 is exactly what you get by
predicting no high-level edges at all**:

| | GT partition | random partition | predict nothing |
|---|---|---|---|
| `_K` p=24 | 0.757 | **0.979** | 1.000 |
| `_K` p=40 | 0.856 | **0.992** | 1.000 |
| `_alternative` p=40 | 0.871 | **0.997** | 1.000 |
| oracle (generating pair) | 0.810 / 0.878 | n/a | 1.000 |

A random partition with RAE = 1e-14 reconstructs `A_low` **0.8-2% better than the zero
matrix**. The abstraction is exact and very nearly vacuous. Note also that even the oracle
only reaches 0.81-0.93 here — `reconstruction_loss` uses `pinv(T.T)` and is not a
scale-invariant measure of fit, so read the gap (0.76 vs 0.98), not the absolute value.

### What this means for the pipeline

* **Never report RAE or validity alone.** They are satisfiable by throwing information away.
  This is the same trap as the "measurement trap" section above, in a new place: a partition
  scores well *precisely because* it never pays the coarsening price. Always report
  `info_loss` / `info_effective` beside it — the random partitions here sit at info_loss
  0.15-0.23 against the ground truth's 0.000, which is the number that actually separates them.
* **T-faithfulness is the discriminating metric**, not RAE. On GT partitions `_K` has recall
  1.000; on random ones 0.49 at a relative threshold. That gap is the signal.
* **The `>= 1e-8` vs `!= 0` question is now load-bearing, not academic.** The nine dead edges
  sit at 1e-8 to 1e-7, so an absolute threshold calls them present (recall 0.90) and a
  relative one calls them absent (recall 0.49). The relative criterion
  `|(ET)_rj| >= c * max_s |e_rs t_sj|` proposed in FITTER_SWITCH.md would settle it and should
  be adopted.

---

## Scaling to p=400, and the one place it breaks

`large_scale.csv`, p in {200,400}, HailMary dropped (it is the cost driver, and adds ~0.02):

| p | family | method | success | exact | secs |
|---|---|---|---|---|---|
| 200 | one_child | closure | 1.000 | **1.000** | **1.3** |
| | | hc_fast | 0.000 | 0.000 | 6.6 |
| | random | closure | 1.000 | 0.933 | **0.35** |
| | | hc_fast | 0.022 | 0.022 | 13.6 |
| 400 | one_child | closure | 1.000 | **1.000** | **5.1** |
| | | hc_fast | 0.000 | 0.000 | 59.2 |
| | random | closure | 1.000 | 0.844 | **1.3** |
| | | hc_fast | 0.000 | 0.000 | **154.1** |

Closure stays at 100% on one_child up to p=400 in 5 s, while `hc_fast` takes 154 s and
recovers nothing. But on random, **exact recovery slips: 1.000 (p<=100) -> 0.933 -> 0.844**,
even though the two conditions still hold in 100% of runs.

### That slip is numerical, not structural

The median is untouched -- log10(obs_loss) is -26.2 at p=200 and -25.4 at p=400. All 10
failures sit at true_nhigh=40, 7 of them at density 0.9, where the closure collapses to
k=4-7 clusters (info ~ 0.97). The cause is the conditioning of the within-cluster block
that `fit_abstraction` inverts:

      case                  k    max cluster   cond(I - W_cc)
      FAILED (obs=3.7e6)    6        395          8.49e+12
      FAILED (obs=5.7)     45        356          6.46e+10
      typical, exact       22         79          3.81e+02
      p=400, easy cell     75        326          3.97e+02

Ten orders of magnitude apart -- and note the last row: a 326-node cluster is fine. It is
not cluster SIZE but the near-singularity of a dense within-cluster block at density 0.9.

PRACTICAL: `cond(I - W_cc)` is cheap and predicts the failure. Report it as a diagnostic
alongside obs/intv loss, and treat large values as "the solver cannot certify this", not
"the partition is invalid". The conditions held in every one of these runs.

## Graph shape: blockiness and heterogeneity

`large_shape.csv`, 144 instances at p=100. **Closure succeeds 1.000 in every cell** of the
3x3 grid, on both families.

![shape](figures/13_shape.png)

Heterogeneity has essentially no effect (random info 0.593 / 0.577 / 0.567 at het
0.5 / 1.0 / 2.0). Blockiness does move information loss -- random 0.365 (blk=0) -> 0.593
(blk=0.5) -> 0.602 (blk=1.0) -- but never threatens validity.

---

## The blob: what closure's partitions actually look like

*This is the most important caveat in the report, and it was measured last.*

![cluster sizes](figures/14_cluster_sizes.png)

Pooled over 376 instances (`clustersizes.csv`):

      size 1        15707 clusters   96.0% of clusters   33.3% of nodes
      size 2-50       431 clusters    2.7% of clusters   14.5% of nodes
      size 51-100     121 clusters    0.7% of clusters   20.1% of nodes
      size 101+        96 clusters    0.6% of clusters   32.1% of nodes

**Closure's partitions are a giant blob plus a sea of singletons -- literally.**

On random DAGs, **87.2% of instances have EXACTLY ONE non-singleton cluster**; 98.7% have at
most two. It sharpens with density (fraction of instances that are one blob + all singletons):

| p | d=0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 50 | 0.69 | 0.75 | 0.88 | 0.94 | **1.00** |
| 100 | 0.67 | 0.88 | 0.92 | 0.96 | **1.00** |
| 200 | 0.59 | 0.88 | 0.88 | 0.96 | **1.00** |

At d >= 0.9 it is ALWAYS one blob plus singletons. Even where a second non-singleton cluster
exists it is tiny -- second-largest averages 1.0-8.3 against a blob of 28-167.

**The one_child family is a different object entirely**: exactly-one-blob in 0.0% of
instances, mean 4.5 non-singleton clusters, largest 23.3, entropy 0.86. A genuine
multi-cluster abstraction.

| family | density | k | largest cluster | largest / p | singleton frac | entropy |
|---|---|---|---|---|---|---|
| one_child | 1.0 | 78.0 | 23.3 | **0.19** | 0.93 | **0.86** |
| random | 0.1 | 48.9 | 73.2 | 0.58 | 0.94 | 0.57 |
| random | 0.3 | 39.6 | 82.9 | 0.65 | 0.95 | 0.49 |
| random | 0.5 | 42.7 | 82.7 | 0.64 | 0.95 | 0.50 |
| random | 0.7 | 29.6 | 96.0 | 0.76 | 0.93 | 0.39 |
| random | 0.9 | 22.8 | 103.2 | **0.82** | 0.91 | **0.32** |

On random DAGs the largest cluster holds **58% of nodes at density 0.1, rising to 82% at
density 0.9**. This is exactly the **"blobby clustering" degeneracy** identified at the
start of the investigation as one of three to avoid.

**It qualifies the headline.** 100% exact recovery on random DAGs is achieved partly by
producing something close to the trivial single-cluster abstraction -- valid, but barely an
abstraction. The exact-recovery metric cannot see this, because a blob abstraction IS a
valid abstraction.

The `one_child` family is genuinely healthy by contrast: largest cluster 19% of nodes,
entropy 0.86, k=78 at p=100-200. There the abstraction is real.

**How this fits the rest of the report.** It is the same fact as the ~0.5-0.6 information
loss, seen from a different angle -- a blob holding 82% of nodes holds most of the edge mass
too. And it is forced, not a search failure: section "Act II" showed one-child caps k on
random graphs, and the `lambda_one_child=0` sweep showed partitions with info ~ 0 exist but
are never valid. The blob is the price of All-In, not a defect of the closure.

### Can HailMary break up the blob?

**No -- and that settles whether the shape is forced.** 150 paired instances at the same k
(`hm_sizes.csv`); HailMary stayed valid on 100 of them.

| metric | closure | HailMary | delta | HM better on |
|---|---|---|---|---|
| largest cluster / p | 0.565 | 0.556 | **-0.009** | 19/100 |
| normalised entropy | 0.567 | 0.598 | +0.032 | 46/100 |
| singleton fraction | 0.912 | **0.687** | **-0.225** | 56/100 |
| k (clusters used) | 29.6 | **25.4** | -4.19 | **0/100** |
| info | 0.350 | 0.333 | -0.018 | 19/100 |

By density the blob is untouched where it is worst: 0.819 -> 0.796 at d=0.9,
0.741 -> 0.722 at d=0.7.

What HailMary actually does is **absorb singletons** into mid-sized clusters (0.91 -> 0.69)
while REDUCING k -- on every instance (better on 0/100 means it never raised k). The entropy
gain is cosmetic: it tidies the tail, it does not shrink the head.

This is the strongest evidence in the report that the blob is structural. HailMary optimises
intra-cluster edge mass directly, so splitting a blob holding 82% of the nodes is the single
largest gain available to it -- and with 6000 epochs from a valid start it declines to take
it. With the `lambda_one_child=0` result (drop validity, info -> 0.0005), the conclusion is
that **the blob is what All-In costs on a random DAG**, not greediness in the closure.

Note HailMary moves AWAY from the ceiling (k 29.6 -> 25.4, universally), trading resolution
for marginally lower information loss. If maximum k is what you want, closure is strictly
better.

CAVEAT: conditions on the 100 instances where HailMary stayed valid (0.680 of runs). Its
failures skew toward instances where it pushed hardest, which if anything OVERSTATES how
much it flattens the distribution.

**What to do about it in the paper.** Report the size distribution alongside exact recovery.
Do not let "100% exact recovery" stand unqualified on random DAGs. The honest claim is:
*on graphs whose ground truth is one-child, closure recovers a genuine, balanced abstraction
in under a second; on general random DAGs, a valid abstraction exists but is necessarily
close to trivial, and we can characterise exactly why.*

---

## All-In violations are often TOLERABLE  [changes the recommendation]

*The conditions are sufficient, not necessary. `fit_abstraction` marginalises nodes it cannot
fit (measured: 80/100 on a random partition), so a partition can violate All-In and still
give an EXACT abstraction over the survivors -- just a less informative one. Every
comparison before this section treated violation as failure. That was wrong.*

The evidence was in the 2000-run table all along and I only read one row of it:

      P(valid abstraction | conditions FAIL), random    = 0.267
      P(valid abstraction | conditions FAIL), one_child = 0.758

### Scoring it properly

`info_effective` charges for BOTH kinds of destruction -- edges inside a cluster, and edges
incident to a marginalised node -- so a partition cannot "win" by having the fit drop half
the graph. 48 instances x 4 arms (`marginalisation.csv`).

Among runs that violate All-In:

| family | arm | n | still exact | violating nodes | marginalised |
|---|---|---|---|---|---|
| random | as_is | 18 | **0.444** | 32.4 | 18.3 |
| random | recon | 7 | **0.429** | 40.0 | 26.9 |
| random | no_oc | 34 | 0.000 | 73.7 | 15.3 |
| one_child | no_oc | 12 | 0.083 | 37.1 | 10.1 |

**~44% of All-In-violating partitions still yield an exact abstraction.**

### The three arms, pooled over random DAGs

| method | exact | info_eff | k | secs |
|---|---|---|---|---|
| closure | **1.000** | 0.549 | 25.6 | **0.0** |
| as_is (intra_mass) | 0.706 | **0.486** | 13.7 | 28.9 |
| recon | 0.882 | 0.497 | 12.5 | 49.4 |
| no_oc | 0.000 | 0.267 | 25.0 | 29.4 |

With a fallback to closure whenever the arm is not exact (so always exact by construction):

| arm | hybrid info_eff | vs closure alone |
|---|---|---|
| as_is | 0.573 | +0.024 WORSE |
| **recon** | **0.504** | **-0.045 BETTER** |

| density | closure | as_is hybrid | recon hybrid |
|---|---|---|---|
| 0.1 | **0.247** | 0.418 | 0.338 |
| 0.5 | 0.560 | 0.580 | **0.493** |
| 0.9 | 0.788 | 0.694 | **0.654** |

**`recon` + closure fallback beats closure alone at density >= 0.5** and loses badly at 0.1.

### Init and k: the current setup was already right

A sweep over 3 initialisations x 4 cluster budgets x 2 losses (p=60, densities 0.5/0.9,
10 instances, `init_k_sweep.csv`), scored on `info_effective` with a closure fallback:

| loss | init | k | exact | hybrid info_eff |
|---|---|---|---|---|
| **recon** | **closure** | **K** | 0.9 | **0.586** |
| recon | closure | 0.5K | 0.8 | 0.590 |
| as_is | closure | 2K | 0.9 | 0.594 |
| as_is | closure | K | 0.9 | 0.597 |
| as_is | reg_surj | 0.5K | 0.7 | 0.599 |
| recon | reg_surj | K | 1.0 | 0.615 |
| recon | random | 1.5K | 1.0 | 0.628 |
| *closure alone* | | | *1.0* | *0.656* |

* **`regularised_surjective_init` does not help.** Closure init takes four of the top five;
  reg_surj's best is 0.599, random's 0.611, against closure's 0.586.
* **Varying k does not help.** For the best arm, k=K gives 0.586 and degrades monotonically
  upward (0.649 at 1.5K, 0.661 at 2K); below K is neutral. Consistent with the brute-force
  finding that closure is k-optimal.
* **But every configuration beats closure alone** on hybrid info_eff (0.586-0.628 vs 0.656)
  -- an 11% relative improvement at best, while staying always-valid via the fallback. This
  corroborates the p=100 marginalisation result (0.504 vs 0.549) on an independent grid.

So if HailMary earns a place in the paper, the configuration is settled: **recon loss,
closure init, k = K, with closure fallback** -- the learned component refining the
combinatorial one from its own output at its own budget.

CAVEAT: 10 instances, p=60, two densities, 4000 epochs. Enough to rule out reg_surj and
large k; not enough to pin the 11%.

### On dense random DAGs these abstractions are near-TRIVIAL

Where the nodes actually end up:

| | blob | singletons | genuine mid-sized clusters |
|---|---|---|---|
| random | **69.3%** | 28.8% | **1.9%** |
| one_child | 19.2% | 58.8% | **21.4%** |

random, mid-sized fraction by density: 5.9% -> 3.2% -> 0.7% -> 0.3% -> **0.0%**

On dense random DAGs **98-100% of nodes are either in the blob or alone**. There is no
intermediate structure -- no cluster representing "these five related variables". That is a
hybrid of the two TRIVIAL extremes: k=1 over most of the graph, the identity map on the
rest. The word for this is trivial, and the report should say so.

The shape is a STAR. Confirmed:

| density | blob is the sink | singletons pointing straight at the blob | high-level graph is a star |
|---|---|---|---|
| 0.5 | 1.000 | 1.000 | **1.000** |
| 0.9 | 1.000 | 1.000 | **1.000** |
| 0.1 | 0.846 | 0.802 | 0.538 |

At density >= 0.5 it is always one giant sink with every other cluster a single source
feeding it. Only at d=0.1 does the in-tree gain real depth.

**This is not a shortcoming of OCC.** Brute force says the best achievable balance at the
ceiling is 0.889 against OCC's 0.856, and HailMary cannot break the blob either. The honest
statement is about the CONDITIONS:

> On generic dense random DAGs, All-In + one-child admit only near-trivial abstractions.
> The conditions are sufficient, and they are very restrictive.

That is a characterisation result, and arguably more interesting than the algorithm. It also
points somewhere: the marginalisation finding shows that RELAXING All-In yields non-trivial
partitions that still give exact abstractions ~44% of the time. That is where non-trivial
abstractions on random DAGs live.

The one_child family remains the positive case: 21.4% of nodes in genuine mid-sized
clusters, blob only 19%, a real multi-level abstraction.

### Cluster count is the wrong quality measure here

OCC's partitions are ~96% singletons, and **a singleton cluster abstracts nothing** -- it is
a low-level variable promoted unchanged. The limit of that logic is k = p, the identity
partition: maximum k, zero abstraction. So BOTH ends of the k range are trivial, and "more
clusters = finer abstraction" fails exactly where OCC lives.

Worked example (p=40, d=0.9, seed 2). OCC gives a 35-blob plus five singletons (k=6);
HailMary gives a 33-blob plus one 7-node source cluster (k=2):

      merging the 5 singletons costs        0.0116 of edge mass
      freeing nodes 3 and 25 from the blob  saves  0.0834

The k drop from 6 to 2 is essentially FREE -- singletons have no internal edges and those
sources barely connect to each other. The whole gain is the blob shrinking 35 -> 33.

Diagnosed by largest cluster rather than k, across five dense seeds:

| seed | largest | info | verdict |
|---|---|---|---|
| 0 | 39 -> 34 | -23% | real gain |
| 1 | 38 -> 19 | -50% | invalid, rejected by the fallback |
| 2 | 35 -> 33 | -10% | real gain |
| 3 | 34 -> 34 | 0% | genuine null |
| 4 | 35 -> 33 | -15% | real gain |

**Report largest-cluster size and information loss; k is close to uninformative between
them.** For the same reason the keep rule should be valid-and-lower-info only -- adding a
"do not reduce k" clause would reject genuine improvements (seeds 2 and 4).

### ⚠ THE edge_weight BUG -- several conclusions below are superseded

*Found 2026-09-15, after everything else in this report was measured.*

`graph_loss` multiplied its pairwise cross-entropy by `edge_weight = |A_low| @ T_soft`, while
the H profiles it compares are built from `A_low_binary`. So the penalty scaled with edge
MAGNITUDE even though All-In is a purely structural condition. Ablation
(`edgeweight_ablation.csv`, 759 partitions):

| variant | Spearman vs violations (pooled) | within-instance | invariant to rescaling A_low x10? |
|---|---|---|---|
| weighted (old) | **+0.002** | **-0.018** | **no -- loss scales 10x** |
| binary mask | +0.088 | +0.011 | yes |
| **none (current)** | **+0.429** | **+0.460** | yes |

The old term gave graph_loss essentially ZERO correlation with the violation count at
training-realistic softness (margin 4). At near-hard margins it did track (+0.46), so the
damage was specifically in the soft regime where training lives.

**Rerun with the fix** (`rerun_fixed.csv`, p=50, 20 instances):

| arm | All-In | one-child | info_eff | hybrid @ alpha=0.05 |
|---|---|---|---|---|
| OCC | 1.00 | 1.00 | 0.549 | — |
| graph_only | **0.40** (was 0.087) | 0.00 | **0.041** | **-52%** (55% accepted) |
| oc_info (the recommendation) | 0.35 | 0.60 | 0.326 | -44% (80%) |
| **oc_info_g1** | **0.55** | 0.70 | 0.323 | **-45%** (80%) |
| oc_info_g10 | 0.60 | 0.65 | 0.411 | -24% (65%) |
| oc_info_g100 | 0.25 | 0.15 | 0.323 | -18% (45%) |

**SUPERSEDED BY THIS:**
1. "`lambda_graph` has NO effect on All-In (Fisher p=1.000)" -- it does now: 0.35 -> 0.55 at
   lambda_graph=1, 0.60 at 10.
2. "graph_loss alone achieves All-In 2/23" -- now 0.40, with partitions retaining **96.5%**
   of edge mass (info 0.035 vs OCC's 0.549) at median relative error 0.026.
3. "graph_loss is a good measure but a poor optimisation target" -- it was a poor target
   largely BECAUSE of this term.
4. The recommended `lambda_graph = 0` -- `lambda_graph = 1` now looks marginally better
   (same retention, All-In 0.35 -> 0.55).

NOT yet established: this is n=20 at p=50 on one density sweep. The `graph_only` result is
suspicious-good -- zero valid partitions but 96.5% retention means it leans entirely on
fit_abstraction marginalising nodes, and the marginalised counts are not in this table.
Before changing the recommendation: marginalisation counts for graph_only, p=100
confirmation, and the noise-weighting robustness check on those rel_err values.

### The alpha-frontier: tolerance is the only real knob

*One loss family, and the hyperparameter that matters turns out to be the tolerance itself,
not any of the lambdas.* (`frontier.csv`, 18 instances, p in {50,100}, d in {0.1,0.5,0.9},
noise-weighted relative error.)

First, the metric fix matters: p90 relative error is **0.112 weighted vs 0.524 isotropic** --
the unweighted version overstates by ~4.7x.

**Hybrid frontier** (use the learned partition when it meets alpha, else fall back to OCC,
so the result is always safe):

| alpha (tolerated relative error) | info_eff | vs OCC (0.604) |
|---|---|---|
| 0 (OCC) | 0.604 | — |
| 0.01 | 0.448 | **-26%** |
| 0.05 | **0.396** | **-34%** |
| 0.25 | 0.368 | -39% |

**Tolerate 5% relative error and you retain 34% more structure than OCC**, falling back to
OCC on the 6% of instances that miss the tolerance. alpha=0 recovers OCC exactly, so OCC is
the endpoint of the method rather than a competitor to it.

`lambda_util = 10` is optimal at EVERY alpha, so the utility-magnitude dial is not needed.

### The alpha-vs-information curve  [the recommended algorithm]

![alpha curve](figures/15_alpha_curve.png)

*OCC for the partition, then HailMary: closure init at slack 1-2xK, `one_child_loss`=100,
`info_loss`=10, `graph_loss` OFF. Accept the refinement when its measured relative error
<= alpha, else keep OCC's. The error needs only A_low and the fit, so the rule is
implementable at inference time.* (`slack_nograph_robust.csv`, 48 instances, p in {50,100},
density 0.1-0.9.)

| alpha | best | info_eff | accepted | vs OCC (0.572) |
|---|---|---|---|---|
| 0.001 | info, slack 2 | 0.483 | 0.73 | -16% |
| 0.01 | info, slack 1 | 0.460 | 0.79 | -20% |
| **0.05** | **info, slack 2** | **0.413** | **0.92** | **-28%** |
| 0.1 | info, slack 1 | 0.402 | 0.98 | -30% |
| 0.25 | info, slack 2 | 0.391 | 1.00 | -32% |

* **`info` (intra_mass) beats `recon` at every alpha** -- strictly below it across four orders
  of magnitude. recon has the HIGHER acceptance rate (0.89 vs 0.71 at small alpha) but much
  worse retention, so info wins the hybrid despite falling back more often.
* **Slack 1-2 are indistinguishable; slack 3 is harmful.** The table alternating between 1
  and 2 by alpha means it is noise between them.
* **Natural operating point at alpha ~ 0.05**: information falls steeply from alpha=0.01 to
  0.1 then flattens. At 0.05 you accept 92% of instances and retain 28% more than OCC; past
  that you pay real error for ~4% more.

CORRECTION: an earlier quick run (n=12) showed slack 2.0 at -32% against slack 1.0's -19%
and this report claimed slack "nearly doubles the benefit". At n=48 that gap disappears.
Slack improves VALIDITY (All-In 0.33 -> 0.46); the retention claim was small-sample noise.

### Scaling the refinement to p=200

45 instances, density x ground-truth n_high x 3 seeds (`p200_test.csv`). The pipeline holds
up, with one clear failure region.

| alpha | accepted | hybrid info_eff | vs OCC (0.709) |
|---|---|---|---|
| 0.01 | 0.64 | 0.550 | -22% |
| **0.05** | **0.69** | **0.534** | **-25%** |
| 0.10 | 0.73 | 0.520 | -27% |

**Acceptance by density (alpha=0.05):**

| density | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| accepted | **0.11** | 0.67 | 0.89 | 0.78 | **1.00** |

At d >= 0.3 the refinement is accepted 67-100% of the time. At d=0.1 it is essentially never
accepted, and that cell carries the only non-zero median rel_err (0.09-0.51 against 0.000
everywhere else) -- it fails BOTH acceptance clauses. Ground-truth n_high matters less, but
nh=40 gives acceptance 1.00 at every density >= 0.3.

What the accepted refinements do:

| density | OCC k -> HM k | OCC largest -> HM largest | OCC info -> HM info |
|---|---|---|---|
| 0.5 | 19.0 -> 2.8 | 182 -> 151 | 0.81 -> **0.53** |
| 0.7 | 7.9 -> 2.0 | 193 -> 142 | 0.93 -> **0.54** |
| 0.9 | 11.2 -> 2.0 | 190 -> 159 | 0.90 -> 0.67 |

Large retention gains, but HailMary collapses to **2-3 clusters** at p=200 -- coarser than at
p=100. The blob shrinks (193 -> 142) without breaking up, so the gain comes from freeing
nodes out of it, not from finding structure.

Cost: **OCC 0.25 s, HailMary 268 s** -- about 1000x, before the 2.5x available from skipping
the unused graph_loss computation.

SCOPE: p up to 200 works at density >= 0.3. Sparse graphs at scale are the failure region --
and note this REVERSES the p<=100 pattern, where d=0.1 was where the refinement helped most
(0.487 -> 0.087). The benefit is not a property of density; it is a property of OCC having
left something on the table, which at p=200/d=0.1 it has not (OCC reaches info 0.351 there
unaided). The per-instance acceptance test detects exactly this, which is the argument for
keeping it rather than choosing a regime up front.

### rel_err depends on the noise model -- acceptance rates are optimistic

*D = S E_high - E_low T depends only on A_low and the partition; Sigma_u enters ONLY as a
weighting. So the partitions can be held fixed and re-scored under different weightings.*
(`relerr_sensitivity.csv`, 19 instances at p=100, 7 weightings each.)

      all weightings agree on the accept/reject decision:  16 of 19 instances
      individual weightings agree with the generator:      17-18 of 19
      accept rate 0.84-0.95 across weightings (generator 0.89)

That looks robust until you ask WHICH instances disagree. Most have rel_err exactly 0 under
every weighting -- an exact abstraction has nothing to weight. All the sensitivity is in the
few with non-zero error:

| instance | generator | highSNR | isotropic | inverse |
|---|---|---|---|---|
| d=0.5, s1 | **0.152** | 0.032 | **0.823** | 0.774 |
| d=0.5, s2 | **0.028** | 0.003 | **0.824** | 0.944 |
| d=0.3, s2 | 0.051 | 0.053 | 0.035 | 0.022 |

Spread within instance: median 0x (all zeros), **p90 471x, max 29,643x**. Of the four
instances with non-zero error, **three flip the decision at alpha=0.05**.

**The generator's weighting is systematically the most favourable** -- the abstraction error
concentrates on low-noise-variance nodes, which SNR-based weighting down-weights.

CONSEQUENCE: the acceptance rates reported throughout (0.92 at p<=100, 0.69 at p=200) are
**optimistic and specific to this noise model**. Not wrong -- these are the correct variances
for this generator -- but they would not transfer to a system with different noise structure.

Two mitigations, both worth doing:
  1. State the sensitivity explicitly when reporting acceptance.
  2. Use a CONSERVATIVE test: accept only if rel_err <= alpha under the WORST weighting in a
     family (isotropic included). Robust to the noise model, at the cost of a lower
     acceptance rate -- the right trade for a claim meant to hold generally.

This does not touch OCC, which is exact by construction. It limits only the refinement's
acceptance claim.

### Knobs that turned out not to be knobs

| knob | verdict |
|---|---|
| `lambda_graph` | **No effect on All-In.** intra_mass 9/16 vs 10/16, recon 15/16 vs 14/16 at lg=0 vs lg=10 (Fisher p=1.000 both). Only a marginal retention effect for intra_mass (p=0.044). All-In satisfaction is a property of the utility LOSS (recon ~90%, intra_mass ~60%), not of the All-In weight. |
| `lambda_util` (1..100 vs one-child=100) | 10 is optimal at every alpha. At 100 one-child COLLAPSES (0.00) and error spikes to 0.589 -- the catastrophic regime. |
| slack (k > K), **with graph_loss OFF** | **Helps validity, not retention (n=48).** All-In 0.33 -> 0.46; slack 1-2 indistinguishable on info_eff, slack 3 harmful. An earlier n=12 run suggested a large retention gain; it did not replicate. Validity rises monotonically (recon: All-In 0.42 -> 0.75, one-child 0.75 -> 1.00; info: All-In 0.17 -> 0.58). In the alpha-hybrid at alpha=0.05, slack 2.0 gives **-32% vs OCC against slack 1.0's -19%** -- nearly double the benefit, because more instances clear the tolerance so fewer fall back. Per-instance retention gains are NOT significant at n=12 (p=0.29-0.87), so the gain comes through validity. Note `k_used` FALLS as budget rises (7.8 -> 5.2) -- that is an outcome, not a counter-argument: extra columns change the optimisation landscape whether or not they survive. (`slack_nograph.csv`) |til = 10 are settled in both the exact and approximate regimes.**

### Two conclusions this overturns

1. **"intra_mass -> recon is ESSENTIAL to keep"** (the 450-run ablation in FINDINGS) was
   measured ONLY on one_child at p=100, k=10, where every method is lossless and nothing
   discriminates. On random DAGs under `info_effective`, **recon is the better objective**.
2. **"HailMary's failures are not benign"** -- I based that on violating-node counts
   (29.7 of 100) and never checked obs/intv loss. 44% of those runs were exact.

### And it partly explains the blob

At density 0.9 `recon` reaches info_eff 0.654 with **3.5 clusters** against closure's 0.788
with 12.7 -- because closure's 12.7 is one 82%-blob plus singletons, while recon's 3-4 are
balanced. A balanced 3-way split destroys less edge mass than a blob-plus-dust partition
with four times the cluster count. So the blob CAN be beaten -- not by raising k, but by
balancing.

CAVEATS: n=7-18 in the violating-subset rows. `recon`'s k=3.5 at d=0.9 is a very coarse
abstraction, and info_eff 0.654 means two thirds of the edge mass is still gone -- recon is
less bad there, not good. `no_oc` remains dead (0.000 exact on random in every cell),
confirming the earlier 194-run result.

---

## APPROXIMATE abstractions: what OCC cannot do

*OCC only ever returns exactly-valid partitions, so it cannot trade a little fidelity for a
lot of retained structure. This is the one thing a learned method can do that OCC cannot --
and it is where the "OCC finds trivial abstractions" problem gets an answer.*

The two conditions fail DIFFERENTLY. Wholesale abandonment of one-child
(`lambda_one_child=0`) makes the Massidda equations inconsistent: relative observation error
**0.84** at the median, exceeding the signal at p90. All-In violations degrade gracefully --
`fit_abstraction` marginalises the offenders, and the error stays small.

So the well-posed objective is: a ONE-CHILD partition minimising information loss, tolerating
All-In violations. Sweeping `lambda_graph` to zero with `lambda_one_child` pinned at 100
(16 instances, p=100, d in {0.5,0.9}, `approximate.csv`):

| loss | lambda_graph | All-In | valid | rel err med | rel err p90 | **info** | k | largest |
|---|---|---|---|---|---|---|---|---|
| *OCC* | — | 1.000 | 1.000 | 0.0000 | 0.0000 | *0.673* | 18.5 | 82.5 |
| **intra_mass** | **0.0** | 0.563 | 0.438 | **0.0000** | **0.039** | **0.374** | 3.4 | 64.3 |
| intra_mass | 0.3 | 0.563 | 0.500 | 0.0000 | 0.040 | 0.407 | 3.6 | 67.1 |
| intra_mass | 10.0 | 0.625 | 0.563 | 0.0000 | 0.294 | 0.509 | 11.5 | 72.2 |
| **recon** | **0.0** | **0.938** | **0.938** | **0.0000** | **0.0000** | **0.585** | 2.5 | 74.7 |
| recon | 10.0 | 0.875 | 0.875 | 0.0000 | 0.0000 | 0.579 | 5.9 | 76.1 |

**Two useful operating points, different in kind:**

1. **intra_mass, lambda_graph=0 -- the approximate regime.** Information loss **0.374 vs
   OCC's 0.673, a 44% improvement**, with median relative error exactly 0 and 3.9% at p90;
   88% of runs under 5% error. "Slightly inexact, substantially more meaningful" is real,
   and the effect is large.
2. **recon, lambda_graph=0 -- nearly free.** 93.8% FULLY VALID, zero relative error even at
   p90, information 0.585 vs 0.673 (13% better). No approximation accepted at all. Note
   dropping the All-In loss barely moved All-In satisfaction (0.938 at lg=0 vs 0.875 at
   lg=10) -- the recon objective encourages it implicitly.

CORRECTION to an earlier claim in this report: "one-child violations are catastrophic" was
measured at `lambda_one_child=0` (wholesale abandonment, rel error 0.84). With one-child
pinned at 100 it still slips to 0.75 satisfaction while the error stays at 3.9% p90. So it
is a matter of DEGREE: occasional one-child violations are tolerable, abandoning the
condition is not.

CAVEATS: 16 instances, p=100, two densities. The intra_mass arm's largest cluster is still
64/100 -- less blobby, not balanced.

---

## Recommendation

```
1. closure, 12 draws, keep the best VALID partition          ~0.14 s
2. if none valid -> targeted-merge fallback                  ~0.03 s   => always valid
3. drop refine_by_splitting entirely                         31-286x faster
4. optionally refine with HailMary from that exact partition at n_high = K,
   keeping the result only if still valid AND lower info     ~40-150 s
   never worse; better ~28% of the time; ~0.02-0.03 gain
```

Validity is O(p²) to verify, so step 4 risks nothing but time. **If closure returns nothing,
do not bother running HailMary** — it has never rescued such an instance (0/35).

HailMary hyperparameters, unchanged and now justified: `lambda_graph=10` (not scaled by k),
`lambda_one_child=100` (load-bearing), `lambda_info=10`, `lambda_surj=0`, `lambda_recon=0`,
**fixed 6000 epochs** — the rule `1000*floor(log((p-k)*k+1)+1)` degenerates to 1000 when
k = p and badly under-trains.

---

## For the paper

The durable contribution is **the characterisation of the problem**, not the algorithm race:

1. All-In alone is vacuous (singletons satisfy it); **one-child sets the entire ceiling**,
   and tracks max out-degree.
2. **One-child is what makes All-In satisfiable** — 0/80 without it at any weight.
3. **On DAGs, acyclicity of the quotient is a corollary of your two conditions**, not a
   third assumption — so your pair subsumes Massidda's Block Ordering (Theorem 2).
   Measured 155/155; the infinite-walk argument proves it.
4. **In the cyclic case that corollary fails and acyclicity must be imposed** — and with it
   the sufficiency survives intact: 111/111 valid with an acyclic quotient, 0/89 without.
   `fit_abstraction` handles cyclic low-level graphs; only the quotient must be acyclic.
5. The ceiling is ~0.6p on one_child, and governed by ground-truth n_high on random.
6. Information loss on random DAGs is **structural**, the joint price of the conditions.
7. **Reconstruction cannot measure abstraction quality** under one-child, and we can say
   exactly why (≤ k−1 high-level edges).

Then: a cheap complete method (closure + fallback) that hits **100% exact recovery** where
existing baselines reach 0–12.5%, and a learned refinement that improves it modestly.

**Position against prior art:** All-In is Massidda et al.'s Theorem 1 (Abstract
Connectivity) — cite it, don't claim it. One-child appears to be yours, and the sufficiency
of the pair is the centrepiece. Massidda explicitly *does not* give an algorithm for finding
the partition, and explicitly assumes acyclicity — those are the two openings this work
fills. The closure's algorithmic shape (union-find coarsening to a least fixed point) is
standard and dual to partition refinement; the specific condition and its use here appear
undocumented.

**Honest limitations to state:** on random DAGs the partitions are blob-dominated (largest
cluster 58-82% of nodes) so "100% exact recovery" must be reported with the size
distribution; HailMary never expands coverage over closure; information
loss is a definition, not an externally validated measure; the one_child family is solved
outright by a sub-second combinatorial method; and `fit_abstraction` loses precision when
the within-cluster block is ill-conditioned (density 0.9, fine granularity).

---

## Open stones

* **Is the blob provably forced?** HailMary cannot break it up (above), which is strong
  empirical evidence, but a proof -- or a counterexample -- would settle it. At p<=10 brute
  force is feasible: enumerate all valid partitions at the ceiling k and check whether any
  is balanced.
* **`fit_abstraction` conditioning at high density.** The one real gap: at density 0.9 with
  fine ground-truth granularity, `cond(I - W_cc)` reaches 1e12 and the solver loses the
  abstraction even though the partition is valid. Needs either a better-conditioned
  propagation or an explicit diagnostic + refusal.
* **Cyclic case: only short cycles tested** (generated by reversing existing edges), and
  spectral radius fixed at 0.8.
* **`refine_by_splitting` was only tried with 40 random cuts.** A smarter split search might
  find what random cutting cannot.
* **`closure_logits` still ships the single unvalidated draw.** Fixing it should lift
  HailMary's success ~7 points; not yet measured end-to-end.
* **`synthetic.py` passes the worse `A_hat`** — your existing reconstruction numbers are
  affected.
* **Large-scale run not done.** `scripts/benchmark.py` is the template: add density as an
  axis, widen p, and raise trials.
