# Partition-learning baselines: algorithms, findings, and why each choice was made

Companion to `FITTER_SWITCH.md`. That document covers the **T**-fitting half of the
pipeline; this one covers partition learning — the combinatorial baselines, how they
compare to HailMary, and the design decisions that were settled empirically rather than by
argument.

Everything here is measured at `p=100`, 10 instances per density over
`d in {0.1, 0.3, 0.5, 0.7, 0.9}` (47 of 50 instances; 3 failed graph generation at d=0.1).
Unless stated otherwise every method runs under the **same budget** `k = |pi_0|`, AOC
Closure's own cluster count on that instance, because singletons trivially satisfy All-In
at zero intra-cluster mass and the comparison is meaningless without a budget.

---

## 1. Why community detection is the wrong family

All-In holds for a cluster iff every member has the **same child-cluster signature** — if
any member points into **X**_pi(y), all must. That is a regular/structural-equivalence
(bisimulation) condition, not a community-detection one.

Community detection — modularity, normalised cut, METIS — **maximises intra-cluster edge
mass**, which is precisely the information loss being minimised. Those methods optimise the
wrong sign of the objective. `spectral-ncut` is retained as a negative control and behaves
exactly as predicted: the worst method measured (intra 0.212, 86 All-In violating nodes).

Louvain (Blondel et al. 2008) and METIS (Karypis & Kumar 1998) were not run because neither
accepts a cluster budget natively; `spectral-ncut` (Shi & Malik 2000) represents the family.

---

## 2. The algorithms

Implementations in `scripts/partition_baselines.py` unless noted.

| method | targets | source |
|---|---|---|
| `oc_closure` | one-child | `occ.py` |
| `occ` (AOC Closure) | one-child + All-In + acyclic | `occ.py` |
| `allin_agglom` ("Greedy All-In") | All-In | new |
| `greedy_merge` | scalarised combinations | new |
| `bisim` | All-In (colour refinement) | new |
| `sig_agglom`, `sig_kmeans` | All-In (structural equivalence) | new |
| `spectral_ncut` | NEGATIVE CONTROL | new |

### Greedy All-In (`allin_agglom`)

Merges from singletons to the budget, at each step choosing the merge leaving fewest All-In
violations, ties broken by the smaller increase in intra-cluster edge mass. The objective is
the **violation pair (incidence) count**

```
J = sum over quotient edges X_x -> X_y of  #{ i in X_x : i has no edge into X_y }
```

so a node failing three different children counts three times, and the term scales with
|X_x| — large-cluster violations are penalised **more**.

Candidate evaluation is exact; the candidate **set** is shortlisted to the `n_cand=120`
most out-neighbourhood-similar cluster pairs, because scoring all O(k^2) pairs exactly at
every one of the p-k steps is prohibitive. Shortlisting by similarity is aligned with the
objective rather than neutral — All-In is a structural-equivalence condition, so similar
out-neighbourhoods are where violation-free merges live.

**Complexity** (see §6): time `O(p^3 + C p^2 (p-k))`, memory `O(p^2)`.

### Citations for the classical baselines

* Colour refinement / equitable partition (`bisim`) — Weisfeiler & Leman (1968); Paige &
  Tarjan (1987), *Three partition refinement algorithms*, SIAM J. Comput. 16(6):973–989;
  Boldi & Vigna (2002), *Fibrations of graphs*, Discrete Math. 243:21–66.
* Structural equivalence (`sig_agglom`, `sig_kmeans`) — Lorrain & White (1971), J. Math.
  Sociol. 1(1):49–80; regular equivalence White & Reitz (1983), Social Networks 5(2):193–234.
* Average linkage — Sokal & Michener (1958). k-means — Lloyd (1982); k-means++ init
  Arthur & Vassilvitskii (2007).
* Normalised cut — Shi & Malik (2000), IEEE TPAMI 22(8):888–905; von Luxburg (2007).

Page numbers were not verified against the sources — check before they go in the .bib.

---

## 3. The headline finding: the conditions are a cliff at alpha = 0, and a dial above it

Pooled over 893 scored partitions (`greedy_variants.csv` + `greedy_init.csv`), acceptance
rate at tolerance alpha and median information loss of the accepted runs:

| cell | n | a=1e-6 | a=1e-3 | a=0.01 | a=0.05 | a=0.1 | a=0.3 |
|---|---|---|---|---|---|---|---|
| both conditions | 149 | **0.87** | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 |
| → info loss | | 0.632 | 0.596 | 0.592 | 0.592 | 0.592 | 0.592 |
| All-In only | 181 | **0.04** | 0.25 | 0.40 | 0.55 | 0.64 | 0.81 |
| → info loss | | 0.627 | 0.262 | 0.230 | **0.190** | 0.164 | **0.100** |
| neither | 555 | 0.01 | 0.14 | 0.20 | 0.30 | 0.37 | 0.57 |

Two different maps, and they must not be conflated:

1. **Conditions → exactness is a cliff.** At alpha = 1e-6, satisfying both conditions gives
   0.87 acceptance; satisfying only All-In gives **0.04**. A near-miss on one-child does not
   give a near-exact abstraction.
2. **Tolerance → information loss is a smooth, favourable trade.** The both-conditions row
   is **flat at ~0.592 for every alpha** — a user who insists on exactness has no dial at
   all. The near-miss row falls 0.627 → 0.100 as alpha relaxes.

So approximate satisfaction of the conditions buys nothing at alpha → 0 and a great deal at
alpha > 0. That is what the method family exposes, and the flat row is the evidence that
the sufficient conditions alone leave a user stuck.

Verification of the sufficiency theorem itself: among partitions satisfying All-In,
one-child and quotient acyclicity, **max RAE = 0.0124** over 149 partitions (vs 1053 for
invalid ones) and acceptance is 1.00. The 13.4% that miss an exact-zero threshold sit at
RAE 1e-6 to 1.2e-2 — SLSQP convergence at `maxiter=500, n_restarts=5`, not counterexamples.

**All-In is not strictly necessary**: 2 of 117 All-In-violating partitions reach RAE ~1e-10.
Both are degenerate (information loss 0.87 and 0.73, i.e. nearly everything marginalised).

**Quotient acyclicity is NOT implied by one-child.** Out-degree <= 1 gives a functional
graph, which admits cycles (X->Y->X via x1->y1 and y2->x2, both consistent with a low-level
DAG). It held on 149/149 instances here, but that is a regularity of the generator, not a
theorem — do not drop the clause on the strength of it.

---

## 4. Method comparison

### On the graph conditions (`partcmp.csv`, figures `paper_allin`, `paper_onechild`)

| method | All-In viol. | one-child excess | intra | exact All-In | secs |
|---|---|---|---|---|---|
| AOC Closure | **0** | **0** | 0.559 | **1.00** | 0.0 |
| all-in greedy | **0** | 76 | **0.000** | 0.81 | 7.3 |
| HM graph-only | 20 | 87 | 0.005 | 0.06 | 16.1 |
| sig-agglom | 22 | 67 | 0.000 | 0.11 | 0.01 |
| HailMary (all λ) | 23 | 56 | 0.019 | 0.05 | 28.1 |
| sig-kmeans | 23 | 69 | 0.016 | 0.15 | 0.08 |
| bisim | 58 | 97 | 0.017 | 0.21 | 0.003 |
| hc | 77 | 90 | 0.291 | 0.02 | 2.1 |
| spectral-ncut | 86 | 222 | 0.212 | 0.02 | 0.05 |
| UCA / random | 87 / 88 | 270 / 295 | 0.043 / 0.039 | 0.00 | — |

On the **All-In axis HailMary loses** to a 40-line greedy that is also 2–5x faster.
On the **one-child axis HailMary is the only method with a dial**: every equivalence method
is pinned at 67–97 excess child-edges and cannot move, while HailMary sweeps 119 → 3 at
d=0.1 holding intra at ~0.00.

`bisim` degrades sharply with density (0 violations at d<=0.3, 65 at d=0.5): colour
refinement cannot reach signature-uniformity under a tight budget — it runs out of clusters
before it runs out of violations.

### Through the full pipeline (`greedy_variants.csv`)

| method | All-In | one-child | intra | RAE | info_eff | acc a=.05 | exact |
|---|---|---|---|---|---|---|---|
| OCC | **0** | **0** | 0.559 | **0.000** | 0.596 | **1.00** | **91.5%** |
| all-in greedy | **0** | 74 | **0.000** | 0.044 | **0.066** | 0.51 | 8.5% |
| joint greedy | **0** | 32 | **0.000** | 0.104 | 0.082 | 0.32 | 2.1% |
| min-intra greedy | 29 | 104 | **0.000** | 0.402 | 0.097 | 0.09 | 0% |
| one-child greedy | 17 | 26 | 0.007 | 0.368 | 0.246 | 0.11 | 6.4% |

**Every greedy wins the plotted axes and loses the abstraction.** This is the answer to
"why not just run a greedy": near-satisfaction of one-child is worth nothing for exactness.

`joint greedy` (scalarising all three) is **dominated by `all-in greedy`** on exactness,
acceptance and information loss — which is why it is cut from the main text rather than
merely being "too complex".

---

## 5. Design decisions settled empirically

### 5.1 OC Closure selects on information loss (CHANGED)

`one_child_closure` is order-dependent, so `n_draws` random orders give different
partitions. All are one-child by construction, so **all draws are eligible**.

Selecting the min-information-loss draw rather than the widest (the obvious proxy):

| d | intra (widest) | intra (min-loss) | gain |
|---|---|---|---|
| 0.1 | 0.051 | 0.051 | 0.000 |
| 0.3 | 0.213 | 0.179 | 0.034 |
| 0.5 | 0.505 | **0.399** | **0.106** |
| 0.7 | 0.624 | 0.574 | 0.050 |
| 0.9 | 0.743 | 0.743 | 0.000 |

Pooled **0.542 → 0.465**, better on 11/47 instances, **never worse**, k unchanged. Adopted
as `oc_closure()`. Cost is ~9% of draw generation (217 ms to score 564 draws vs 2.4 s to
generate them). Gains are concentrated at mid-density, where the draws actually differ.

### 5.2 AOC Closure keeps select-then-repair (UNCHANGED)

Two alternatives were tested and both rejected:

* **Select valid draws on information loss instead of width**: +0.000 median, better on
  2/47. No effect.
* **Repair every invalid draw, then select least information loss**: **identical on 47/47**
  instances for 28x the repair calls (0.2 → 5.6 per instance) and 14x the selection time.

The reason is that **repair is confluent here**: whatever draw it is given, it merges the
blob with its child the same way and lands on the same partition. The draws differ only in
where singletons land, and repair does not touch singletons.

So selection is valuable at the OC stage and worthless at the AOC stage.

### 5.3 What `n_draws` actually buys

Distinct partitions from 12 draws, and how many satisfy All-In:

| d | seed | distinct partitions | k range | valid |
|---|---|---|---|---|
| 0.1 | 0 | 1 | 36–36 | 12/12 |
| 0.5 | 0 | **11** | 22–23 | **2/12** |
| 0.9 | 0 | 1 | 7–7 | 12/12 |

Up to 11 distinct partitions, yet k varies by at most 1. **Redrawing buys VALIDITY, not
partition quality** — avoiding the repair stage is what `n_draws` is for.

### 5.4 The All-In violation weighting does not matter (`allin_objective.csv`)

Three weightings of the same violations, identical greedy otherwise:

| weighting | RAE | intra | acceptance | secs |
|---|---|---|---|---|
| `pairs` (adopted) | 0.124 | **0.049** | 0.404 | 8.1 |
| `nodes` | 0.115 | 0.065 | 0.426 | 24.9 |
| `dilution` | **0.088** | 0.079 | **0.468** | 7.7 |

Every paired median difference is exactly 0.0000; the methods produce identical partitions
on 26–37 of 47 instances, and differ only at d >= 0.7 where clusters are large enough for
size-weighting to bite. `nodes` has no effect and is 3x slower. `dilution` trades
information for exactness (RAE better 15/6, p=0.079; intra worse 2/12, p=0.003).

`pairs` retained: best information loss, as fast as anything, no alternative beats it on a
median. Honest caveat: dilution's RAE advantage is suggestive and might reach significance
with more instances.

### 5.5 AOC Closure's All-In stage: what it costs and buys (`occ_ablation.csv`)

Paired, same instances, both under the adopted selection criteria (`oc_closure` selects
min-information-loss; AOC selects widest All-In-valid then repairs).

| d | OC intra → AOC intra | OC exact → AOC exact | repair fired |
|---|---|---|---|
| 0.1 | 0.051 → 0.249 (**4.9x**) | 0.29 → 0.71 | 71% |
| 0.3 | 0.179 → 0.316 | 0.50 → 0.80 | 40% |
| 0.5 | 0.399 → 0.558 | 0.40 → 1.00 | 20% |
| 0.7 | 0.574 → 0.660 | 0.80 → 1.00 | **0%** |
| 0.9 | 0.743 → 0.743 | 1.00 → 1.00 | **0%** |

Pooled: intra 0.465 → 0.559, info_eff 0.667 → 0.596, exact 61.7% → 91.5%, acceptance
0.915 → 1.000, exact All-In 25.5% → 100%.

**Paired All-In stage delta: intra +0.040, exact +30 pp, acceptance +9 pp.**

**One-child does the work; All-In is a 62% → 92% amplifier that binds only at low density.**
At d >= 0.7 the stage is entirely free — repair never fires, so every closure draw already
satisfied All-In, and intra is unchanged at d=0.9. At d = 0.1 it costs 4.9x the information
loss.

Note the §5.1 change widened the gap the All-In stage must justify: the paired intra cost
rose from +0.022 (widest-draw OC Closure) to **+0.040**, because OC Closure improved and AOC
did not. The conclusion is unchanged — 30 pp of exactness for 4% more intra-cluster mass is
a good trade — but quote the current number.

---

## 6. Complexity

| | time | memory |
|---|---|---|
| all-in greedy | `O(p^3 + C p^2 (p-k))` → **O(p^3)** | **O(p^2)** |
| HailMary graph-only | **O(E b p^2 k)** → O(E b p^3) | **O(b p^2 k)** |

Greedy: per-candidate evaluation `O(p^2)`, per merge iteration `O(C p^2)`, exactly `p-k`
iterations, plus a one-off `O(p^3)` for the similarity matrix. Note k enters only through
`(p-k)` — a tighter budget makes it **cheaper**.

HailMary: set by the `(b, p, p, k)` tensors in `forward` ([hail_mary.py:325](../../src/models/hail_mary.py#L325), `:331`).

Greedy is cheaper iff `C(p-k) < E b k`, i.e. **k > p/26** at C=120, E=3000, b=1. Every
density in this study has k between 0.07p and 0.40p, so the greedy is analytically cheaper
throughout, and 2–5x faster measured (p=40: 0.5s vs 2.4s; p=220: 16.1s vs 53.9s). Memory is
the sharper separation and has no crossover: `O(p^2)` against `O(p^2 k)`.

Measured exponents over p in [40, 220] are ~2.0 for both, because per-call numpy overhead in
the k-length Python loops dominates the asymptotic term at these sizes.

---

## 7. HailMary initialisation (`greedy_init.csv`)

Three inits, same instances, same budget (verified: k identical across all methods on 47/47),
lambda_oc in {0, 10, 100, 1000}. Hybrid information loss at alpha=0.05, one **fixed** lambda
across all densities:

| lambda_oc | greedy-init | closure-init | random-init |
|---|---|---|---|
| 0 | **0.448** | 0.483 | 0.503 |
| 100 | 0.450 | 0.496 | 0.464 |
| 1000 | 0.464 | 0.483 | 0.479 |

AOC Closure baseline: 0.529.

**Greedy init beats closure init at every fixed lambda** and wins 3 of 5 densities on the
per-density best — but the margin is 7% over closure init, not the 2.7x the d=0.1 slice
alone suggests. The largest single win is d=0.1 (0.074 vs OCC's 0.277, 3.7x); by d=0.7 all
three inits converge to ~0.57–0.62 and the ordering scrambles.

The best `(init, lambda)` pair is **not stable across density** — (greedy, 0) at d=0.1,
(random, 100) at d=0.3, (greedy, 100) at d=0.5. Quote a fixed configuration and state the
gap, as `paper_alpha_main` does; do not per-instance select.

Pooled, greedy init inherits the greedy's informativeness (iem 0.166, best of the three)
and loses exactness for it (6.4% exact vs closure-init's 17.6%). The same trade, moved.

---

## 8. Known issues

* **Name collision.** `all-in greedy` in `partcmp.csv` is `allin_agglom` (lexicographic
  tie-break on added intra mass); in `greedy_variants.csv` it is `greedy_merge(1,0,0)` (pure
  scalar, no tie-break). **Do not merge those tables without relabelling.** Pooled intra is
  0.000 for both, so the tie-break appears immaterial, but they are different algorithms.
* **Two agglomerative frameworks.** `clustering.py:hierarchical_clustering` already provides
  pluggable greedy machinery via `loss_list`, and the new greedies duplicate it. A naive port
  would be **incorrect**, though: that function caches each pair's loss at push time and
  invalidates only by `cluster_version` of the two clusters merged. That is fine for a local
  loss like `heuristic_loss`, but a condition count is **global** — merging (c1,c2) changes
  the violation count of pairs involving neither, and those stale values are never
  recomputed. A correct port must drop the heap and rescore each step.
* **Model class vs tested class.** The synthetic generator produces graphs with ~50% negative
  weights and rho(A) = 0 exactly (DAGs are nilpotent). The non-negative cyclic class is
  exercised **only** on Leontief.
