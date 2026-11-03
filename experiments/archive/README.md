# Abstraction study — 2026-09-09 → 09-30

The historical record of everything we tried before settling on the current
implementations (`src/algos/greedy.py` for partitions, `src/algos/T_fitter.py` for T).

**Start here: [FINDINGS.md](FINDINGS.md)** — the concise narrative of how we got to the final pipeline.
**Detail, 09-10 → 09-17: [REPORT.md](REPORT.md)** — question → answer → evidence, with figures.
**The fitter switch: [FITTER_SWITCH.md](FITTER_SWITCH.md)** — which loss conclusions survive
`process_output` moving to `fit_abstraction_alternative` (2026-09-15), plus two metric bugs
that affect earlier numbers.
**Partition baselines: [PARTITION_BASELINES.md](PARTITION_BASELINES.md)** — the combinatorial
partition learners (OC Closure, AOC Closure, Greedy All-In, structural-equivalence and
community-detection baselines), how they compare to HailMary, complexity, and the selection
criteria settled empirically (2026-09-22).
Dense reference: [FINDINGS.md](../../results/smoke_tests/20260911-235555_FINAL/FINDINGS.md).

## Layout

    FINDINGS.md    concise narrative, start -> final pipeline
    REPORT.md      detailed report, question -> answer -> evidence (09-10 -> 09-17)
    scripts/       every experiment, one file per question
    data/          the CSV each script produced (+ .log where useful)
    figures/       PNGs used by the report

## Headline

`closure` (one-child merge closure + targeted-merge fallback) reaches **100% exact
recovery** across density 0.1-0.9, p up to 400, and every blockiness/heterogeneity cell
tested; no existing baseline exceeds 12.5%. HailMary on top is a strictly non-negative
refinement worth ~0.02-0.03 information, at ~200x the runtime.

**Cyclic low-level graphs:** All-In + one-child + *acyclic quotient* is sufficient
(111/111 valid with it, 0/89 without). `fit_abstraction` handles cycles; only the quotient
must be acyclic. On DAGs that third condition is free.

1666 runs across 6 tiers.

## Which script answers what

| question | script | data |
|---|---|---|
| **Large benchmark: density x p x n_high, all methods** | `large_benchmark.py --tier main` | `large_main.csv` |
| **Scaling to p=200/400** | `large_benchmark.py --tier scale` | `large_scale.csv` |
| **Blockiness x heterogeneity** | `large_benchmark.py --tier shape` | `large_shape.csv` |
| **The cyclic case (generator + sufficiency)** | `cyclic.py`, `cyclic.py --heavy` | `cyclic.csv`, `cyclic_heavy.csv` |
| Does All-In + one-child imply an acyclic quotient? | `cyclecheck.py` | `cyclecheck.csv` |
| Baselines vs closure vs closure+HailMary (first pass) | `benchmark.py` | `benchmark.csv` |
| Does the targeted-merge fallback close closure's failures? | `fallback.py` | `fallback.csv` |
| How to choose k; ceiling vs p and n_high | `kpolicy.py` | `kpolicy.csv` |
| Closure alone, with/without refine_by_splitting | `ceilbase.py` | `ceilbase.csv` |
| Closure partition -> HailMary (the intended pipeline) | `warm.py` | `warm.csv` |
| Does slack (k > K) help? | `slack.py` | `slack.csv` |
| Estimator failure rate and time scaling | `estscale.py` | `estscale.csv` |
| Why do closure failures happen? | `bimodal.py` | `bimodal.csv` |
| Density sweep | `density.py` | `density.csv` |
| HailMary vs closure across density | `lowdens.py` | `lowdens.csv` |
| Do restarts help? | `restarts.py` | `restarts.csv` |
| Violation severity; closure vs random init | `violinit.py` | `violinit.csv` |
| Initialisation margin/noise sweep | `marginsweep.py` | `marginsweep.csv` |
| Drop one-child; scale lambda_graph up | `graphweight.py`, `nochild.py` | `graphweight.csv`, `nochild.csv` |
| Does recon track intra-cluster mass? | `metric.py` | `metric.csv` |
| Three information measures compared | `infometric.py` | `infometric.csv` |
| one_child_loss variants | `ocnorm.py`, `cminus1.py`, `leak.py` | matching CSVs |
| Max-k brute force / estimator | `maxk_scale.py` | `maxk.csv` |
| T_soft training animation (+ losses & hard metrics) | `animate_T2.py` | `results/timelapse/*.gif` |
| **Are All-In violations tolerable? (marginalisation)** | `marginalisation.py` | `marginalisation.csv` |
| **⚠ The edge_weight bug in graph_loss** | `edgeweight_ablation.py` | `edgeweight_ablation.csv` |
| **Rerun with the fixed graph_loss** | `rerun_fixed.py` | `rerun_fixed.csv` |
| Does one_child_loss track satisfaction? | `onechild_tracks.py` | `onechild_tracks.csv` |
| Does graph_loss track All-In? | `graphloss_tracks.py` | `graphloss_tracks.csv` |
| **Is rel_err robust to the noise model?** | `relerr_sensitivity.py` | `relerr_sensitivity.csv` |
| **The pipeline at p=200 (density x n_high)** | `p200_test.py` | `p200_test.csv` |
| **graph_loss as the sole objective** | `graph_only.py` | `graph_only.csv`, `graph_only_err.csv` |
| **The alpha-vs-information curve (FINAL)** | `slack_nograph.py --robust`, `plot_alpha_curve.py` | `slack_nograph_robust.csv` |
| **The alpha-frontier (tolerance vs retention)** | `frontier.py` | `frontier.csv` |
| **Does extra k help with graph_loss OFF?** | `slack_nograph.py` | `slack_nograph.csv` |
| **Approximate abstractions: relax All-In, keep one-child** | `approximate.py` | `approximate.csv` |
| **Does a different init or k help HailMary?** | `init_k_sweep.py` | `init_k_sweep.csv` |
| **Is the blob forced? (exhaustive at p<=10)** | `blob_bruteforce.py` | `blob_bruteforce.csv` |
| **OCC: the complete method, one file** | `../occ.py` | — |
| Inspect closure on a toy graph | `explore.py` | — (prints; `--brute` enumerates) |
| **Cluster-size distributions (the blob)** | `clustersizes.py` | `clustersizes.csv` |
| **Can HailMary break up the blob?** | `hm_sizes.py` | `hm_sizes.csv` |
| **Loss conclusions under BOTH fitters (paired, same partitions)** | `lossfit.py`, `analyse_lossfit.py` | `lossfit.csv` |
| **Drop one-child and restart until a fit certifies it?** | `nochild_restarts.py`, `analyse_nochild.py` | `nochild_restarts.csv` |
| **graph+recon, graph+recon+one_child, recon alone** | `graph_recon.py` | `graph_recon.csv` |
| **lambda_graph : lambda_recon ratio (the loss terms differ by 4e4)** | `ratio.py` | `ratio.csv` |
| **Is the alternative's error floor just SLSQP budget?** | `altbudget.py` | `altbudget.csv` |
| **Restart/maxiter budget; is n_restarts=5 safe?** | `altbudget2.py`, `restartcheck.py` | `altbudget2.csv`, `restartcheck.csv` |
| `fit_abstraction` vs `fit_abstraction_alternative` (first pass) | `fitcmp.py` | `fitcmp.csv` |
| **Partition baselines vs HailMary on the conditions** | `partcmp.py`, `plot_partcmp.py` | `partcmp.csv`, `figures/paper_allin*.png`, `figures/paper_onechild*.png` |
| **Greedy All-In through the full pipeline (RAE, info_eff)** | `greedy_pipeline.py` | `greedy_pipeline.csv` |
| **Scalarised greedy variants (min-intra / one-child / joint)** | `greedy_variants.py` | `greedy_variants.csv` |
| **Does the All-In violation weighting matter? pairs/nodes/dilution** | `allin_objective.py` | `allin_objective.csv` |
| **OC Closure draw selection: widest vs min-information-loss** | `occ_select.py`, `occ_select2.py` | `occ_select.csv`, `occ_select2.csv` |
| **AOC Closure's All-In stage: what it costs and buys** | `occ_ablation.py` | `occ_ablation.csv` |
| **HailMary init: greedy vs closure vs random** | `greedy_init.py` | `greedy_init.csv` |
| **Hard (argmax) copy of graph/recon loss added to the soft one** | `hardmix.py`, `plot_hardmix.py`, `animate_hardmix.py` | `hardmix.csv`, `hardmix_w0.1.csv`, `figures/hardmix*.png/gif` |
| **Compression (k_used/p) by density and method** | `plot_compression.py` | `figures/paper_compression.png` |
| Report figures | `make_figures.py`, `analyse_large.py` | `figures/*.png` |
| *Earliest phase (09-10 → 09-12), outputs in `results/smoke_tests/`, `results/diagnostics/`* | | |
| Surjective loss / initialisation smoke test | `smoke_test_surj_init.py`, `analyse_smoke_test.py` | `results/smoke_tests/*` |
| Sweep D (recon vs surjective, connected graphs); final config choice | `analyse_sweep_d.py`, `analyse_final.py` | `results/smoke_tests/*` |
| Why is All-In minimised on random DAGs but not one_child? | `diagnose_graph_loss.py` | `results/diagnostics/` |
| Max-k lower bound (closure + greedy splitting); imported by many scripts | `max_k_estimate.py` | — |
| *Final phase (09-23 → 09-30): T-fitting and paper outputs* | | |
| Three-way fitter comparison; faithful `_K` | `fit3cmp.py`, `kfaith.py`, `recon_blockiness.py` | `fit3cmp.csv`, `kfaith_b1.csv` |
| How random partitions reach RAE = 0 | `randpart_probe.py`, `emit_probe.py`, `mechanism_split.py` | `randpart_probe.csv`, `emit_probe.csv` |
| `fit_abstraction_auto` on GT vs random partitions | `fitter_gt_random.py`, `plot_/table_fitter_gt_random.py` | `fitter_gt_random.csv` |
| Proper cyclic ground truth generator | `cyclic_gen.py` | — |
| **Eigen fitter vs `_K` vs `auto`, five partition types** | `eigen_bench.py` | `eigen_bench.csv` |
| **T-fitter vs oracle as p and k grow (paper figure)** | `tk_sweep.py`, `plot_tk_sweep.py` | `tk_sweep_eigen.csv`, `figures/paper_tk_sweep_compact.pdf` |
| **Partition learners on the structural conditions (paper table)** | `partition_quality.py`, `table_partition_only.py` | `partition_quality_k20.csv` |
| **Partition learners vs p and k (paper figure)** | `plot_partition_sweep.py` | `eigen_bench.csv` |
| **α-frontier at blockiness 0.8 (paper figure)** | `alpha_b08_k10.py --fitter eigen` | `alpha_b08_k10*_eigen.csv` |
| **Leontief (EXIOBASE3, GB) high-level graph (paper figure)** | `leontief_tikz.py`, `leontief_graph_fig.py` | `figures/leontief_tikz_eigen.tex` |
| Leontief exploration (thresholding, SCC core, restarts) | `leontief_*.py`, `plot_leontief_*.py` | `leontief_*.csv` |

## Reproducing

Scripts are standalone and write into `data/`. From the repo root:

    venv/bin/python experiments/abstraction_study/scripts/large_benchmark.py --tier main
    venv/bin/python experiments/abstraction_study/scripts/cyclic.py --heavy
    venv/bin/python experiments/abstraction_study/scripts/make_figures.py
    venv/bin/python experiments/abstraction_study/scripts/analyse_large.py

Most use `multiprocessing.Pool(6)` and pin BLAS threads to 1; adjust for your machine.
`scripts/max_k_estimate.py` is imported by path from many scripts; keep it in `scripts/`.

**Scripts from before the `src/models` → `src/algos` refactor will not run on the current
tree.** 74 of them import `src.models.hail_mary` (HailMary, now `src.algos.gradcap.GradCAP`)
or `src.models.wrapper`. Check out commit `0d402b4` to reproduce them. Several older scripts
also wrote to, or `exec` code from, session scratchpads that no longer exist (`grep -l /tmp/`);
their CSVs were copied into `data/` and are the record.

## Caveats carried by this data

* **MOST RESULTS PREDATE THE edge_weight FIX** (src/models/hail_mary.py, 2026-09-15). Anything
  involving `lambda_graph` or `graph_loss` measured before that date used a version whose
  correlation with All-In violations was ~0 in the soft regime. See REPORT.md, "THE
  edge_weight BUG".

* Random-DAG results outside `density.py` / `lowdens.py` fix **density = 0.5**, which is the
  easy regime for closure.
* **Acceptance rates are specific to this generator's noise model.** rel_err weights by the
  per-node noise variance; on inexact instances a different weighting changes it by up to 30x
  and flips the accept decision. Use a worst-case weighting for claims meant to hold generally.
* The two conditions are **sufficient, not necessary**: ~44% of All-In-violating partitions
  still give an exact abstraction, because `fit_abstraction` marginalises the offenders.
  Score those with `info_effective` (intra-cluster mass + mass incident to marginalised
  nodes), not `info`.
* Information loss must be read **conditional on validity** — failing runs score better on
  it. See REPORT.md, "the measurement trap".
* `experiments/synthetic.py` passes the worse `A_hat` to `reconstruction_loss_with_data`.
* `fit_abstraction` loses precision when `cond(I - W_cc)` is large (density 0.9 + fine
  granularity): the partition is valid but obs/intv loss blows up. Report the condition
  number as a diagnostic.
* **`info_eff` is NOT comparable across fitters.** It tests marginalisation at EXACT zero;
  `fit_abstraction_alternative` never returns a hard zero, so it scores as retaining mass it
  has discarded (understated by 0.07-0.29). Use a 1e-2 tolerance. See FITTER_SWITCH.md.

* **`faithful=True` IS NOW THE DEFAULT (2026-09-23)** in `process_output`,
  `fit_abstraction` and `fit_abstraction_alternative`. The old objective was T-UNFAITHFUL:
  it attenuated 62-89% of the true cross-cluster edges while still achieving RAE = 0, and
  reconstructed A at 0.42-0.96 relative error against ~1e-12 for the new one. **Every CSV in
  `data/` predates this and used the old objective**; pass `faithful=False` to reproduce.
  Costs 2-6x more time. See FITTER_SWITCH.md, "2026-09-23".

* **OC Closure's selection criterion CHANGED on 2026-09-22.** `occ.oc_closure()` now picks
  the min-information-loss draw, not the widest (0.542 -> 0.465 intra pooled, never worse,
  k unchanged). Anything that hand-rolled "widest of n draws" for the one-child-only method
  predates this and is not comparable; `occ()` (AOC Closure) is UNCHANGED. See
  PARTITION_BASELINES.md §5.1-5.2.

* **`all-in greedy` names two different algorithms.** In `partcmp.csv` it is `allin_agglom`
  (lexicographic tie-break on added intra mass); in `greedy_variants.csv` it is
  `greedy_merge(1,0,0)` (pure scalar, no tie-break). Do not merge those tables without
  relabelling.

* **The synthetic generator does not exercise the non-negative cyclic model class.** It
  produces ~50% negative weights and rho(A) = 0 exactly (DAGs are nilpotent). Only the
  Leontief data tests cyclic non-negative models.

* **`fit_abstraction_alternative` defaulted to `seed=None`** so the pipeline was not
  reproducible; re-scoring identical partitions moved exactness by +-0.04 and A_high entries
  by up to 22. Fixed 2026-09-16: `process_output` forwards `seed` (default 0).

* **`is_dag` tests `A != 0`**, so the alternative's numerical dust (1e-16..1e-19) fabricates
  cycles. `fitcmp.csv`'s `alt_dag = 0.242` is an artifact; threshold at 1e-9 first.

* `reverse_topo_order` has NO cycle detection -- on a cyclic quotient it silently returns an
  invalid order. Guard any cyclic-case code.
