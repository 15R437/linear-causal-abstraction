"""Panel (a) of the main table on its own: the combinatorial partition learners.

Split out of `table_combined.py` because GradCAP is scrapped and the T-learning results have
moved to a figure (`plot_tk_sweep.py`), so the two panels no longer share a table. The
numbers, the metric definitions and the density argument are unchanged from that script --
this reads the SAME `partition_quality_k20.csv`, which the current code still reproduces
exactly (verified row-for-row on seeds 0-2, 2026-09-25), so no rerun was needed.

Columns, all on a common [0,1] "fraction of the ideal" scale, higher better:
    All-In     1 - allin_pairs / max_pairs
    1-child    1 - oc_excess  / (n_qedges - 1)
    info       1 - intra_edges          (unweighted EDGE COUNT, not mass)

NOTE the `info` column counts EDGES while both learners tie-break on edge MASS
(`occ.information_loss`, `partition_baselines.allin_agglom`). The two agree closely here but
are not the same quantity; `intra` is the mass column if that is wanted instead.

DENSITY: a SINGLE density, not pooled -- `info` runs 0.761 -> 0.670 -> 0.317 -> 0.217 for
AOC Closure across d = 0.1, 0.3, 0.6, 0.9, so a pooled median describes no actual regime.
"""
import sys
import numpy as np, pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
DENSITY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
TAG = 'partition_quality_k20'
LEARNERS = ['AOC Closure', 'OC Closure', 'Greedy All-In']
PREF = 'GT partition'
PTEX = dict(zip(LEARNERS, LEARNERS), **{PREF: r'\textit{ground truth}'})
PC = ['allin_score', 'oc_score', 'info_kept']

p = pd.read_csv(f'{B}/data/{TAG}.csv')
p = p[(p.method.isin(LEARNERS + [PREF])) & (p.density == DENSITY)].copy()
p['allin_score'] = 1.0 - p.allin_pairs / p.max_pairs.clip(lower=1)
p['oc_score'] = 1.0 - p.oc_excess / (p.n_qedges - 1).clip(lower=1)
p.loc[p.max_pairs <= 0, 'allin_score'] = 1.0
p.loc[p.n_qedges <= 1, 'oc_score'] = 1.0
p['info_kept'] = 1.0 - p.intra_edges
rows = [a for a in LEARNERS if a in set(p.method)] + [PREF]
med = p.groupby('method')[PC + ['k_used', 'budget_k']].median().reindex(rows)
n = int(p.groupby('method').size().min())
best = {}
for c in PC:
    v = med.loc[[a for a in rows if a != PREF], c]
    best[c] = set(v[v >= v.max() - 5e-4].index)


def num(v, bold):
    return rf'$\mathbf{{{v:.3f}}}$' if bold else f'${v:.3f}$'


L = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
     r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
     r'Method & All-In & 1-child & info \\', r'\midrule']
for a in rows:
    if a == PREF:
        L.append(r'\midrule')
    L.append(f'{PTEX[a]} & ' + ' & '.join(num(med.loc[a, c], a in best[c]) for c in PC) + r' \\')
L += [r'\bottomrule', r'\end{tabular}',
      r'\caption{Median values of different partition-learning algorithms over $n=%d$ '
      r'instances of low-level graphs with ground truth abstractions to a $k^{*}=10$ variable '
      r'model. Density is fixed at $%g$, $p=100$, blockiness $0.8$ and budget $k\leq%d$. '
      r'All-In quantifies the proportion of node pairs that satisfy the All-In condition. '
      r'Similarly $1$-child gives the proportion of clusters with at most one child and '
      r'\textit{info}, the proportion of low-level edges that are not within clusters. All '
      r'three columns are a fraction of the ideal, $1.000$ best; \textbf{bold} is the best '
      r'learner. Note that the One-Child condition is not a property of the generating '
      r'partition, which scores only $%.3f$ on it. Both OC Closure and Greedy All-In break '
      r'ties by minimising intra-cluster edge mass.}'
      % (n, DENSITY, med.loc[rows[0], 'budget_k'], med.loc[PREF, 'oc_score']),
      r'\label{tab:partition}', r'\end{table}']

out = '\n'.join(L) + '\n'
with open(f'{B}/figures/table_partition.tex', 'w') as fh:
    fh.write(out)
print(out)
print(f'wrote figures/table_partition.tex   (density={DENSITY}, n={n})')
