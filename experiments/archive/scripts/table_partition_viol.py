"""Panel (a), lower-is-better: partition learners scored by VIOLATION RATE.

Same data, same normalisers and the same single density as `table_partition_only.py`
(`partition_quality_k20.csv`); only the orientation is flipped, so every entry here is the
complement of the corresponding entry there. Medians commute with `x -> 1-x`, so the two
tables agree entry-for-entry -- no rerun, and no possibility of drift between them.

Columns, all on a common [0,1] scale, LOWER better, 0 = ideal:
    All-In     allin_pairs / max_pairs   violating (node, quotient-edge) incidences
    1-child    oc_excess  / (n_qedges-1) excess child-edges beyond one per cluster
    info       intra_edges               low-level EDGES swallowed inside clusters

NOTE `info` counts EDGES while both learners tie-break on edge MASS; `intra` is the mass
column if that is wanted instead.

DENSITY: a SINGLE density, not pooled -- see the note in `table_partition_only.py`.
"""
import sys
import pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
DENSITY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
TAG = 'partition_quality_k20'
LEARNERS = ['AOC Closure', 'OC Closure', 'Greedy All-In']
PREF = 'GT partition'
PTEX = dict(zip(LEARNERS, LEARNERS), **{PREF: r'\textit{ground truth}'})
PC = ['allin_v', 'oc_v', 'info_v']

p = pd.read_csv(f'{B}/data/{TAG}.csv')
p = p[(p.method.isin(LEARNERS + [PREF])) & (p.density == DENSITY)].copy()
p['allin_v'] = p.allin_pairs / p.max_pairs.clip(lower=1)
p['oc_v'] = p.oc_excess / (p.n_qedges - 1).clip(lower=1)
p.loc[p.max_pairs <= 0, 'allin_v'] = 0.0
p.loc[p.n_qedges <= 1, 'oc_v'] = 0.0
p['info_v'] = p.intra_edges
rows = [a for a in LEARNERS if a in set(p.method)] + [PREF]
med = p.groupby('method')[PC + ['k_used', 'budget_k']].median().reindex(rows)
n = int(p.groupby('method').size().min())
best = {}
for c in PC:                                   # LOWER is better, so bold the minimum
    v = med.loc[[a for a in rows if a != PREF], c]
    best[c] = set(v[v <= v.min() + 5e-4].index)


def num(v, bold):
    return rf'$\mathbf{{{v:.3f}}}$' if bold else f'${v:.3f}$'


L = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
     r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
     r'Method & All-In $\downarrow$ & 1-child $\downarrow$ & info $\downarrow$ \\', r'\midrule']
for a in rows:
    if a == PREF:
        L.append(r'\midrule')
    L.append(f'{PTEX[a]} & ' + ' & '.join(num(med.loc[a, c], a in best[c]) for c in PC) + r' \\')
L += [r'\bottomrule', r'\end{tabular}',
      r'\caption{Violation rates of different partition-learning algorithms, median over '
      r'$n=%d$ instances of low-level graphs with ground-truth abstractions to a $k^{*}=10$ '
      r'variable model. Each instance is a single fixed setting, not a pool: $p=100$ '
      r'low-level variables, density $%g$, blockiness $0.8$, and a shared budget ceiling of '
      r'$k\leq%d$ clusters for every learner. \emph{All-In} is the fraction of '
      r'(node, quotient-edge) incidences that violate the All-In condition; \emph{1-child} '
      r'is the number of child-edges in excess of one per cluster, normalised by the '
      r'quotient edge count; \emph{info} is the fraction of low-level edges lost inside '
      r'clusters. All three columns are violation rates on a common $[0,1]$ scale, '
      r'$0.000$ ideal, lower better; \textbf{bold} is the best learner. Note that the '
      r'One-Child condition is not a property of the generating partition, which itself '
      r'violates it at $%.3f$. Both OC Closure and Greedy All-In break ties by minimising '
      r'intra-cluster edge mass.}'
      % (n, DENSITY, med.loc[rows[0], 'budget_k'], med.loc[PREF, 'oc_v']),
      r'\label{tab:partition}', r'\end{table}']

out = '\n'.join(L) + '\n'
with open(f'{B}/figures/partition_table_viol.tex', 'w') as fh:
    fh.write(out)
print(out)
print(f'wrote figures/partition_table_viol.tex   (density={DENSITY}, n={n})')
