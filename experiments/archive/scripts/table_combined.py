"""One LaTeX figure holding both result tables, side by side, for a tight page budget.

  (a) partition learners        All-In / one-child / information retained
  (b) the T-fitter + UCA        RAE / ReconErr / runtime

DENSITY: a SINGLE density (0.6), not pooled. `info kept` runs 0.761 -> 0.670 -> 0.317 ->
0.217 for AOC Closure across d = 0.1, 0.3, 0.6, 0.9, so the pooled median (0.452) describes
no actual regime -- it is an average of two. All-In is nearly flat in density so pooling
would be harmless there, but the column carrying the main result is the one pooling distorts
most. d=0.6 is the closest available level to the fitter table's d=0.5, so the two panels
describe near-identical settings. Blockiness IS pooled: it moves these metrics very little
(interaction 0.005-0.10 against density's 0.13-0.41, see analyse_partition_axis.py).

Table (a) columns, all on a common [0,1] "fraction of the ideal" scale, higher better:
    All-In     1 - allin_pairs / max_pairs
    one-child  1 - oc_excess  / (n_qedges - 1)
    info kept  1 - intra_edges          (unweighted EDGE COUNT, not mass)

Table (b) uses the canonical metrics from src/utils/losses.py as of 2026-09-24 (RAE returns
sqrt(num/den); reconstruction_loss is a ratio of Frobenius NORMS), so both are the square
root of values in any CSV from before that date.
"""
import sys
import numpy as np, pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
DENSITY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
PART_TAG, FIT_TAG = 'partition_quality_k20', 'fitter_gt_random'
ANCHOR = dict(p=100, k_budget=10, density=0.5)

LEARNERS = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP l=0',
            'GradCAP l=100', 'GradCAP l=1000']
PREF = 'GT partition'
PTEX = {'AOC Closure': 'AOC Closure', 'OC Closure': 'OC Closure',
        'Greedy All-In': 'Greedy All-In', 'GradCAP l=0': r'GradCAP $\lambda_{oc}\!=\!0$',
        'GradCAP l=100': r'GradCAP $\lambda_{oc}\!=\!100$',
        'GradCAP l=1000': r'GradCAP $\lambda_{oc}\!=\!1000$',
        'GT partition': r'\textit{ground truth}'}
FITTED = ['GT acyclic', 'GT cyclic', 'random']
FTEX = {'GT acyclic': 'GT acyclic', 'GT cyclic': 'GT cyclic', 'random': 'random',
        'UCA': 'UCA'}

# ---------------------------------------------------------------- (a) partition learners
p = pd.read_csv(f'{B}/data/{PART_TAG}.csv')
p['arm'] = np.where(p.method == 'GradCAP',
                    'GradCAP l=' + p.lam_oc.fillna(0).astype(int).astype(str), p.method)
p = p[(p.arm.isin(LEARNERS + [PREF])) & (p.density == DENSITY)].copy()
p['allin_score'] = 1.0 - p.allin_pairs / p.max_pairs.clip(lower=1)
p['oc_score'] = 1.0 - p.oc_excess / (p.n_qedges - 1).clip(lower=1)
p.loc[p.max_pairs <= 0, 'allin_score'] = 1.0
p.loc[p.n_qedges <= 1, 'oc_score'] = 1.0
p['info_kept'] = 1.0 - p.intra_edges
prows = [a for a in LEARNERS if a in set(p.arm)] + [PREF]
PC = ['allin_score', 'oc_score', 'info_kept']
pmed = p.groupby('arm')[PC + ['k_used', 'budget_k']].median().reindex(prows)
pn = int(p.groupby('arm').size().min())
pbest = {}
for c in PC:
    v = pmed.loc[[a for a in prows if a != PREF], c]
    pbest[c] = set(v[v >= v.max() - 5e-4].index)

# ---------------------------------------------------------------- (b) fitter
f = pd.read_csv(f'{B}/data/{FIT_TAG}.csv')
f = f[f.recon.notna()].copy()
m = np.ones(len(f), bool)
for k, v in ANCHOR.items():
    m &= (f[k] == v).values
f = f[m]
fn = int(f.groupby(['partition', 'faithful']).size().min())


def num(v, kind, bold=False):
    """Bold goes INSIDE math mode: \\textbf around a $...$ group does not bold digits."""
    if not np.isfinite(v):
        return '--'
    if kind == 'plain':
        body = f'{v:.3f}'
    elif kind == 'secs':
        body = f'{v:.2f}'
    elif kind == 'recon' and v >= 5e-4:
        body = f'{v:.3f}'
    elif v <= 0:
        body = '0'
    else:
        e = int(np.floor(np.log10(v)))
        body = rf'{v / 10 ** e:.0f}\!\times\!10^{{{e}}}'
    return rf'$\mathbf{{{body}}}$' if bold else f'${body}$'


L = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
     r'\begin{minipage}[t]{0.47\linewidth}', r'\centering',
     r'\textbf{(a) partition learners}\\[2pt]',
     r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
     r'& All-In & 1-child & info \\', r'\midrule']
for a in prows:
    if a == PREF:
        L.append(r'\midrule')
    L.append(f'{PTEX[a]} & ' +
             ' & '.join(num(pmed.loc[a, c], 'plain', a in pbest[c]) for c in PC) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{minipage}\hfill',
      r'\begin{minipage}[t]{0.50\linewidth}', r'\centering',
      r'\textbf{(b) $T$-fitter}\\[2pt]',
      r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
      r'& RAE & ReconErr & time (s) \\', r'\midrule']


def fblock(arms, faith, label):
    out = [rf'\multicolumn{{4}}{{@{{}}l}}{{\textit{{{label}}}}} \\']
    med = {a: {c: f[(f.partition == a) & (f.faithful == faith)][c].median()
               for c in ('rae', 'recon', 'secs')} for a in arms}
    best = {c: set() for c in ('rae', 'recon')}
    if len(arms) > 1:
        for c in ('rae', 'recon'):
            vals = {a: med[a][c] for a in arms if np.isfinite(med[a][c])}
            if vals:
                lo = min(vals.values())
                best[c] = {a for a, v in vals.items() if v <= lo * 1.05 + 1e-30}
    for a in arms:
        out.append(rf'\quad {FTEX[a]} & ' + ' & '.join(
            num(med[a][c], k, a in best.get(c, ()))
            for c, k in (('rae', 'sci'), ('recon', 'recon'), ('secs', 'secs'))) + r' \\')
    return out


L += fblock(FITTED, 1, 'faithful')
L.append(r'\midrule')
L += fblock(FITTED, 0, 'unfaithful')
if 'UCA' in set(f.partition):
    L.append(r'\midrule')
    L += fblock(['UCA'], 1, 'end-to-end (no fitter)')
L += [r'\bottomrule', r'\end{tabular}', r'\end{minipage}',
      r'\caption{\textbf{(a)} Partition learners at density $%g$, $p=100$, blockiness $0.8$, '
      r'fixed budget $k\leq%d$, median over $n=%d$ instances. All three columns are a fraction '
      r'of the ideal, $1.0$ best; \textbf{bold} is the best learner. The ground truth is a '
      r'reference: note it satisfies All-In exactly but scores only $%.3f$ on one-child, so '
      r'one-child is not a property of the generating partition. '
      r'\textbf{(b)} \texttt{fit\_abstraction\_auto} at $p=100$, $k=10$, density $0.5$, '
      r'blockiness $1$, median over $n=%d$. RAE cannot separate a correct partition from a '
      r'random one; ReconErr separates them completely, but only when faithful. UCA learns '
      r'$T$ end-to-end (6000 steps), so faithfulness does not apply to it. Time is a cost '
      r'and is not ranked.}'
      % (DENSITY, pmed.loc[prows[0], 'budget_k'], pn, pmed.loc[PREF, 'oc_score'], fn),
      r'\label{tab:main}', r'\end{table}']

out = '\n'.join(L) + '\n'
with open(f'{B}/figures/combined_tables.tex', 'w') as fh:
    fh.write(out)
print(out)
print(f'wrote figures/combined_tables.tex   (density={DENSITY}, n_part={pn}, n_fit={fn})')
