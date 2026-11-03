"""LaTeX table for the fit_abstraction_auto / partition-type experiment.

Replaces figures/fitter_gt_random_bars: same data, same anchor cell, as a table.

Rows are grouped by faithfulness, with UCA in its own block because it is not fitted at all
-- it learns T end-to-end, so the `faithful` flag (a property of the fitter) does not apply
and repeating it under both settings would imply a comparison that was never run.

Metrics are the canonical ones from src/utils/losses.py as of 2026-09-24: RAE returns
sqrt(num/den) and reconstruction_loss is a ratio of Frobenius NORMS. Both are therefore the
square root of the values in any CSV from before that date.

Bold marks the LOWEST value in each (block, column) for RAE and ReconErr only. Time is a
cost, not a quality, so it is left unbolded -- the fastest fit here is the one that does the
least work, which is not something to reward.
"""
import sys
import numpy as np, pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'fitter_gt_random'
ANCHOR = dict(p=100, k_budget=10, density=0.5)
FITTED = ['GT acyclic', 'GT cyclic', 'random']
TEX = {'GT acyclic': 'ground truth (acyclic)', 'GT cyclic': 'ground truth (cyclic)',
       'random': 'random partition', 'UCA': 'UCA'}

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d = d[d.recon.notna()].copy()
m = np.ones(len(d), bool)
for k, v in ANCHOR.items():
    m &= (d[k] == v).values
s = d[m]
n_inst = int(s.groupby(['partition', 'faithful']).size().min())


def fmt(v, kind, bold=False):
    """Render one cell. Bold is applied INSIDE math mode via \\mathbf -- wrapping a
    $...$ group in \\textbf does nothing to the digits."""
    if not np.isfinite(v):
        return '--'
    if kind == 'secs':
        body = rf'{v:.2f}'
    elif kind == 'recon' and v >= 5e-4:
        body = rf'{v:.3f}'
    elif v <= 0:
        body = '0'
    else:
        e = int(np.floor(np.log10(v)))
        body = rf'{v / 10 ** e:.1f}\!\times\!10^{{{e}}}'
    return rf'$\mathbf{{{body}}}$' if bold else rf'${body}$'


def block(rows_arms, faith, label):
    out = [rf'\multicolumn{{4}}{{l}}{{\textit{{{label}}}}} \\']
    sub = {a: s[(s.partition == a) & (s.faithful == faith)] for a in rows_arms}
    meds = {a: {c: sub[a][c].median() for c in ('rae', 'recon', 'secs')} for a in rows_arms}
    # a block with one row has no comparison to make, so nothing is bolded in it
    best = {c: set() for c in ('rae', 'recon')}
    if len(rows_arms) > 1:
        for c in ('rae', 'recon'):
            vals = {a: meds[a][c] for a in rows_arms if np.isfinite(meds[a][c])}
            if vals:
                lo = min(vals.values())
                best[c] = {a for a, v in vals.items() if v <= lo * 1.05 + 1e-30}
    for a in rows_arms:
        cells = [fmt(meds[a][c], c, bold=(a in best.get(c, ())))
                 for c in ('rae', 'recon', 'secs')]
        out.append(rf'\quad {TEX[a]} & ' + ' & '.join(cells) + r' \\')
    return out


tex = [r'\begin{table}[t]', r'\centering',
       r'\caption{\texttt{fit\_abstraction\_auto} on ground-truth and random partitions, '
       r'at both faithfulness settings. RAE cannot separate a correct partition from a '
       r'random one -- both reach $\sim\!10^{-5}$ -- while ReconErr separates them '
       r'completely, but only when \texttt{faithful=True}. UCA is not fitted: it learns '
       r'$T$ end-to-end over 6000 gradient steps, so the faithfulness flag does not apply '
       r'to it. Medians over $n=%d$ instances at $p=100$, $k=10$, density $0.5$, '
       r'blockiness $1$. Bold marks the lowest RAE / ReconErr within each block; time is a '
       r'cost and is not ranked.}' % n_inst,
       r'\label{tab:fitter-gt-random}',
       r'\begin{tabular}{lccc}', r'\toprule',
       r'& RAE $\downarrow$ & ReconErr $\downarrow$ & time (s) \\', r'\midrule']
tex += block(FITTED, 1, r'faithful $=$ True')
tex.append(r'\midrule')
tex += block(FITTED, 0, r'faithful $=$ False')
if 'UCA' in set(s.partition):
    tex.append(r'\midrule')
    tex += block(['UCA'], 1, r'no fitter (end-to-end)')
tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

out = '\n'.join(tex) + '\n'
with open(f'{B}/figures/fitter_gt_random_table.tex', 'w') as f:
    f.write(out)
print(out)
print('wrote figures/fitter_gt_random_table.tex\n')

pd.set_option('display.width', 220)
print('=== underlying medians (anchor cell) ===')
print(s.groupby(['faithful', 'partition'])[['rae', 'recon', 'secs', 'marg', 'norm_Ahigh']]
      .median().to_string(float_format=lambda v: f'{v:.4g}'))
