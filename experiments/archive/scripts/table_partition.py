"""Partition learners on the three metric axes, as a table.

Emits three things from one source of truth:
  figures/partition_table.{png,pdf}   rendered table for slides / quick reading
  figures/partition_table.tex         booktabs LaTeX for the paper
  stdout                              plain text

Every column is on a common [0, 1] "fraction of the ideal" scale, higher is better:

    All-In     1 - allin_pairs / max_pairs      max_pairs = sum over quotient edges x->y of
                                                |X_x|, i.e. every chance to violate
    one-child  1 - oc_excess  / (n_qedges - 1)  n_qedges-1 = the most excess a quotient with
                                                that many edges can have
    info kept  1 - intra_edges                  intra_edges = share of A's nonzero EDGES
                                                swallowed inside a cluster (count, not mass)

TWO THINGS THE BOLDING DOES NOT MEAN.

1. The ground truth is a REFERENCE ROW, not a competitor, and is excluded from the bolding.
   It is the partition the data was generated from, so "beating" it is not a goal.

2. Bolding the highest `one-child` is the most questionable of the three, because the ground
   truth scores only 0.333 on it. One-child is not a property of the generating partition, so
   a learner that maximises it is moving AWAY from the truth, not toward it. The column is
   reported because the algorithms target it; the bold marks the highest value, which is not
   the same as the best partition.

`k used` is reported because it is load-bearing: singletons satisfy All-In vacuously AND
swallow no edges, so a method spending more clusters gets part of two columns for free. Every
method ran at the same budget (AOC Closure's own cluster count); the ground truth reaches its
row at k = 10 while the learners spend 18-24.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'partition_quality_n20'
INK, INK2, RULE, HI = '#111111', '#555555', '#333333', '#dcebf7'
LEARNERS = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP l=0', 'GradCAP l=100']
REF = 'GT partition'
COLS = [('allin_score', 'All-In', 'All-In'),
        ('oc_score', 'one-child', 'one-child'),
        ('info_kept', 'info kept', 'info kept')]
TEX = {'AOC Closure': 'AOC Closure', 'OC Closure': 'OC Closure',
       'Greedy All-In': 'Greedy All-In', 'GradCAP l=0': r'GradCAP $\lambda_{oc}=0$',
       'GradCAP l=100': r'GradCAP $\lambda_{oc}=100$', 'GT partition': 'ground truth'}

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d['arm'] = np.where(d.method == 'GradCAP',
                    'GradCAP l=' + d.lam_oc.fillna(0).astype(int).astype(str), d.method)
d = d[d.arm.isin(LEARNERS + [REF])].copy()
d['allin_score'] = 1.0 - d.allin_pairs / d.max_pairs.clip(lower=1)
d['oc_score'] = 1.0 - d.oc_excess / (d.n_qedges - 1).clip(lower=1)
d.loc[d.max_pairs <= 0, 'allin_score'] = 1.0
d.loc[d.n_qedges <= 1, 'oc_score'] = 1.0
d['info_kept'] = 1.0 - d.intra_edges

rows = [a for a in LEARNERS if a in set(d.arm)] + ([REF] if REF in set(d.arm) else [])
med = d.groupby('arm')[[c for c, _, _ in COLS] + ['k_used']].median().reindex(rows)
q1 = d.groupby('arm')[[c for c, _, _ in COLS]].quantile(.25).reindex(rows)
q3 = d.groupby('arm')[[c for c, _, _ in COLS]].quantile(.75).reindex(rows)
n_inst = int(d.groupby('arm').size().min())

# best among LEARNERS only; ties (within 5e-4) all count
best = {}
for c, _, _ in COLS:
    v = med.loc[[a for a in rows if a != REF], c]
    best[c] = set(v[v >= v.max() - 5e-4].index)

# ------------------------------------------------------------------ plain text
print(f'\npartition metrics, median over n={n_inst} instances '
      f'(p=100, pooled over density and blockiness)')
print('all columns: fraction of the ideal, 1.0 = best.  * = best among learners\n')
hdr = f'{"":24s}' + ''.join(f'{lab:>16s}' for _, lab, _ in COLS) + f'{"k used":>9s}'
print(hdr); print('-' * len(hdr))
for a in rows:
    if a == REF:
        print('-' * len(hdr))
    line = f'{TEX[a].replace("$", "").replace(chr(92) + "lambda_{oc}", "lam"):24s}'
    for c, _, _ in COLS:
        star = '*' if (a in best[c]) else ' '
        line += f'{med.loc[a, c]:>15.3f}{star}'
    line += f'{med.loc[a, "k_used"]:>9.0f}'
    print(line)

# ------------------------------------------------------------------ LaTeX
tex = [r'\begin{tabular}{lcccc}', r'\toprule',
       r'& All-In $\uparrow$ & one-child $\uparrow$ & info kept $\uparrow$ & $k$ used \\',
       r'\midrule']
for a in rows:
    if a == REF:
        tex.append(r'\midrule')
    cells = []
    for c, _, _ in COLS:
        v = f'{med.loc[a, c]:.3f}'
        cells.append(rf'\textbf{{{v}}}' if a in best[c] else v)
    tex.append(f'{TEX[a]} & ' + ' & '.join(cells) + f' & {med.loc[a, "k_used"]:.0f}' + r' \\')
tex += [r'\bottomrule', r'\end{tabular}']
with open(f'{B}/figures/partition_table.tex', 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f'\nwrote figures/partition_table.tex')

# ------------------------------------------------------------------ rendered table
nr, nc = len(rows) + 1, len(COLS) + 2
fig, ax = plt.subplots(figsize=(6.5, 0.42 * nr + 0.55))
ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, nr + 0.6)
XC = [0.045] + list(np.linspace(0.42, 0.955, len(COLS) + 1))


def yof(i):
    return nr - 0.55 - i


for j, (_, lab, _) in enumerate(COLS):
    ax.text(XC[j + 1], yof(0), lab + r' $\uparrow$', fontsize=9.5, color=INK,
            ha='center', va='center')
ax.text(XC[-1], yof(0), r'$k$ used', fontsize=9.5, color=INK, ha='center', va='center')

for i, a in enumerate(rows, start=1):
    y = yof(i)
    is_ref = (a == REF)
    ax.text(XC[0], y, TEX[a], fontsize=9.5, ha='left', va='center',
            color=INK2 if is_ref else INK, style='italic' if is_ref else 'normal')
    for j, (c, _, _) in enumerate(COLS):
        b = a in best[c]
        if b:
            ax.add_patch(plt.Rectangle((XC[j + 1] - 0.078, y - 0.42), 0.156, 0.84,
                                       facecolor=HI, edgecolor='none', zorder=0))
        ax.text(XC[j + 1], y + 0.07, f'{med.loc[a, c]:.3f}', fontsize=9.5, ha='center',
                va='center', color=INK2 if is_ref else INK,
                fontweight='bold' if b else 'normal')
        ax.text(XC[j + 1], y - 0.27, f'[{q1.loc[a, c]:.2f}, {q3.loc[a, c]:.2f}]',
                fontsize=6.5, ha='center', va='center', color='#8a8a8a')
    ax.text(XC[-1], y, f'{med.loc[a, "k_used"]:.0f}', fontsize=9.5, ha='center',
            va='center', color=INK2 if is_ref else INK)

for yy, lw in ((yof(0) + 0.55, 1.1), (yof(0) - 0.5, 0.7),
               (yof(len(rows) - 1) + 0.5, 0.7), (yof(len(rows)) - 0.5, 1.1)):
    ax.plot([0.02, 0.985], [yy, yy], color=RULE, lw=lw, zorder=2)

ax.text(0.02, yof(len(rows)) - 1.05,
        'median over $n=%d$ instances ($p=100$, pooled over density and blockiness); '
        '[q1, q3] below.\nAll columns are a fraction of the ideal, $1.0$ = best. '
        'Shaded = best among learners; ground truth is a reference, not ranked.' % n_inst,
        fontsize=7, color=INK2, ha='left', va='top')
fig.savefig(f'{B}/figures/partition_table.pdf')
fig.savefig(f'{B}/figures/partition_table.png', dpi=300)
plt.close(fig)
print('wrote figures/partition_table.png (+pdf)')


# ================================================================== per-metric x density
# One table per metric axis, densities across the columns. Bold = best among LEARNERS in
# that column; the ground-truth row is a reference and is never bolded.
dens = sorted(d.density.unique())
kbud = d.groupby('density').budget_k.median()
nper = d.groupby(['density', 'arm']).size().groupby('density').min()

CAPTION = {
    'allin_score': ('All-In satisfaction', 'tab:allin',
                    r'$1-\text{violations}/\text{chances}$; $1.0$ means All-In holds exactly'),
    'oc_score': ('One-child satisfaction', 'tab:onechild',
                 r'$1-\text{excess}/\text{max excess}$; $1.0$ means one-child holds exactly'),
    'info_kept': ('Information kept', 'tab:infokept',
                  r'$1-$ the share of $A$\textquotesingle s nonzero edges falling inside a cluster'),
}

blocks = []
for c, lab, _ in COLS:
    piv = d.pivot_table(index='arm', columns='density', values=c, aggfunc='median').reindex(rows)
    bestcol = {}
    for dn in dens:
        v = piv.loc[[a for a in rows if a != REF], dn]
        bestcol[dn] = set(v[v >= v.max() - 5e-4].index)
    title, lblref, expl = CAPTION[c]
    t = [r'\begin{table}[t]', r'\centering',
         rf'\caption{{{title} by density. {expl}. Higher is better; '
         rf'\textbf{{bold}} marks the best learner in each column. '
         rf'The ground truth is a reference, not a competitor. '
         rf'Budget $k = {", ".join(f"{kbud[x]:.0f}" for x in dens)}$ and '
         rf'$n = {", ".join(f"{nper[x]:.0f}" for x in dens)}$ instances '
         rf'at the four densities respectively; $p = 100$.}}',
         rf'\label{{{lblref}}}',
         r'\begin{tabular}{l' + 'c' * len(dens) + '}', r'\toprule',
         r'& \multicolumn{%d}{c}{density} \\' % len(dens),
         r'\cmidrule(lr){2-%d}' % (len(dens) + 1),
         '& ' + ' & '.join(f'${x:g}$' for x in dens) + r' \\', r'\midrule']
    for a in rows:
        if a == REF:
            t.append(r'\midrule')
        cells = []
        for dn in dens:
            v = f'{piv.loc[a, dn]:.3f}'
            cells.append(rf'\textbf{{{v}}}' if a in bestcol[dn] else v)
        nm = TEX[a] if a != REF else r'\textit{ground truth}'
        t.append(f'{nm} & ' + ' & '.join(cells) + r' \\')
    t += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    blocks.append('\n'.join(t))

out = '\n\n'.join(blocks) + '\n'
with open(f'{B}/figures/partition_tables_by_density.tex', 'w') as f:
    f.write(out)
print('\nwrote figures/partition_tables_by_density.tex\n')
print(out)
