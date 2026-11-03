"""The dial in interpretable units: loss relative to AOC Closure, and absolute RAE.

Ranks answered the statistical question but are hard to read. Here each run's information
loss is divided by AOC CLOSURE'S loss ON THE SAME INSTANCE, so 1.0 means "as lossy as
closure" and 0.5 means "half closure's loss" -- comparable across graphs of very different
absolute scale, without asking the reader to think in ranks.

RAE is shown on its own panel rather than a second y-axis (never a dual axis), since it has
a natural scale of its own and 0 is meaningful.

Instances where closure's own loss is ~0 are excluded from the left panel: the ratio is
undefined there. Median with interquartile band, so the run-to-run spread is visible rather
than hidden behind a mean.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
BLUE, ORANGE = '#2a78d6', '#eb6834'
FLOOR = 1e-4
mpl.rcParams.update({'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01})

d = pd.read_csv(f'{B}/data/frontier2.csv')
occ = d[(d.method == 'OCC') & d.iem.notna()].set_index(['density', 'seed']).iem
h = d[(d.method == 'HailMary') & d.iem.notna() & d.rae.notna()].copy()
h['occ_iem'] = [occ.get((r.density, r.seed), np.nan) for r in h.itertuples()]
usable = h[h.occ_iem > 0.01].copy()
usable['rel'] = usable.iem / usable.occ_iem
n_excl = h[['density', 'seed']].drop_duplicates().shape[0] - \
         usable[['density', 'seed']].drop_duplicates().shape[0]
print('instances used %d, excluded %d (closure loss ~ 0, ratio undefined)'
      % (usable[['density','seed']].drop_duplicates().shape[0], n_excl))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.2))
for ax in (a1, a2):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    ax.set_xscale('symlog', linthresh=0.3)
    ax.set_xlim(-0.05, 1600)
    ax.set_xticks([0, 1, 10, 100, 1000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: '%g' % v))
    ax.set_xlabel(r'$\lambda_{\mathrm{oc}}$  (one-child weight)', fontsize=8.5,
                  color=INK, labelpad=1.5)
    ax.tick_params(colors=INK2, labelsize=8)

g = usable.groupby('lam_oc').rel
med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
a1.fill_between(med.index.values, q1.values, q3.values, color=BLUE, alpha=0.16, lw=0, zorder=2)
a1.plot(med.index.values, med.values, '-o', color=BLUE, lw=1.6, ms=4.2, mec='white',
        mew=0.7, zorder=4)
a1.axhline(1.0, color=ORANGE, lw=1.4, ls=(0, (4, 2)), zorder=3)
a1.annotate('AOC Closure', (1500, 1.03), fontsize=7.5, color=ORANGE, ha='right', va='bottom')
a1.set_ylabel('information loss\n(relative to AOC Closure)', fontsize=8.5, color=INK, labelpad=2)
a1.set_ylim(0, 1.45)

gr = usable.assign(rae_p=usable.rae.clip(lower=FLOOR)).groupby('lam_oc').rae_p
m2, l2, u2 = gr.median(), gr.quantile(.25), gr.quantile(.75)
a2.fill_between(m2.index.values, l2.values, u2.values, color=ORANGE, alpha=0.16, lw=0, zorder=2)
a2.plot(m2.index.values, m2.values, '-s', color=ORANGE, lw=1.6, ms=4.2, mec='white',
        mew=0.7, zorder=4)
a2.set_yscale('log'); a2.set_ylim(FLOOR * 0.6, 2.0)
a2.set_ylabel(r'abstraction error $\mathrm{RAE}$', fontsize=8.5, color=INK, labelpad=2)
a2.annotate('numerically exact', (0.0, FLOOR * 0.78), fontsize=7.5, color=INK2,
            ha='left', va='bottom')

fig.subplots_adjust(left=0.135, right=0.985, top=0.96, bottom=0.215, wspace=0.42)
fig.savefig(f'{B}/figures/paper_dial_rel.pdf')
fig.savefig(f'{B}/figures/paper_dial_rel.png', dpi=300)
print('wrote figures/paper_dial_rel.pdf (+png)')
print()
print('median loss relative to closure, and median RAE:')
print(pd.DataFrame({'rel_loss': med.round(3), 'rae': m2.round(4)}).to_string())
