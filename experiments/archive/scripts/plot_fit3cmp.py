"""Figure for fit3cmp.py: RAE, T-faithfulness and runtime for the three fitters."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
df = pd.read_csv(f'{B}/data/fit3cmp.csv')
M = [('reg', 'fit_abstraction', '#d62728'),
     ('alt', '..._alternative', '#1f77b4'),
     ('K',   '..._K',           '#2ca02c')]
cells = [('acyclic', 'gt'), ('cyclic', 'gt'), ('acyclic', 'random'), ('cyclic', 'random')]
lbl = ['acyclic\nGT part.', 'cyclic\nGT part.', 'acyclic\nrandom', 'cyclic\nrandom']
x = np.arange(len(cells)); w = 0.26

fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))

for a, (m, name, c) in enumerate(M):
    v = [ (df[(df.kind==k)&(df.part==p)][f'{m}_rae_loc'] < 1e-6).mean() for k, p in cells ]
    ax[0].bar(x+(a-1)*w, v, w, label=name, color=c)
    v = [ df[(df.kind==k)&(df.part==p)][f'{m}_faithful_abs'].mean() for k, p in cells ]
    ax[1].bar(x+(a-1)*w, v, w, color=c)
    v = [ df[(df.kind==k)&(df.part==p)][f'{m}_time'].median() for k, p in cells ]
    ax[2].bar(x+(a-1)*w, v, w, color=c)

for a, t in zip(ax, ['exact abstraction\n(frac. RAE < 1e-6)',
                     'fully T-faithful\n(frac. no cross-cluster edge dropped)',
                     'median runtime (s)']):
    a.set_xticks(x); a.set_xticklabels(lbl, fontsize=9); a.set_title(t, fontsize=10)
    a.grid(axis='y', alpha=.3); a.set_axisbelow(True)
ax[0].set_ylim(0, 1.05); ax[1].set_ylim(0, 1.05)
ax[0].legend(fontsize=8.5, loc='lower left')
fig.suptitle('Three T-fitters on ground-truth vs random partitions  (n=30 per cell, p=24/40)', fontsize=11)
fig.tight_layout()
fig.savefig(f'{B}/figures/fit3cmp.png', dpi=150)
print('wrote', f'{B}/figures/fit3cmp.png')
