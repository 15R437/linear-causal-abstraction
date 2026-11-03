"""Analysis + figures for the large benchmark tiers."""
import os, sys, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
COL = {'random': '#cccccc', 'uca': '#999999', 'hc_fast': '#666666',
       'closure': '#1f77b4', 'closure+HM': '#2ca02c'}
ORDER = ['random', 'uca', 'hc_fast', 'closure', 'closure+HM']


def load(tier):
    f = f'{B}/data/large_{tier}.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if 'obs_loss' in d:
        d['exact'] = ((d.obs_loss < 1e-8) & (d.intv_loss < 1e-8)).astype(float)
    d['fam'] = np.where(d.graph_type == 'one_child', 'one_child',
                        'random d=' + d.density.astype(str))
    return d


def table(d, by, metrics=('success', 'exact', 'info', 'recon_data', 'secs')):
    ms = [m for m in metrics if m in d.columns]
    agg = {m: (m, 'median' if m == 'recon_data' else 'mean') for m in ms}
    agg['n'] = ('method', 'size')
    return d.groupby(by).agg(**agg).round(3)


d = load('main')
if d is not None:
    order = [m for m in ORDER if m in set(d.method)]
    print('=' * 78); print('MAIN TIER  (%d rows, %d instances)' % (len(d), len(d) // len(order)))
    print('=' * 78)
    print('\n--- overall by family ---')
    print(table(d, ['graph_type', 'method']).reindex(order, level=1).to_string())
    print('\n--- random DAGs, by density ---')
    r = d[d.graph_type == 'random']
    print(table(r, ['density', 'method']).reindex(order, level=1).to_string())
    print('\n--- by p ---')
    print(table(d, ['p', 'method']).reindex(order, level=1).to_string())
    print('\n--- by ground-truth n_high (random only) ---')
    print(table(r, ['true_nhigh', 'method']).reindex(order, level=1).to_string())
    print('\n--- how often was the fallback needed? ---')
    c = d[d.method == 'closure']
    print(c.groupby(['graph_type', 'density']).agg(
        n=('used_fallback', 'size'), closure_raw=('closure_raw_ok', 'mean'),
        fallback_used=('used_fallback', 'mean'), final_success=('success', 'mean'),
        k=('k', 'mean')).round(3).to_string())

    # FIGURE: exact recovery vs density, per method
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for a, (metric, ttl) in zip(ax, [('exact', 'exact recovery (obs & intv < 1e-8)'),
                                     ('success', 'All-In AND one-child'),
                                     ('info', 'information loss (valid partitions only)')]):
        for m in order:
            g = r[r.method == m].groupby('density')[metric].mean()
            a.plot(g.index, g.values, 'o-', color=COL[m], label=m, lw=1.7, ms=5)
        a.set_xlabel('density'); a.set_title(ttl, fontsize=9); a.set_ylim(-0.04, 1.04)
    ax[0].set_ylabel('random DAGs'); ax[0].legend(fontsize=7)
    fig.suptitle('Random DAGs: every method across density', y=1.04)
    fig.savefig(f'{B}/figures/11_large_density.png'); plt.close(fig)
    print('\nwrote figures/11_large_density.png')

    # FIGURE: exact recovery heat by (density, n_high) for closure
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for a, m in zip(ax, ['closure', 'hc_fast']):
        piv = r[r.method == m].pivot_table(index='true_nhigh', columns='density',
                                           values='exact', aggfunc='mean')
        im = a.imshow(piv.values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        a.set_xticks(range(len(piv.columns))); a.set_xticklabels(piv.columns)
        a.set_yticks(range(len(piv.index))); a.set_yticklabels(piv.index)
        a.set_xlabel('density'); a.set_title(m); a.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    a.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                           color='white' if v > 0.5 else 'black')
    ax[0].set_ylabel('ground-truth n_high')
    fig.colorbar(im, ax=ax, fraction=0.03, label='exact recovery')
    fig.suptitle('Exact recovery across the random-DAG grid', y=1.04)
    fig.savefig(f'{B}/figures/12_large_grid.png'); plt.close(fig)
    print('wrote figures/12_large_grid.png')

d = load('scale')
if d is not None:
    order = [m for m in ORDER if m in set(d.method)]
    print('\n' + '=' * 78); print('SCALE TIER (p=200,400)'); print('=' * 78)
    print(table(d, ['p', 'graph_type', 'method']).to_string())

d = load('shape')
if d is not None:
    print('\n' + '=' * 78); print('SHAPE TIER (blockiness x heterogeneity)'); print('=' * 78)
    print(table(d, ['graph_type', 'blockiness', 'heterogeneity'],
                metrics=('success', 'info', 'secs')).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for a, gt in zip(ax, ['random', 'one_child']):
        g = d[d.graph_type == gt]
        if not len(g):
            continue
        piv = g.pivot_table(index='blockiness', columns='heterogeneity', values='success')
        im = a.imshow(piv.values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        a.set_xticks(range(len(piv.columns))); a.set_xticklabels(piv.columns)
        a.set_yticks(range(len(piv.index))); a.set_yticklabels(piv.index)
        a.set_xlabel('heterogeneity'); a.set_title(gt); a.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                a.text(j, i, f'{piv.values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                       color='white' if piv.values[i, j] > 0.5 else 'black')
    ax[0].set_ylabel('blockiness')
    fig.suptitle('closure success across graph shape', y=1.04)
    fig.savefig(f'{B}/figures/13_shape.png'); plt.close(fig)
    print('wrote figures/13_shape.png')
