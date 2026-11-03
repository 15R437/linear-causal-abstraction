"""Figures for the study report. Writes PNGs into ../figures/."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
FIG, DAT = f'{D}/figures', f'{D}/data'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
C = {'closure': '#1f77b4', 'hm': '#d62728', 'hybrid': '#2ca02c', 'other': '#7f7f7f'}


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png')
    plt.close(fig)
    print('wrote', name)


# 1 -- the ceiling is governed by n_high, not p
d = pd.read_csv(f'{DAT}/kpolicy.csv').drop_duplicates(['graph_type', 'p', 'true_nhigh', 'seed'])
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for a, (gt, ttl) in zip(ax, [('one_child', 'one_child family'), ('random', 'random family')]):
    g = d[d.graph_type == gt].groupby(['true_nhigh', 'p']).ceiling.mean().reset_index()
    g['frac'] = g.ceiling / g.p
    for p, gg in g.groupby('p'):
        a.plot(gg.true_nhigh, gg.frac, 'o-', label=f'p={p}', lw=1.6, ms=5)
    a.set_xscale('log'); a.set_xticks([3, 5, 10, 20, 40])
    a.set_xticklabels([3, 5, 10, 20, 40])
    a.set_xlabel('ground-truth n_high'); a.set_ylim(0, 0.75)
    a.set_title(ttl); a.legend(fontsize=7)
ax[0].set_ylabel('max feasible k  /  p')
fig.suptitle('The ceiling on k is set by ground-truth granularity, not by p', y=1.04)
save(fig, '01_ceiling')

# 2 -- density governs difficulty, and the two criteria oppose
d = pd.read_csv(f'{DAT}/density.csv')
g = d.groupby(['p', 'density']).agg(succ=('returned', 'mean'), info=('info', 'mean'),
                                    sink=('frac_zero_out', 'mean')).reset_index()
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for p, gg in g.groupby('p'):
    ax[0].plot(gg.density, gg['succ'], 'o-', label=f'p={p}', lw=1.6, ms=5)
    ax[1].plot(gg.density, gg['info'], 'o-', label=f'p={p}', lw=1.6, ms=5)
    ax[2].plot(gg.density, gg['sink'], 'o-', label=f'p={p}', lw=1.6, ms=5)
ax[0].set_ylabel('closure success'); ax[0].set_title('validity improves with density')
ax[1].set_ylabel('information loss at the ceiling'); ax[1].set_title('but information loss worsens')
ax[2].set_ylabel('frac. zero-out-degree nodes'); ax[2].set_title('the mechanism: stranded nodes')
for a in ax:
    a.set_xlabel('density'); a.legend(fontsize=7)
fig.suptitle('Sparse graphs are the most worth abstracting and the hardest to abstract', y=1.04)
save(fig, '02_density')

# 3 -- refine_by_splitting contributes nothing
d = pd.read_csv(f'{DAT}/ceilbase.csv')
s = d.groupby(['graph_type', 'policy']).secs.mean().unstack()
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
x = np.arange(len(s))
ax[0].bar(x - 0.18, s['closure_only'], 0.36, label='closure only', color=C['closure'])
ax[0].bar(x + 0.18, s['closure+split'], 0.36, label='+ refine_by_splitting', color=C['hm'])
ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(s.index)
ax[0].set_ylabel('seconds (12 draws)'); ax[0].legend(fontsize=7)
ax[0].set_title('refinement is 99.5% of the runtime')
k = d.groupby(['graph_type', 'policy']).k.mean().unstack()
ax[1].bar(x - 0.18, k['closure_only'], 0.36, label='closure only', color=C['closure'])
ax[1].bar(x + 0.18, k['closure+split'], 0.36, label='+ refine_by_splitting', color=C['hm'])
ax[1].set_xticks(x); ax[1].set_xticklabels(k.index); ax[1].set_ylabel('clusters found (k)')
ax[1].set_title('and finds nothing: identical on 90/90'); ax[1].legend(fontsize=7)
save(fig, '03_refine')

# 4 -- information loss vs validity are antagonistic
d = pd.read_csv(f'{DAT}/violinit.csv')
d['ok'] = ((d.allin == 1) & (d.one_child == 1))
r = d[d.graph_type == 'random']
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for i, (lab, sub) in enumerate([('failed', r[~r.ok]), ('succeeded', r[r.ok])]):
    ax[0].scatter(sub['used'], sub['info'], s=28, alpha=.75,
                  color=[C['hm'], C['closure']][i], label=lab)
ax[0].set_xlabel('clusters used'); ax[0].set_ylabel('information loss')
ax[0].set_title('finer partitions lose less info -- and break All-In'); ax[0].legend(fontsize=7)
gw = pd.read_csv(f'{DAT}/graphweight.csv')
gw = gw[(gw.graph_type == 'random') & (gw.lambda_one_child == 0)].groupby('lambda_graph')
m = gw.agg(allin=('allin', 'mean'), info=('info', 'mean'), he=('high_edges', 'mean'))
ax[1].plot(m.index, m['info'], 'o-', color=C['hybrid'], lw=1.6, label='information loss')
ax[1].plot(m.index, m['allin'], 's-', color=C['hm'], lw=1.6, label='All-In satisfied')
ax[1].set_xscale('log'); ax[1].set_xlabel('lambda_graph  (one_child OFF)')
ax[1].set_ylim(-0.03, 1.0); ax[1].legend(fontsize=7)
ax[1].set_title('All-In never satisfied without one-child')
save(fig, '04_antagonism')

# 5 -- method comparison (paired)
cb = pd.read_csv(f'{DAT}/ceilbase.csv'); cb = cb[cb.policy == 'closure_only']
kp = pd.read_csv(f'{DAT}/kpolicy.csv'); kp = kp[kp.policy == 'ceiling']
key = ['graph_type', 'p', 'true_nhigh', 'seed']
j = cb.set_index(key)[['success', 'info']].join(
    kp.set_index(key)[['success', 'info']], lsuffix='_clo', rsuffix='_hm', how='inner')
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for a, gt in zip(ax, ['random', 'one_child']):
    g = j.xs(gt, level='graph_type')
    b = g[(g.success_clo == 1) & (g.success_hm == 1)]
    a.scatter(b.info_clo, b.info_hm, s=30, alpha=.8, color=C['closure'])
    lim = [-0.02, max(0.05, b[['info_clo', 'info_hm']].max().max() * 1.1)]
    a.plot(lim, lim, 'k--', lw=1, alpha=.6)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel('closure info'); a.set_title(f'{gt}  (n={len(b)} both valid)')
    a.text(.05, .92, 'below the line =\nHailMary better', transform=a.transAxes,
           fontsize=7, va='top', color=C['other'])
ax[0].set_ylabel('closure + HailMary info')
fig.suptitle('HailMary never worsens a partition, and improves a minority', y=1.03)
save(fig, '05_paired')

# 6 -- reconstruction cannot discriminate
d = pd.read_csv(f'{DAT}/metric.csv'); r = d[d.graph_type == 'random'].dropna(subset=['recon_l1', 'null_std'])
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].hist(np.clip(r.recon_l1 / r.null_std, 0, 3), bins=40, color=C['closure'])
ax[0].axvline(1, color='k', ls='--', lw=1.2)
ax[0].set_xlabel('recon / null  (1.0 = no better than predicting no edges)')
ax[0].set_ylabel('partitions')
ax[0].set_title('~90% of valid abstractions are worse than the null')
from scipy import stats
res = []
for kk, g in d[d.graph_type == 'random'].groupby(['p', 'true_nhigh', 'seed']):
    g = g.dropna(subset=['info', 'recon_l1'])
    if g['info'].nunique() >= 4 and g.recon_l1.nunique() > 1:
        res.append(stats.spearmanr(g['info'], g.recon_l1).correlation)
ax[1].hist(res, bins=14, color=C['hm'])
ax[1].axvline(0, color='k', ls='--', lw=1.2)
ax[1].set_xlabel('within-instance Spearman(info, recon)')
ax[1].set_title(f'positive would support the hypothesis ({sum(np.array(res)>0)}/{len(res)})')
save(fig, '06_recon')

# 7 -- the three info measures agree exactly
d = pd.read_csv(f'{DAT}/infometric.csv')
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].scatter(d.info_raw, d.info_var, s=12, alpha=.5, color=C['closure'])
ax[0].set_xlabel('info_raw  (|A| weighted)'); ax[0].set_ylabel('info_var  (variance standardised)')
ax[1].scatter(d.info_raw, d.anc_lost, s=12, alpha=.5, color=C['hybrid'])
ax[1].set_xlabel('info_raw'); ax[1].set_ylabel('anc_lost  (weight-free)')
for a in ax:
    a.plot([0, 1], [0, 1], 'k--', lw=1, alpha=.5); a.set_xlim(0.3, 1); a.set_ylim(0.3, 1)
fig.suptitle('Three weightings, identical rankings (rho = +1.000 on 14/14 instances)', y=1.04)
save(fig, '07_infomeasures')

# 8 -- levers that did not work
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
sl = pd.read_csv(f'{DAT}/slack.csv'); sl = sl[(sl.graph_type == 'random') & (sl.returned == 1)]
g = sl.groupby('slack').agg(s=('success', 'mean'), i=('info', 'mean'))
ax[0].plot(g.index, g['s'], 'o-', color=C['hm'], label='success')
ax[0].plot(g.index, g['i'], 's-', color=C['hybrid'], label='info')
ax[0].set_xlabel('slack (k / ceiling)'); ax[0].set_title('slack: trades success for info')
ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
rs = pd.read_csv(f'{DAT}/restarts.csv')
g = rs.groupby('density').agg(a=('hm1', 'mean'), b=('hm_any', 'mean'))
w = 0.35; x = np.arange(len(g))
ax[1].bar(x - w/2, g['a'], w, label='1 restart', color=C['other'])
ax[1].bar(x + w/2, g['b'], w, label='5 restarts', color=C['hm'])
ax[1].set_xticks(x); ax[1].set_xticklabels(g.index); ax[1].set_xlabel('density')
ax[1].set_title('restarts: no effect'); ax[1].legend(fontsize=7); ax[1].set_ylabel('success')
ms = pd.read_csv(f'{DAT}/marginsweep.csv')
ms['hyb'] = np.where(ms['success'] == 1, np.minimum(ms['info'], ms['info_start']), ms['info_start'])
g = ms.groupby(['margin', 'noise']).agg(h=('hyb', 'mean'), mv=('moved', 'mean')).reset_index()
ax[2].scatter(g['mv'], g['h'], s=55, color=C['closure'])
ax[2].axhline(ms['info_start'].mean(), color='k', ls='--', lw=1.2)
ax[2].text(35, ms['info_start'].mean() + .002, 'closure alone', fontsize=7)
ax[2].set_xlabel('nodes moved off the initialisation'); ax[2].set_ylabel('info (with fallback)')
ax[2].set_title('initialisation: 5x more movement,\nsame result')
save(fig, '08_failed_levers')
print('done')

# 9 -- head-to-head against the existing baselines
d = pd.read_csv(f'{DAT}/benchmark.csv')
d['exact'] = ((d.obs_loss < 1e-8) & (d.intv_loss < 1e-8)).astype(float)
order = ['random', 'uca', 'hc_fast', 'closure', 'closure+HM']
col = {'random': '#bbbbbb', 'uca': '#999999', 'hc_fast': '#777777',
       'closure': C['closure'], 'closure+HM': C['hybrid']}
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
for a, (metric, ttl, logy) in zip(ax, [('exact', 'exact recovery\n(obs & intv loss < 1e-8)', False),
                                       ('info', 'information loss\n(valid partitions only pay this)', False),
                                       ('secs', 'runtime (s)', True)]):
    g = d.groupby(['graph_type', 'method'])[metric].mean().unstack()[order]
    x = np.arange(len(g)); w = 0.16
    for i, mth in enumerate(order):
        a.bar(x + (i - 2) * w, g[mth], w, label=mth, color=col[mth])
    a.set_xticks(x); a.set_xticklabels(g.index); a.set_title(ttl, fontsize=9)
    if logy:
        a.set_yscale('log')
ax[0].set_ylim(0, 1.08); ax[0].legend(fontsize=7, ncol=2)
fig.suptitle('closure reaches exact recovery on 100% of instances; no baseline exceeds 12.5%', y=1.05)
save(fig, '09_benchmark')

# 10 -- the fallback closes closure's structural failures
d = pd.read_csv(f'{DAT}/fallback.csv')
f = d[d.closure_ok == 0]
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
ax[0].bar([0, 1], [d.closure_ok.mean(), ((d.closure_ok == 1) | (d.fb_valid == 1)).mean()],
          0.55, color=[C['other'], C['closure']])
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['closure\nalone', 'closure +\nfallback'])
ax[0].set_ylabel('coverage (valid partition returned)'); ax[0].set_ylim(0, 1.08)
ax[0].set_title('coverage')
ax[1].bar([0, 1], [f.fb_info.mean(), 1.0], 0.55, color=[C['closure'], C['hm']])
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels([f'targeted merge\n(k={f.fb_k.mean():.0f})',
                                                 'collapse to one\ncluster (k=1)'])
ax[1].set_ylabel('information loss'); ax[1].set_ylim(0, 1.08)
ax[1].set_title(f'on the {len(f)} instances closure could not solve')
fig.suptitle('One merge, 32 ms, and closure always returns a valid partition', y=1.04)
save(fig, '10_fallback')
print('done 9-10')

# 14 -- cluster size distributions: the blob
d = pd.read_csv(f'{DAT}/clustersizes.csv')
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
allsz = np.concatenate([np.array(s.split(';'), int) for s in d.sizes])
bins = [1, 2, 3, 6, 11, 26, 51, 101, 10000]
h, _ = np.histogram(allsz, bins=bins)
nodes = np.array([allsz[(allsz >= bins[i]) & (allsz < bins[i+1])].sum() for i in range(len(bins)-1)])
lab = ['1', '2', '3-5', '6-10', '11-25', '26-50', '51-100', '101+']
x = np.arange(len(lab)); w = 0.4
ax[0].bar(x - w/2, 100*h/h.sum(), w, label='% of clusters', color=C['other'])
ax[0].bar(x + w/2, 100*nodes/nodes.sum(), w, label='% of nodes', color=C['closure'])
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=45); ax[0].legend(fontsize=7)
ax[0].set_ylabel('%'); ax[0].set_title('96% of clusters are singletons;\n32% of nodes sit in clusters of 101+')
r = d[d.graph_type == 'random']
g = r.groupby('density').agg(lf=('largest_frac', 'mean'), sf=('singleton_frac', 'mean'),
                             en=('entropy', 'mean'))
ax[1].plot(g.index, g.lf, 'o-', color=C['hm'], lw=1.7, label='largest cluster / p')
ax[1].plot(g.index, g.en, 's-', color=C['hybrid'], lw=1.7, label='normalised entropy')
oc = d[d.graph_type == 'one_child']
ax[1].axhline(oc.largest_frac.mean(), color=C['hm'], ls=':', lw=1.2)
ax[1].axhline(oc['entropy'].mean(), color=C['hybrid'], ls=':', lw=1.2)
ax[1].text(0.12, oc['entropy'].mean()+.02, 'one_child (dotted)', fontsize=7, color=C['other'])
ax[1].set_xlabel('density'); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7)
ax[1].set_title('random DAGs: the blob grows with density')
for p_, gg in r.groupby('p'):
    sz = np.concatenate([np.array(s.split(';'), int) for s in gg.sizes])
    ax[2].hist(np.log10(sz), bins=30, histtype='step', lw=1.6, label=f'p={p_}', density=True)
ax[2].set_xlabel('log10(cluster size)'); ax[2].set_ylabel('density')
ax[2].legend(fontsize=7); ax[2].set_title('bimodal at every p:\nsingletons + one blob')
fig.suptitle('Closure partitions are a giant blob plus a sea of singletons', y=1.05)
save(fig, '14_cluster_sizes')
print('done 14')
