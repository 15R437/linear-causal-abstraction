"""Timelapse: soft-only vs graph-hard, same instance, same init, frame for frame.

Two runs are trained from an IDENTICAL initialisation and differ only in whether a hard
(argmax) copy of the graph loss is added at weight W. Each row shows that run's T_soft
alongside the hard metrics recomputed from argmax(T_soft) at every frame, so you can watch
the discrete partition move rather than only the soft surrogate.

The point of interest is the cluster count: at lambda_oc=1000 the soft-only run collapses
to a handful of clusters (buying RAE=0 by marginalising variables away), and the hard term
is what stops it.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from hardmix import HardMix
from occ import occ

P, SEED, DENS, EPOCHS = 50, 2, 0.5, 2000
W = float(os.environ.get('ANIM_W', '0.1'))
LAM_OC = float(os.environ.get('ANIM_LAM', '1000'))

A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=DENS, shuffle=True,
    blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
    seed=SEED, connected=True, max_attempts=400)
Ab = (A != 0); absA = np.abs(A); tot = absA.sum()
K = max(occ(A, seed=SEED)[2]['k'], 2)
print('p=%d K=%d lam_oc=%g W=%g' % (P, K, LAM_OC, W), flush=True)


def hard_metrics(Tsoft):
    lab = Tsoft.argmax(1)
    groups = [np.where(lab == j)[0] for j in range(Tsoft.shape[1])]
    groups = [g for g in groups if len(g)]
    Kc = len(groups)
    M = np.zeros((Kc, Kc), bool)
    for x in range(Kc):
        for y in range(Kc):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    bad = set()
    for x in range(Kc):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    bad.add(int(i))
    T = np.zeros((P, Kc))
    for j, gg in enumerate(groups):
        T[gg, j] = 1
    co = T @ T.T; np.fill_diagonal(co, 0)
    return len(bad), Kc, (absA * co).sum() / tot


def run(wg):
    torch.manual_seed(31337 + SEED)              # identical init for both arms
    m = HardMix(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
    m.w_graph_hard, m.w_recon_hard = wg, 0.
    h = m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100.,
                lambda_one_child=LAM_OC, lambda_info=0., lambda_surj=0,
                verbose=False, keep_history=True)
    return h


runs = [('soft only', run(0.0), '#6b6b6b'),
        ('+ graph-hard (w=%g)' % W, run(W), '#e4572e')]
print('trained both', flush=True)

idx = np.unique(np.concatenate([np.linspace(0, 400, 34).astype(int),
                                np.linspace(400, EPOCHS - 1, 20).astype(int)]))
met = {lab: np.array([hard_metrics(h['T'][i]) for i in idx]) for lab, h, _ in runs}
orders = {lab: np.argsort(h['T'][-1].argmax(1)) for lab, h, _ in runs}
frames = {lab: [h['T'][i][orders[lab]] for i in idx] for lab, h, _ in runs}
print('%d frames' % len(idx), flush=True)

fig = plt.figure(figsize=(10.5, 6.2))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.75], hspace=0.30, wspace=0.22)
arts = []
for r, (lab, h, col) in enumerate(runs):
    axT = fig.add_subplot(gs[r, 0]); axM = fig.add_subplot(gs[r, 1])
    im = axT.imshow(frames[lab][0], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                    interpolation='nearest')
    axT.set_ylabel('node (sorted by final cluster)', fontsize=8)
    axT.set_xlabel('cluster', fontsize=8); axT.tick_params(labelsize=7)
    axT.set_title(r'$T_{\rm soft}$ — %s' % lab, fontsize=9.5, color=col)

    M = met[lab]
    axM.plot(idx, M[:, 1], color='#3d6bd6', lw=1.5, label='clusters used')
    axM.plot(idx, M[:, 0], color='#e4572e', lw=1.5, label='All-In violating nodes')
    axM.set_xlim(0, EPOCHS); axM.set_ylim(0, max(K, met['soft only'][:, 0].max()) * 1.1)
    axM.set_ylabel('count', fontsize=8); axM.tick_params(labelsize=7)
    axM.set_title('hard metrics from argmax($T_{\\rm soft}$) — %s' % lab,
                  fontsize=9.5, color=col)
    ax2 = axM.twinx()
    ax2.plot(idx, M[:, 2], color='#118a6e', lw=1.5, ls='--', label='intra-cluster mass')
    ax2.set_ylim(0, 1); ax2.set_ylabel('info loss', color='#118a6e', fontsize=8)
    ax2.tick_params(labelsize=7, colors='#118a6e')
    if r == 1:
        axM.set_xlabel('epoch', fontsize=8)
    if r == 0:   # ONE shared legend for the figure; per-panel boxes sat on the data
        _hs = axM.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        _lb = axM.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        fig.legend(_hs, _lb, fontsize=8, ncol=3, frameon=False, loc='lower center',
                   bbox_to_anchor=(0.5, -0.005))
    vl = axM.axvline(0, color='k', lw=1.1)
    d1 = axM.plot([idx[0]], [M[0, 1]], 'o', color='#3d6bd6', ms=5.5)[0]
    d2 = axM.plot([idx[0]], [M[0, 0]], 'o', color='#e4572e', ms=5.5)[0]
    d3 = ax2.plot([idx[0]], [M[0, 2]], 'o', color='#118a6e', ms=5.5)[0]
    arts.append((im, vl, d1, d2, d3, M, lab))

sup = fig.suptitle('', fontsize=10.5)
fig.subplots_adjust(bottom=0.11)
# No frame duplication: PillowWriter collapsed the repeats, so HOLD bought nothing and the
# gif ran at ~4s. Slow it with fps instead -- 53 frames at 6fps is ~9s, readable.
HOLD = 1


def update(f):
    i = f // HOLD
    e = idx[i]
    txt = []
    for im, vl, d1, d2, d3, M, lab in arts:
        im.set_array(frames[lab][i])
        vl.set_xdata([e, e])
        d1.set_data([e], [M[i, 1]]); d2.set_data([e], [M[i, 0]]); d3.set_data([e], [M[i, 2]])
        txt.append('%s: k=%d  viol=%d  info=%.3f' % (lab, M[i, 1], M[i, 0], M[i, 2]))
    sup.set_text(r'$\lambda_{\rm oc}=%g$   epoch %4d        ' % (LAM_OC, e) + '     |     '.join(txt))
    return []


ani = animation.FuncAnimation(fig, update, frames=len(idx) * HOLD, blit=False)
out = 'experiments/abstraction_study/figures/hardmix_training_w%g.gif' % W
ani.save(out, writer='pillow', fps=6)
print('saved', out, flush=True)
for lab, h, _ in runs:
    M = met[lab]
    print('  %-24s start k=%d info=%.3f -> end k=%d viol=%d info=%.3f'
          % (lab, M[0, 1], M[0, 2], M[-1, 1], M[-1, 0], M[-1, 2]), flush=True)
