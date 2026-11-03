"""T_soft over training, with the losses and the HARD metrics alongside.

Left   : T_soft (rows sorted by final cluster)
Top-R  : lambda-weighted training losses -- what actually drives the gradient
Bot-R  : hard metrics recomputed from argmax(T_soft) at each frame --
         All-In violating nodes, clusters used, intra-cluster edge mass
A moving line marks the current epoch on both plots.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)

P, SEED, DENS, GT, EPOCHS = 50, 2, 0.5, 'random', 6000
A, _, _, _ = sample_low_level_graph(
    n_high=10, n_low=P, density=DENS, shuffle=True, blockiness=0.5, heterogeneity=1.,
    internal_density=0., graph_type=GT, seed=SEED, connected=True, max_attempts=400)
Ab = (A != 0)
absA = np.abs(A); absA_sum = absA.sum()

rng = np.random.default_rng(SEED)
best = 0
for _ in range(12):
    g = mk.closure(A, rng)
    if all(mk.conditions(A, g)) and len(g) > best:
        best = len(g)
K = max(2, best)
print(f'closure k={K}', flush=True)

gen = torch.Generator().manual_seed(31337 + SEED)
li = closure_logits(A, K, rng=np.random.default_rng(31337 + SEED), generator=gen)
m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
h = m.train(num_epochs=EPOCHS, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False, keep_history=True)
print('trained', flush=True)


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
    return len(bad), Kc, (absA * co).sum() / absA_sum, int(M.sum(1).max() <= 1) if Kc else 1


assign = np.array([t.argmax(1) for t in h['T']])
ch = (assign[1:] != assign[:-1]).any(1)
last = int(np.where(ch)[0].max()) + 1 if ch.any() else 200
idx = np.unique(np.concatenate([np.linspace(0, last, 45).astype(int),
                                np.linspace(last, EPOCHS - 1, 12).astype(int)]))
print(f'last reassignment epoch {last}; {len(idx)} frames', flush=True)

order = np.argsort(assign[-1])
frames = [h['T'][i][order] for i in idx]
met = np.array([hard_metrics(h['T'][i]) for i in idx])   # viol, k_used, info, onechild
ep = idx

fig = plt.figure(figsize=(11, 6.5))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], hspace=0.32, wspace=0.25)
axT = fig.add_subplot(gs[:, 0])
axL = fig.add_subplot(gs[0, 1])
axM = fig.add_subplot(gs[1, 1])

im = axT.imshow(frames[0], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                interpolation='nearest')
axT.set_xlabel('cluster'); axT.set_ylabel('low-level node (sorted by final cluster)')
fig.colorbar(im, ax=axT, fraction=0.046, label='T_soft')

E = np.arange(EPOCHS)
for key, lab, c in [('graph_loss', 'graph x10 (All-In)', '#d62728'),
                    ('one_child_loss', 'one_child x100', '#1f77b4'),
                    ('info_loss', 'info x10', '#2ca02c')]:
    axL.plot(E, np.maximum(np.array(h[key]), 1e-6), lw=1.2, color=c, label=lab)
axL.set_yscale('log'); axL.set_xlim(0, EPOCHS)
axL.set_ylabel('weighted loss'); axL.legend(fontsize=7, loc='upper right', framealpha=0.9)
axL.set_title('training losses (as weighted in the objective)', fontsize=9)
vl1 = axL.axvline(0, color='k', lw=1.1)

axM.plot(ep, met[:, 0], color='#d62728', lw=1.4, label='All-In violating nodes')
axM.plot(ep, met[:, 1], color='#7f7f7f', lw=1.4, label='clusters used')
axM.set_xlim(0, EPOCHS); axM.set_xlabel('epoch'); axM.set_ylabel('count')
axM2 = axM.twinx()
axM2.plot(ep, met[:, 2], color='#2ca02c', lw=1.4, ls='--', label='intra-cluster mass')
axM2.set_ylabel('info', color='#2ca02c'); axM2.set_ylim(0, 1)
hs = axM.get_legend_handles_labels()[0] + axM2.get_legend_handles_labels()[0]
ls_ = axM.get_legend_handles_labels()[1] + axM2.get_legend_handles_labels()[1]
axM.legend(hs, ls_, fontsize=7, loc='center right', framealpha=0.9)
axM.set_title('hard metrics from argmax(T_soft)', fontsize=9)
vl2 = axM.axvline(0, color='k', lw=1.1)
d1 = axM.plot([ep[0]], [met[0, 0]], 'o', color='#d62728', ms=6)[0]
d2 = axM.plot([ep[0]], [met[0, 1]], 'o', color='#7f7f7f', ms=6)[0]
d3 = axM2.plot([ep[0]], [met[0, 2]], 'o', color='#2ca02c', ms=6)[0]
sup = fig.suptitle('', fontsize=10)

HOLD = 6


def update(f):
    i = f // HOLD
    im.set_array(frames[i])
    e = ep[i]
    vl1.set_xdata([e, e]); vl2.set_xdata([e, e])
    d1.set_data([e], [met[i, 0]]); d2.set_data([e], [met[i, 1]]); d3.set_data([e], [met[i, 2]])
    sup.set_text(f'epoch {e:5d}   |   violating nodes {int(met[i,0]):3d}   '
                 f'clusters {int(met[i,1]):2d}   info {met[i,2]:.3f}   '
                 f'one-child {"yes" if met[i,3] else "no"}')
    return im, vl1, vl2, d1, d2, d3, sup


ani = animation.FuncAnimation(fig, update, frames=len(frames) * HOLD, blit=False)
out = f'results/timelapse/T_soft_with_losses_{GT}_p{P}_k{K}.gif'
ani.save(out, writer='pillow', fps=15)
print('saved', out, flush=True)
print(f'start: viol={met[0,0]} k={met[0,1]} info={met[0,2]:.3f} | '
      f'end: viol={met[-1,0]} k={met[-1,1]} info={met[-1,2]:.3f}', flush=True)
