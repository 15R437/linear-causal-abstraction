"""Why does All-In (graph_loss) get minimised on random DAGs but not on one_child?

Trains both families at p=100/k=10 with history, and records for each:
  - the weighted loss terms the model optimises, per epoch;
  - the HARD graph_loss (losses.graph_loss on argmax(T)) per epoch, which is the
    criterion we actually score -- the model only ever sees a soft surrogate of it;
  - the learned partition next to the generating one, and the per-edge All-In violations.

Run:  venv/bin/python experiments/abstraction_study/scripts/diagnose_graph_loss.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import cluster_indices, high_level_mask
from src.utils.losses import graph_loss
from src.utils.plots import animate_matrices, plot_assignment_raster

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8983', '#e8e7e3', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'axes.titlecolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'grid.color': GRID, 'grid.linewidth': 0.7, 'legend.frameon': False, 'font.size': 9,
})

OUT = Path('results/diagnostics')
FAMILIES = [('one_child', 1.0), ('random', 0.5)]
N_LOW, N_HIGH, SEED = 100, 10, 0
EPOCHS = 7000
STRIDE = 25          # how often to evaluate the hard criterion


def allin_report(A_low, T_hard):
    """Per-edge All-In check against the structural high-level mask."""
    idx = cluster_indices(T_hard)
    mask = high_level_mask(A_low, T_hard)
    Ab = (A_low != 0)
    out = []
    for a in range(T_hard.shape[1]):
        for b in np.where(mask[a, :])[0]:
            par, ch = idx[a], idx[b]
            hit = [p for p in par if Ab[p][ch].any()]
            out.append((a, int(b), len(hit), len(par)))
    return out


def run(graph_type, density):
    A_low, T_gt, _, A_high_gt = sample_low_level_graph(
        n_high=N_HIGH, n_low=N_LOW, density=density, shuffle=True, blockiness=0.5,
        heterogeneity=1.0, internal_density=0.0, graph_type=graph_type, seed=SEED,
        connected=True, max_attempts=200)

    torch.manual_seed(1000 * SEED + 7)
    model = HailMary(n_low=N_LOW, n_high=N_HIGH, A_low=torch.from_numpy(A_low).float())
    history = model.train(num_epochs=EPOCHS, lambda_recon=0.0, lambda_surj=10.0,
                          lambda_graph=10.0, lambda_one_child=100.0,
                          verbose=False, keep_history=True)

    # The hard criterion, recomputed from the stored soft T at intervals.
    hard_epochs, hard_gl, n_used = [], [], []
    for e in range(0, len(history['T']), STRIDE):
        Ts = history['T'][e]
        Th = np.zeros_like(Ts)
        Th[np.arange(N_LOW), Ts.argmax(1)] = 1.0
        hard_epochs.append(e)
        hard_gl.append(graph_loss(A_low=A_low, T_hard=Th))
        n_used.append(int((Th.sum(0) > 0).sum()))

    T_hard = np.zeros((N_LOW, N_HIGH))
    T_hard[np.arange(N_LOW), history['T'][-1].argmax(1)] = 1.0
    return dict(graph_type=graph_type, A_low=A_low, T_gt=T_gt, A_high_gt=A_high_gt,
                history=history, hard_epochs=np.array(hard_epochs),
                hard_gl=np.array(hard_gl), n_used=np.array(n_used), T_hard=T_hard)


def figure_dynamics(results):
    fig, axes = plt.subplots(2, len(results), figsize=(6.6 * len(results), 7.4), squeeze=False)
    for j, r in enumerate(results):
        h = r['history']
        ax = axes[0][j]
        for i, key in enumerate(['graph_loss', 'one_child_loss', 'surjective_loss', 'entropy_loss']):
            ax.plot(np.array(h[key]) + 1e-12, color=SERIES[i], lw=1.4,
                    label=key.replace('_loss', '') + ' (weighted)')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('weighted loss' if j == 0 else None)
        ax.set_title(f"{r['graph_type']}  p={N_LOW}, k={N_HIGH}  — what the model optimises",
                     loc='left', pad=8)
        ax.grid(True, axis='y'); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.legend(fontsize=8)

        ax = axes[1][j]
        ax.plot(r['hard_epochs'], r['hard_gl'], color=SERIES[0], lw=1.8,
                label='hard graph_loss on argmax(T)')
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel('epoch')
        ax.set_ylabel('hard graph_loss (All-In)' if j == 0 else None)
        ax.set_title(f"final = {r['hard_gl'][-1]:.1f}   "
                     f"(0 means All-In holds)", loc='left', pad=8)
        ax.grid(True, axis='y'); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax2 = ax.twinx()
        ax2.plot(r['hard_epochs'], r['n_used'], color=MUTED, lw=1.0, ls=':')
        ax2.set_ylabel('clusters used', color=MUTED, fontsize=8)
        ax2.set_ylim(0, N_HIGH + 0.5)
        for sp in ('top',):
            ax2.spines[sp].set_visible(False)
    fig.suptitle('The model minimises a soft surrogate; the criterion we score is the hard one',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'dyn_losses.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for gt, d in FAMILIES:
        print(f"training {gt} ...", flush=True)
        results.append(run(gt, d))

    figure_dynamics(results)

    for r in results:
        gt = r['graph_type']
        Th, A_low, T_gt = r['T_hard'], r['A_low'], r['T_gt']
        gt_assign = np.abs(T_gt).argmax(1)
        learned = Th.argmax(1)
        print(f"\n{'='*78}\n{gt}  p={N_LOW} k={N_HIGH} seed={SEED}\n{'='*78}")
        print(f"  final hard graph_loss = {r['hard_gl'][-1]:.2f}   "
              f"min over training = {r['hard_gl'].min():.2f} "
              f"at epoch {r['hard_epochs'][r['hard_gl'].argmin()]}")
        print(f"  soft graph_loss: first={r['history']['graph_loss'][0]:.3f} "
              f"last={r['history']['graph_loss'][-1]:.3f}")
        print(f"  clusters used = {int((Th.sum(0) > 0).sum())}/{N_HIGH}")

        print("\n  learned clusters (generating cluster of each member in brackets):")
        for c in range(N_HIGH):
            mem = np.where(learned == c)[0]
            if not len(mem):
                print(f"    Y{c+1:<2}: <empty>")
                continue
            src = [gt_assign[i] for i in mem]
            uniq, cnt = np.unique(src, return_counts=True)
            purity = cnt.max() / len(mem)
            print(f"    Y{c+1:<2}: n={len(mem):<3} purity={purity:.2f}  "
                  f"drawn from gt clusters {dict(zip(uniq.tolist(), cnt.tolist()))}")

        print("\n  All-In per high-level edge (members reaching the child / cluster size):")
        bad = 0
        for a, b, hit, tot in allin_report(A_low, Th):
            flag = 'ok  ' if hit == tot else 'FAIL'
            if hit != tot:
                bad += 1
            print(f"    {flag} Y{a+1}->Y{b+1}: {hit}/{tot}")
        print(f"  => {bad} violated edges")

        # the user's own tools
        fig = plot_assignment_raster(r['history'], subsample=50, T_gt=T_gt, soft=False)
        fig.savefig(OUT / f'raster_{gt}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        animate_matrices(r['history']['T'], xlabel='high-level cluster',
                         ylabel='low-level variable', title=f'T_{gt}',
                         subsample=100, save_dir=str(OUT) + '/')
    print(f"\nwrote figures to {OUT}")
