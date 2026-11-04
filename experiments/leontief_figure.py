"""The learned high-level graph on the UK Leontief (EXIOBASE3, GB) economy, one panel
per method, with each panel's metrics in its title so the picture and the numbers are read
together.

Nodes are the NON-EMPTY clusters, sized by membership. Edges are the fitted A_high entries,
width by |weight|. Each cluster is labelled by its highest-weighted member -- a gloss, not a
claim about what the cluster means, and emphatically not a description of the other members:
the largest cluster here holds over a hundred sectors.

Layout is by longest-path depth, which is well defined because the quotients here are acyclic
even though the low-level graph is not: 22-31 sectors consume their own output (Forestry at
0.244, for one), so `is_dag(A_low)` is False while every SCC is a single node.

Requires data/A_low.npy and data/A_low_labels.pkl, both written by
experiments/leontief.py::parse_data.

    python experiments/leontief_figure.py            # -> assets/fig4_leontief.png
"""

import pickle
import re
import textwrap
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from src.algos.greedy import aoc_closure, greedy_allin
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import RAE, reconstruction_loss
from src.utils.helper import reconstruct_adj

THRESH = 0.05   # prune |A| below this; the raw table is dense with negligible coefficients
K = 5           # a drawing budget, not an experimental one: k=20 is unreadable as a picture
WRAP = 26       # characters per label line; names run to 99 chars and are NEVER truncated

# SEEDED, and with the same restart count as the other experiments. The closures draw random
# merge orders, so an unseeded call redraws the partition on every run -- which moved RAE
# across 5.9e-17 to 7.9e-17 and relabelled clusters between two runs of this very script.
# A figure in a README has to come out the same way twice.
SEED, N_RESTARTS = 0, 12

METHODS = [('AOC Closure',
            lambda A, k: aoc_closure(A, k, n_restarts=N_RESTARTS, seed=SEED)),
           ('Greedy All-In',
            lambda A, k: greedy_allin(A, k))]

# categorical bubble fills, legible on white and distinguishable in greyscale
PALETTE = ['#8c6d1f', '#7a7a7a', '#17807c', '#8d5bb5', '#d4701a', '#2a78d6', '#b03a3a']
EDGE_COLOR = '#d62728'


def load():
    """A_low thresholded and restricted to rows/columns that still carry an edge."""
    A0 = np.load('data/A_low.npy')
    with open('data/A_low_labels.pkl', 'rb') as f:
        L = pickle.load(f)
    names_all = [s for (_, s) in L['sectors']] + list(L['factors'])
    A = np.where(np.abs(A0) > THRESH, A0, 0.0)
    live = np.where((A != 0).any(0) | (A != 0).any(1))[0]
    return A[np.ix_(live, live)], [names_all[i] for i in live]


# EXIOBASE spells these out at up to 99 characters. The long form says nothing the short
# form does not, and a five-line label crowds the drawing, so these four are shortened by
# hand. Everything else is shown in full -- the cluster's gloss is the only thing telling the
# reader what it is, so names are never silently truncated.
ALIASES = {
    'Extraction of natural gas and services': 'Extraction of natural gas',
    'Extraction of crude petroleum and services': 'Extraction of crude oil',
    'Manufacture of gas; distribution': 'Manufacture of gas',
    'Financial intermediation, except': 'Financial intermediation',
}


def label(s):
    """An EXIOBASE sector name, tidied but never truncated.

    These run to 99 characters ("Fishing, operating of fish hatcheries and fish farms; service
    activities incidental to fishing (05)"), and the name is the only thing telling the reader
    what a cluster is, so it is wrapped over as many lines as it needs rather than cut. Four
    names are replaced by a shorter equivalent via ALIASES; the trailing numeric code is
    dropped, since it identifies the row rather than the industry.
    """
    s = re.sub(r'\s*\(\d+\)\s*$', '', s.strip()).rstrip(' ,;.-')
    for prefix, short in ALIASES.items():
        if s.startswith(prefix):
            s = short
            break
    return '\n'.join(textwrap.wrap(s, WRAP))


def depths(M):
    """Longest-path depth of each node in an acyclic boolean quotient."""
    k = len(M)
    d = np.zeros(k, int)
    for _ in range(k):
        changed = False
        for i in range(k):
            for j in np.where(M[i])[0]:
                if d[j] < d[i] + 1:
                    d[j] = d[i] + 1
                    changed = True
        if not changed:
            break
    return d


def analyse(A_low, names, fn):
    partition = [c for c in fn(A_low, K) if len(c)]
    A_high, T, S = fit_abstraction_eigen(A_low=A_low, partition=partition)
    A_recon = reconstruct_adj(A_high=A_high, T=T)
    E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)

    co = np.zeros_like(A_low)
    for c in partition:
        co[np.ix_(c, c)] = 1.0
    np.fill_diagonal(co, 0)
    absA = np.abs(A_low)

    return dict(
        partition=partition,
        A_high=A_high,
        labels=[label(names[c[np.argmax(np.abs(T[c, j]))]]) for j, c in enumerate(partition)],
        rae=RAE(A_low=A_low, T=T, A_high=A_high, S=S, E_low=E_low),
        recon=reconstruction_loss(A_low=A_low, T=T, A_high=A_high, A_hat=A_recon),
        intra=float((absA * co).sum() / max(absA.sum(), 1e-12)),
        intra_edges=float(((A_low != 0) * co).sum() / max(int((A_low != 0).sum()), 1)))


def draw(ax, r, title):
    """One panel: cluster bubbles by depth layer, arrows weighted by |A_high|."""
    partition, A_high = r['partition'], r['A_high']
    M = np.abs(A_high) > 1e-9
    k = len(partition)
    d = depths(M)

    # x spread across the full width of the layer, y one row per depth. Layers are kept
    # close together because the labels, not the arrows, are what need the room.
    X, DY = 2.1, 0.78
    pos = np.zeros((k, 2))
    for layer in np.unique(d):
        idx = np.where(d == layer)[0]
        xs = np.linspace(0, X, len(idx) + 1)[:-1] + X / (2 * len(idx))
        for i, x in zip(idx, xs):
            pos[i] = (x, -DY * float(layer))

    sizes = np.array([len(c) for c in partition], float)
    radii = 0.035 + 0.105 * np.sqrt(sizes / sizes.max())

    for i in range(k):
        for j in np.where(M[i])[0]:
            w = abs(A_high[i, j]) / max(np.abs(A_high).max(), 1e-12)
            u = pos[j] - pos[i]
            n = np.linalg.norm(u) or 1.0
            ax.add_patch(FancyArrowPatch(pos[i] + u / n * radii[i], pos[j] - u / n * radii[j],
                                         arrowstyle='-|>', mutation_scale=10,
                                         lw=0.7 + 2.3 * w, color=EDGE_COLOR,
                                         shrinkA=0, shrinkB=0, alpha=0.85, zorder=1))

    # Label placement follows the node's role, which is what keeps text off the arrows:
    # sources sit above their node, sinks below, and anything with both parents and children
    # goes out to the side, where the vertical edges are not.
    indeg, outdeg = M.sum(0), M.sum(1)
    ytop, ybot = max(pos[:, 1]), min(pos[:, 1])
    for i in range(k):
        ax.add_patch(Circle(pos[i], radii[i], facecolor=PALETTE[i % len(PALETTE)],
                            edgecolor='white', lw=0.6, zorder=2))
        n_sec = int(sizes[i])
        text = f"{r['labels'][i]}\n({n_sec} sector{'s' if n_sec != 1 else ''})"
        off = radii[i] * 165 + 9
        if indeg[i] == 0:
            kw = dict(xytext=(0, off), ha='center', va='bottom')
        elif outdeg[i] == 0:
            kw = dict(xytext=(0, -off), ha='center', va='top')
        elif pos[i, 0] <= X / 2:
            kw = dict(xytext=(-off, 0), ha='right', va='center')
        else:
            kw = dict(xytext=(off, 0), ha='left', va='center')
        ax.annotate(text, pos[i], textcoords='offset points', fontsize=6.6,
                    linespacing=1.3, zorder=3, **kw)

    ax.set_title(f"{title}\nRAE {r['rae']:.2g}   ReconErr {r['recon']:.3f}   "
                 f"intra {r['intra']:.3f} mass / {r['intra_edges']:.3f} edges",
                 fontsize=9.5, pad=14)
    ax.set_xlim(-0.52, X + 0.52)
    ax.set_ylim(ybot - 0.80, ytop + 0.72)
    ax.set_aspect('equal')
    ax.axis('off')


def main(save_to='assets/fig4_leontief.png'):
    mpl.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix'})
    A_low, names = load()
    print(f"A_low {A_low.shape} after pruning at |a| > {THRESH}")

    results = [(name, analyse(A_low, names, fn)) for name, fn in METHODS]
    fig, axes = plt.subplots(1, len(results), figsize=(7.0 * len(results), 5.2))
    for ax, (name, r) in zip(np.atleast_1d(axes), results):
        draw(ax, r, name)
        print(f"  {name:14s} k={len(r['partition'])}  RAE {r['rae']:.3g}  "
              f"ReconErr {r['recon']:.3f}  intra {r['intra']:.3f} mass "
              f"{r['intra_edges']:.3f} edges  "
              f"sizes {[len(c) for c in r['partition']]}")
    fig.tight_layout()
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=300, bbox_inches='tight')
    print(f"Saved {save_to}")
    return results


if __name__ == '__main__':
    main()
