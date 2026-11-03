"""The learned high-level graph on the Leontief (EXIOBASE3, GB) economy, with its metrics.

One panel per method. Nodes are the NON-EMPTY clusters, sized by how many members actually
carry weight (marginalised members -- fitted |T| below tau -- are excluded from the count and
reported separately, since they contribute nothing to the abstraction). Edges are the fitted
A_high entries, width by |weight|. Each panel's title carries RAE, ReconErr and intra so the
picture and the numbers are read together.

Layout is by longest-path depth, which is well defined because the quotients here are acyclic
(the LOW-LEVEL graph is a DAG apart from self-loops: 22-31 sectors consume their own output,
e.g. Forestry 0.244, so `is_dag` is False while the largest SCC is a single node).

Each cluster is labelled by its single highest-weighted member -- a gloss, not a claim about
what the cluster means, and emphatically not a description of the other members (the largest
cluster here holds 111 sectors).
"""
import os, sys, pickle, time, argparse, textwrap, warnings
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import re
import numpy as np, torch, networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from partition_baselines import allin_agglom
from src.models.hail_mary import HailMary, check_conditions
from src.utils.helper import (fit_abstraction_auto, get_S, cluster_indices, is_dag,
                              quotient_mask)
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import RAE, reconstruction_loss
from occ import occ, oc_closure, budget_coarsening

B = '/Users/issy/Projects/linear-causal-abstraction'
TAU, EPOCHS = 1e-2, 3000
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
# same convention as plots.visualize_causal_graph: red positive, navy negative
POS, NEG = '#d62728', '#1f3b73'
# Node palette deliberately excludes red and dark blue: Set1's first colour is the same red
# as a positive edge, which makes a node read as an edge crossing behind it.
NODE_COLS = ['#0f8b8d', '#e08214', '#7b3294', '#8c6d31', '#7f7f7f',
             '#17becf', '#bcbd22', '#c994c7', '#5aae61', '#dfc27d']

ap = argparse.ArgumentParser()
ap.add_argument('--thr', type=float, default=0.05)
ap.add_argument('--fitter', default='auto', choices=['auto', 'eigen'],
                help="'eigen' is the closed-form SCC solver: sinks free, cyclic SCCs pinned "
                     'by an eigenproblem. The Leontief quotients are acyclic apart from '
                     'self-loops, so this mainly tests it outside the synthetic generator.')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--methods', nargs='+', default=['AOC Closure', 'Greedy All-In', 'GradCAP'])
ap.add_argument('--gc-restarts', type=int, default=10,
                help='GradCAP restarts; the best by ReconErr is kept. Selection is part of '
                     'the method, not seed-picking: at lambda_oc=0 only 1 run in 10 avoids '
                     'a degenerate near-zero A_high, so a single run is not representative.')
ap.add_argument('--title', default='Learned high-level graphs',
                help='kept short on purpose; thresholds/p/k belong in the caption')
ap.add_argument('--lam-oc', type=float, default=0.0,
                help='lambda_one_child for GradCAP; shown in its panel title')
ap.add_argument('--out', default=None)
ap.add_argument('--k', type=int, default=5,
                help='cluster budget. 20 is unreadable as a drawing; 5-8 is legible.')
args = ap.parse_args()


def _fit(A_low, T_hard, E_low, seed):
    """Dispatch on --fitter. Returns (A_high, T, method_name)."""
    if args.fitter == 'eigen':
        Ah, Tp = fit_abstraction_eigen(A_low=A_low, T_hard=T_hard, seed=seed,
                                       n_restarts=5)[:2]
        return Ah, Tp, 'eigen'
    return fit_abstraction_auto(A_low=A_low, T_hard=T_hard, E_low=E_low, n_restarts=5,
                                seed=seed, maxiter=100, faithful=True, return_method=True)
K = args.k
# few nodes -> room for real labels; many nodes -> shrink to fit
FS = 7.0 if K <= 6 else (5.8 if K <= 10 else 4.8)
NAMELEN = 28 if K <= 6 else (22 if K <= 10 else 17)

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

A0 = np.load(f'{B}/data/A_low.npy')
L = pickle.load(open(f'{B}/data/A_low_labels.pkl', 'rb'))
names_all = [s for (_, s) in L['sectors']] + list(L['factors'])
A = np.where(np.abs(A0) > args.thr, A0, 0.0)
live = np.where((A != 0).any(0) | (A != 0).any(1))[0]
A = A[np.ix_(live, live)]
names = [names_all[i] for i in live]
p = A.shape[0]
E = np.linalg.inv(np.eye(p) - A); su = np.ones(p)
absA = np.abs(A); tot = absA.sum()


def partition(meth):
    if meth == 'AOC Closure':
        return occ(A, n_draws=12, seed=0, k=K)[1]
    if meth == 'OC Closure':
        g, T, _ = oc_closure(A, n_draws=12, seed=0)
        if len(g) > K:
            try:
                g2, _ = budget_coarsening(A, g, K, rng=np.random.default_rng(0))
                T = np.zeros((p, len(g2)))
                for j, mem in enumerate(g2):
                    T[np.asarray(mem), j] = 1.0
            except Exception:
                pass
        return T
    if meth == 'Greedy All-In':
        return allin_agglom(A, min(K, p))
    # Restart selection, as occ() does over its 12 draws: train `--gc-restarts` times and
    # keep the partition whose fitted abstraction reconstructs best. Measured at k=5 on this
    # graph with lambda_oc=0, 9 of 10 restarts return an essentially edgeless A_high
    # (max |A_high| ~ 1e-2 against a real scale of ~1.4), so reporting one run would
    # misrepresent the method in either direction depending on the seed drawn.
    best, best_recon = None, np.inf
    for sd in range(args.gc_restarts):
        torch.manual_seed(args.seed + sd)
        m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100.,
                lambda_one_child=args.lam_oc, lambda_info=0., lambda_surj=0, verbose=False)
        Th_c = m.get_partition()[0].cpu().detach().numpy()
        try:
            Ah_c, Tp_c = _fit(A, Th_c, E, 0)[:2]
            rc = reconstruction_loss(A_low=A, T=Tp_c, A_high=Ah_c)
        except Exception:
            continue
        if rc < best_recon:
            best, best_recon = Th_c, rc
    return best


def short(s, n=NAMELEN):
    """Shorten a sector name WITHOUT a trailing ellipsis.

    EXIOBASE names are long and comma-qualified ("Extraction of crude petroleum and
    services related to crude oil extraction, excluding surveying"). Take the part before
    the first comma/semicolon, drop the trailing NACE code, then WRAP rather than truncate,
    so nothing is silently cut off.
    """
    s = s.split(';')[0].split(',')[0].strip()
    s = re.sub(r'\s*\(\d+\)$', '', s).strip()
    # Abbreviate at a WORD boundary rather than wrapping the whole name: the full EXIOBASE
    # names run to four wrapped lines each, which is what made the figure tall. Cutting whole
    # words keeps it readable and leaves no trailing ellipsis.
    words, out = s.split(), []
    for w in words:
        if out and len(' '.join(out + [w])) > n:
            break
        out.append(w)
    return '\n'.join(textwrap.wrap(' '.join(out), n // 2 + 6))


def analyse(meth):
    Th = partition(meth)
    ai, oc, dq = check_conditions(A, cluster_indices(Th))
    Ah, Tp, fm = _fit(A, Th, E, args.seed)
    S = get_S(T=Tp, T_hard=Th, A_low=A)
    rae = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su, tau=TAU)
    rec = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
    co = Th @ Th.T; np.fill_diagonal(co, 0)
    # the label says "edges", so report the unweighted EDGE COUNT fraction (matching
    # `intra_edges` in the synthetic tables), not the mass-weighted version
    intra = float(((A != 0) * co).sum() / max(int((A != 0).sum()), 1))
    intra_mass = float((absA * co).sum() / tot)
    marg = np.abs(Tp).max(1) <= TAU
    idx = cluster_indices(Th)
    kept = [j for j, c in enumerate(idx) if len(c) and not marg[c].all()]
    lab, size = {}, {}
    for j in kept:
        mem = [i for i in idx[j] if not marg[i]]
        size[j] = len(mem)
        top = int(np.argmax(np.abs(Tp[mem, j])))
        lab[j] = short(names[mem[top]])
    # A_high is already masked by the quotient structure inside fit_abstraction_K (added
    # 2026-09-24), so a nonzero entry here is guaranteed to have a real low-level block
    # behind it. No magnitude threshold is applied or needed.
    qm = quotient_mask(A, Th)[np.ix_(kept, kept)]
    sub = Ah[np.ix_(kept, kept)]
    edges = [(kept[a], kept[b], float(sub[a, b])) for a in range(len(kept))
             for b in range(len(kept)) if a != b and sub[a, b] != 0.0]

    off = ~np.eye(len(kept), dtype=bool)
    print('  %-15s structural %d/%d pairs; |A_high| kept %s; dropped max %.1e'
          % (meth, int((qm & off).sum()), int(off.sum()),
             np.array2string(np.sort(np.abs(sub[off]))[::-1][:6], precision=2),
             float(np.abs(Ah[np.ix_(kept, kept)][off & ~qm]).max()) if (off & ~qm).any() else 0.))
    return dict(meth=meth, kept=kept, lab=lab, size=size, edges=edges, rae=rae, recon=rec,
                intra=intra, intra_mass=intra_mass, marg=int(marg.sum()), allin=bool(ai), oc=bool(oc),
                acyc=bool(dq), fm=fm)


ALL_TEXTS = []
res = [analyse(m) for m in args.methods]
fig, axes = plt.subplots(1, len(res), figsize=(3.4 * len(res), 3.0), squeeze=False)
for ax, r in zip(axes[0], res):
    G = nx.DiGraph()
    G.add_nodes_from(r['kept'])
    for a, b, w in r['edges']:
        G.add_edge(a, b, w=abs(w))
    # Vertical gap between layers. Node markers are sized in POINTS while positions are in
    # data units, so a big node (111 members) visually eats the short edges around it at
    # unit spacing; stretching y pushes the layers apart without shrinking the nodes.
    Y_GAP = 3.0
    # layered by longest-path depth; falls back to spring layout if a cycle slipped through
    rows = {}
    try:
        depth = {n: 0 for n in G.nodes}
        for n in nx.topological_sort(G):
            for s_ in G.successors(n):
                depth[s_] = max(depth[s_], depth[n] + 1)
        rows = {}
        for n, dp in depth.items():
            rows.setdefault(dp, []).append(n)
        pos = {}
        for dp, ns in rows.items():
            ns = sorted(ns, key=lambda n: -r['size'][n])
            for i, n in enumerate(ns):
                pos[n] = (1.6 * (i - (len(ns) - 1) / 2.0), -Y_GAP * dp)
    except nx.NetworkXUnfeasible:
        pos = nx.spring_layout(G, seed=0)

    sgn = {(a, b): w for a, b, w in r['edges']}
    ws = np.array([abs(sgn[(a, b)]) for a, b in G.edges]) if G.number_of_edges() else np.array([1.])
    ecol = [POS if sgn[(a, b)] > 0 else NEG for a, b in G.edges]
    nsz = [55 + 40 * np.sqrt(r['size'][n]) for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ecol, alpha=0.85,
                           width=0.5 + 1.9 * ws / ws.max(), arrowsize=6,
                           node_size=nsz, connectionstyle='arc3,rad=0.07')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=nsz,
                           node_color=[NODE_COLS[i % len(NODE_COLS)]
                                       for i, _ in enumerate(G.nodes)],
                           alpha=0.9, linewidths=0.9, edgecolors='white')
    # Place each label where it cannot sit on top of an edge:
    #   source (no incoming) -> ABOVE, nothing comes down into it
    #   sink   (no outgoing) -> BELOW, nothing leaves downward
    #   interior            -> to the SIDE, since edges pass both above and below it.
    # The interior case is what the big Greedy All-In cluster hits: it has edges in and out,
    # so any vertical label lands on one of them.
    texts = []          # (artist, node) so collisions can be resolved after drawing
    xs = [q[0] for q in pos.values()]
    xmid = (min(xs) + max(xs)) / 2.0 if xs else 0.0
    for n, (x, y) in pos.items():
        rad = (nsz[list(G.nodes).index(n)] ** 0.5) / 95.0     # node radius in data units
        txt = f'{r["lab"][n]}\n({r["size"][n]} sectors)'
        indeg, outdeg = G.in_degree(n), G.out_degree(n)
        if outdeg == 0:                       # sink: hang the label below
            texts.append((ax.text(x, y - rad - 0.10, txt, fontsize=FS, ha='center',
                                  va='top', color=INK, linespacing=1.15), n))
        elif indeg == 0:                      # source: sit the label above
            texts.append((ax.text(x, y + rad + 0.10, txt, fontsize=FS, ha='center',
                                  va='bottom', color=INK, linespacing=1.15), n))
        else:
            # interior: a vertical label is only a problem if a neighbour sits roughly
            # straight above/below, since that is where its edge runs. Otherwise the
            # space is free and above reads better than a side label.
            BAND = 0.55
            nbrs = list(G.predecessors(n)) + list(G.successors(n))
            above = [m for m in nbrs if pos[m][1] > y and abs(pos[m][0] - x) < BAND]
            below = [m for m in nbrs if pos[m][1] < y and abs(pos[m][0] - x) < BAND]
            if not above:
                texts.append((ax.text(x, y + rad + 0.10, txt, fontsize=FS, ha='center',
                                      va='bottom', color=INK, linespacing=1.15), n))
            elif not below:
                texts.append((ax.text(x, y - rad - 0.10, txt, fontsize=FS, ha='center',
                                      va='top', color=INK, linespacing=1.15), n))
            else:
                right = x >= xmid
                texts.append((ax.text(x + (rad + 0.12) * (1 if right else -1), y, txt,
                                      fontsize=FS, ha='left' if right else 'right',
                                      va='center', color=INK, linespacing=1.15), n))

    ALL_TEXTS.append((ax, texts, dict(pos), list(G.nodes), list(G.edges), list(nsz)))

    # Metrics go UNDER the drawing, not in the title: the title names the method, the
    # caption-style block below carries the numbers. The All-In / one-child ticks are
    # dropped -- the metrics plus the cluster/marginalised/edge counts are what matter.
    nm = r['meth'] + (r'  ($\lambda_{oc}=%g$)' % args.lam_oc
                      if r['meth'].lower().startswith('gradcap') else '')
    ax.set_title(nm, fontsize=9, color=INK, pad=10)
    ax.text(0.5, -0.05,
            f'RAE {r["rae"]:.2g}    ReconErr {r["recon"]:.3f}    '
            f'Intra-Cluster Edges {r["intra"]:.3f}\n'
            f'{len(r["kept"])} clusters used,  {r["marg"]} marginalised,  '
            f'{len(r["edges"])} high-level edges',
            transform=ax.transAxes, ha='center', va='top', fontsize=7, color=INK2,
            linespacing=1.35)
    ax.axis('off')
    ax.margins(x=0.40, y=0.42)

if args.title:
    fig.suptitle(args.title, fontsize=10, color=INK, y=1.0)
def _resolve_overlaps(fig, panels, pad=2.0, step=2.0, iters=80, MAXPUSH=40.0):
    """Nudge labels off nodes, edges and each other.

    Role-based placement gets the label to the right SIDE of a node, but it cannot know how
    wide the rendered text is, so a long label can still land on a marker or an edge. This
    measures the actual rendered boxes and pushes each label outward, away from its own
    node, until it is clear. Purely cosmetic: the anchor node is unchanged.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, texts, pos, nodes, edges, nsz in panels:
        if not texts:
            continue
        # node markers and edge polylines, in display coordinates
        cent = {n: np.array(ax.transData.transform(pos[n]), float) for n in nodes}
        radpx = {n: (np.sqrt(nsz[i]) / 2.0) * fig.dpi / 72.0 + pad
                 for i, n in enumerate(nodes)}
        segs = []
        for a, b in edges:
            pa, pb = cent[a], cent[b]
            segs.extend(pa + (pb - pa) * t for t in np.linspace(0.12, 0.88, 24))
        title_bb = ax.title.get_window_extent(rend).expanded(1.02, 1.15)
        start = {i: np.array(ax.transData.transform(t.get_position()), float)
                 for i, (t, _) in enumerate(texts)}
        for _ in range(iters):
            boxes = [t.get_window_extent(rend).expanded(1.02, 1.06) for t, _ in texts]
            moved = False
            for i, (t, n) in enumerate(texts):
                bb = boxes[i]
                hit = any(bb.overlaps(boxes[j]) for j in range(len(texts)) if j != i)
                if not hit:
                    hit = bb.overlaps(title_bb)         # the panel title is an obstacle too
                if not hit:
                    hit = any(bb.contains(*q) for q in segs)
                if not hit:
                    for m in nodes:
                        cx, cy = cent[m]
                        nx_, ny_ = np.clip(cx, bb.x0, bb.x1), np.clip(cy, bb.y0, bb.y1)
                        if (nx_ - cx) ** 2 + (ny_ - cy) ** 2 < radpx[m] ** 2:
                            hit = True
                            break
                if not hit:
                    continue
                anchor = cent[n]
                here = np.array([bb.x0 + bb.width / 2, bb.y0 + bb.height / 2])
                d = here - anchor
                nrm = np.linalg.norm(d)
                d = d / nrm if nrm > 1e-9 else np.array([0.0, 1.0])
                xy = np.array(ax.transData.transform(t.get_position()), float) + d * step
                # cap the travel: a label that drifts far from its node stops identifying it
                if np.linalg.norm(xy - start[i]) > MAXPUSH:
                    continue
                t.set_position(ax.transData.inverted().transform(xy))
                moved = True
            if not moved:
                break


_resolve_overlaps(fig, ALL_TEXTS)
# The key only earns its space when the colour distinguishes something: with no negative
# edge in the figure, red is the only colour and a one-entry legend says nothing.
_neg = any(w < 0 for r in res for _, _, w in r['edges'])
if _neg:
    fig.legend(handles=[Line2D([], [], color=POS, lw=1.8, label='positive high-level edge'),
                        Line2D([], [], color=NEG, lw=1.8, label='negative high-level edge')],
               loc='lower center', ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.015), handlelength=1.4, handletextpad=0.5)
fig.tight_layout(rect=[0, 0.06 if _neg else 0.0, 1, 0.94 if args.title else 1.0], w_pad=0.6)
out = args.out or f'leontief_highlevel_thr{str(args.thr).replace(".", "")}_k{K}'
fig.savefig(f'{B}/experiments/abstraction_study/figures/{out}.pdf')
fig.savefig(f'{B}/experiments/abstraction_study/figures/{out}.png', dpi=300)
print(f'wrote figures/{out}.png (+pdf)')
for r in res:
    print(f'  {r["meth"]:16s} RAE {r["rae"]:.3g}  ReconErr {r["recon"]:.3f}  '
          f'intra {r["intra"]:.3f}  k {len(r["kept"])}  marg {r["marg"]}  '
          f'edges {len(r["edges"])}  fitter {r["fm"]}')
