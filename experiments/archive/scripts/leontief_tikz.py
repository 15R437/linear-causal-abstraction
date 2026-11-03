"""Emit the learned high-level graphs as standalone TikZ, for direct inclusion in the paper.

Matplotlib fights you on a figure this small: label placement, node/edge overlap and
whitespace all have to be tuned by hand and none of it scales with the document font. A
5-node graph is small enough to hand to TikZ instead, where positions are explicit, text uses
the paper's own font, and the whole thing stays vector.

Node areas are sqrt-scaled in the member count (linear scaling makes a 111-member cluster wide
enough to hide the short edges around it). Edges are the fitted A_high, already masked to the
quotient structure inside fit_abstraction_K, so every drawn edge has a real low-level block
behind it. Red = positive, navy = negative, matching plots.visualize_causal_graph.

Writes one tikzpicture per method to figures/leontief_tikz.tex. Requires \\usepackage{tikz}
and \\usetikzlibrary{arrows.meta, positioning}.
"""
import os, sys, re, pickle, argparse, warnings
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, networkx as nx
from partition_baselines import allin_agglom
from src.models.hail_mary import HailMary
from src.utils.helper import fit_abstraction_auto, get_S, cluster_indices, quotient_mask
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import RAE, reconstruction_loss
from occ import occ, oc_closure, budget_coarsening

B = '/Users/issy/Projects/linear-causal-abstraction'
TAU, EPOCHS = 1e-2, 3000
ap = argparse.ArgumentParser()
ap.add_argument('--thr', type=float, default=0.05)
ap.add_argument('--fitter', default='eigen', choices=['auto', 'eigen'],
                help='At thr=0.05 the quotient is acyclic apart from low-level self-loops, '
                     'so eigen is sink-optimisation plus propagation and there is no eigen '
                     'step to take; it still edges out auto (Greedy RAE 2.9e-09 vs 7.9e-08).')
ap.add_argument('--k', type=int, default=5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--lam-oc', type=float, default=0.0)
ap.add_argument('--gc-restarts', type=int, default=10)
ap.add_argument('--namelen', type=int, default=26)
ap.add_argument('--methods', nargs='+', default=['AOC Closure', 'Greedy All-In'])
ap.add_argument('--out', default='leontief_tikz')
args = ap.parse_args()
K = args.k

A0 = np.load(f'{B}/data/A_low.npy')
L = pickle.load(open(f'{B}/data/A_low_labels.pkl', 'rb'))
names_all = [s for (_, s) in L['sectors']] + list(L['factors'])
A = np.where(np.abs(A0) > args.thr, A0, 0.0)
live = np.where((A != 0).any(0) | (A != 0).any(1))[0]
A = A[np.ix_(live, live)]
names = [names_all[i] for i in live]
p = A.shape[0]
E = np.linalg.inv(np.eye(p) - A); su = np.ones(p)


def _fit(T_hard, seed):
    """Dispatch on --fitter; returns (A_high, T)."""
    if args.fitter == 'eigen':
        return fit_abstraction_eigen(A_low=A, T_hard=T_hard, seed=seed, n_restarts=5)[:2]
    return fit_abstraction_auto(A_low=A, T_hard=T_hard, E_low=E, n_restarts=5, seed=seed,
                                maxiter=100, faithful=True)[:2]
nedge = int((A != 0).sum())


def esc(s):
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def short(s, n):
    s = s.split(';')[0].split(',')[0].strip()
    s = re.sub(r'\s*\(\d+\)$', '', s).strip()
    words, out = s.split(), []
    for w in words:
        if out and len(' '.join(out + [w])) > n:
            break
        out.append(w)
    return esc(' '.join(out))


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
    best, best_recon = None, np.inf
    for sd in range(args.gc_restarts):
        torch.manual_seed(args.seed + sd)
        m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100.,
                lambda_one_child=args.lam_oc, lambda_info=0., lambda_surj=0, verbose=False)
        Th_c = m.get_partition()[0].cpu().detach().numpy()
        try:
            Ah_c, Tp_c = _fit(Th_c, 0)
            rc = reconstruction_loss(A_low=A, T=Tp_c, A_high=Ah_c)
        except Exception:
            continue
        if rc < best_recon:
            best, best_recon = Th_c, rc
    return best


def build(meth):
    Th = partition(meth)
    Ah, Tp = _fit(Th, args.seed)
    S = get_S(T=Tp, T_hard=Th, A_low=A)
    rae = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su, tau=TAU)
    rec = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
    co = Th @ Th.T; np.fill_diagonal(co, 0)
    intra = float(((A != 0) * co).sum() / nedge)
    marg = np.abs(Tp).max(1) <= TAU
    idx = cluster_indices(Th)
    kept = [j for j, c in enumerate(idx) if len(c) and not marg[c].all()]
    lab, size = {}, {}
    for j in kept:
        mem = [i for i in idx[j] if not marg[i]]
        size[j] = len(mem)
        lab[j] = short(names[mem[int(np.argmax(np.abs(Tp[mem, j])))]], args.namelen)
    sub = Ah[np.ix_(kept, kept)]
    edges = [(kept[a], kept[b], float(sub[a, b])) for a in range(len(kept))
             for b in range(len(kept)) if a != b and sub[a, b] != 0.0]
    # layered by longest-path depth, same as the matplotlib version
    G = nx.DiGraph(); G.add_nodes_from(kept)
    for a, b, _ in edges:
        G.add_edge(a, b)
    try:
        depth = {n: 0 for n in G.nodes}
        for n in nx.topological_sort(G):
            for t in G.successors(n):
                depth[t] = max(depth[t], depth[n] + 1)
    except nx.NetworkXUnfeasible:
        depth = {n: 0 for n in G.nodes}
    rows = {}
    for n, d in depth.items():
        rows.setdefault(d, []).append(n)
    pos = {}
    for d, ns in rows.items():
        ns = sorted(ns, key=lambda n: -size[n])
        for i, n in enumerate(ns):
            pos[n] = (2.5 * (i - (len(ns) - 1) / 2.0), -2.1 * d)
    return dict(meth=meth, kept=kept, lab=lab, size=size, edges=edges, pos=pos,
                rae=rae, recon=rec, intra=intra, marg=int(marg.sum()))


res = [build(m) for m in args.methods]
out = [r'% Requires: \usepackage{tikz}  \usetikzlibrary{arrows.meta}',
       r'% Colours match plots.visualize_causal_graph (red positive, navy negative).',
       r'\definecolor{hlpos}{HTML}{D62728}',
       r'\definecolor{hlneg}{HTML}{1F3B73}', '']
PAL = ['0F8B8D', 'E08214', '7B3294', '8C6D31', '7F7F7F', '17BECF', 'BCBD22']
for i, c in enumerate(PAL):
    out.append(r'\definecolor{hlc%d}{HTML}{%s}' % (i, c))
out.append('')

for r in res:
    wmax = max((abs(w) for _, _, w in r['edges']), default=1.0)
    out += [r'\begin{tikzpicture}[',
            r'    node distance=0pt,',
            r'    every node/.style={font=\scriptsize},',
            r'    cl/.style={circle, draw=white, line width=0.6pt, fill=#1, text=white,'
            r' inner sep=0pt},',
            r'    hl/.style={-{Stealth[length=1.6mm]}, line width=#1},',
            r'  ]']
    out.append(r'  %% ' + esc(r['meth']))
    for i, n in enumerate(r['kept']):
        x, y = r['pos'][n]
        # sqrt-scaled radius: linear in members makes the 111-member node hide its own edges
        rad = 0.16 + 0.055 * np.sqrt(r['size'][n])
        out.append(r'  \node[cl={hlc%d}, minimum size=%.2fcm] (n%d) at (%.2f,%.2f) {};'
                   % (i % len(PAL), 2 * rad, n, x, y))
    for n in r['kept']:
        below = any(r['pos'][m][1] < r['pos'][n][1] for m in r['kept']
                    if (n, m) in [(a, b) for a, b, _ in r['edges']])
        anchor = 'below' if not below else 'above'
        out.append(r'  \node[%s=1mm of n%d, align=center] {%s\\(%d sectors)};'
                   % (anchor, n, r['lab'][n], r['size'][n]))
    for a, b, w in r['edges']:
        col = 'hlpos' if w > 0 else 'hlneg'
        lw = 0.3 + 1.1 * abs(w) / wmax
        out.append(r'  \draw[hl=%.2fpt, %s] (n%d) -- (n%d);' % (lw, col, a, b))
    out += [r'\end{tikzpicture}',
            r'%% %s: RAE %.2g, ReconErr %.3f, Intra-Cluster Edges %.3f, '
            r'%d clusters, %d marginalised, %d edges'
            % (r['meth'], r['rae'], r['recon'], r['intra'], len(r['kept']), r['marg'],
               len(r['edges'])), '']

txt = '\n'.join(out) + '\n'
open(f'{B}/experiments/abstraction_study/figures/{args.out}.tex', 'w').write(txt)
print(txt)
print('wrote figures/%s.tex' % args.out)
