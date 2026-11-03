"""How do the partition LEARNERS compare as partition learners, and along which axis?

Measures the three structural conditions directly on (A_low, T_hard) -- no fitter involved,
so these are properties of the PARTITION, not of any downstream abstraction:

    allin_pairs   sum over quotient edges x->y of #members of x with no edge into y
    allin_nodes   #low-level nodes involved in at least one such violation
    oc_excess     sum over clusters of max(#child clusters - 1, 0)
    intra         fraction of |A| edge MASS falling inside clusters
    intra_edges   fraction of A's nonzero EDGES falling inside clusters (unweighted count)

`intra` and `intra_edges` answer different questions: mass says how much causal influence is
hidden, count says how many mechanisms are. They come apart when a partition swallows many
weak edges or few strong ones, so both are recorded.

FULL FACTORIAL over blockiness x density, same number of levels each, so the two candidate
axes of variation can be compared on equal terms. That comparison is the point: which axis
actually separates the methods, and does either reorder them.

THE k BUDGET IS LOAD-BEARING (partition_baselines.py). Singletons satisfy All-In vacuously
AND have zero intra mass, so the finest partition trivially optimises both, and every method
must be held to the same budget.

The budget is a CEILING, not an exact target: a method whose natural partition is wider is
coarsened down to it, and one that is narrower simply does not spend it (`k_used` records
what was actually used, `budget_met` whether the ceiling was reachable). That is what makes a
FIXED budget feasible at every density -- an exact-k rule is not, because the one-child
closure's fixed point gets coarse as density rises (median k 47 at d=0.3 but 13 at d=0.9), so
demanding exactly k=20 would silently drop 58%% of instances at d=0.6 and 80%% at d=0.9, and
the survivors would be the atypically fine ones.

Without --budget, the ceiling is AOC Closure's own closure width, which is the finest
one-child-valid resolution the instance admits. That is principled but varies 2-53 across
instances, so any pooled median mixes regimes; pass --budget for a fixed one.

THREE CONDITIONS, NOT TWO. A 2-D (violations, intra) view is not enough: Greedy All-In sits at
exactly (0, 0.000) -- it looks like a free win -- while running one-child excess up to ~69.
Any display of this data has to carry `oc_excess` or it actively misleads.

A downstream arm (--downstream, ON by default) answers "does low violation translate?" using
`fit_abstraction_auto` at `faithful=False`, which routes acyclic quotients to the propagation
solver and cyclic ones to `_K`. `faithful=False` is deliberate: the faithful variants cost
2-6x more and this arm is supplementary to the partition metrics, which need no fitter at all.
The chosen solver is recorded per row as `fit_method`, so it can be cross-checked against
`qcyc` (the two should agree by construction).

READ `recon` WITH CARE at faithful=False. An unfaithful fit attenuates A_high while still
solving the abstraction equation, so it can reconstruct even the GROUND-TRUTH partition badly
(ALPHA_RECON_K.md). `rae`, `marg` and `norm_Ahigh` are the informative downstream columns.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import fit_abstraction_auto, get_S, is_dag
from src.algos.hail_mary import HailMary
from occ import occ, oc_closure, budget_coarsening
from partition_baselines import allin_agglom, violations
from frontier_fig import rae

P, EPOCHS = 100, 3000
BLKS = [0.2, 0.4, 0.7, 1.0]
DENS = [0.1, 0.3, 0.6, 0.9]
LAMBDAS = [0., 100.]
TAU = 1e-2


def job(a):
    torch.set_num_threads(1)
    dens, seed, blk, DOWNSTREAM, LAMS, SKIP_BASE, BUDGET = a
    try:
        A, T_gt, _, _ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=blk, heterogeneity=0., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=400)
    except Exception:
        return [], {}
    absA = np.abs(A); tot = absA.sum(); n_edges = int((A != 0).sum())
    E = np.linalg.inv(np.eye(P) - A); nA = (A ** 2).sum(); su = np.ones(P)
    if BUDGET is not None:
        K = int(BUDGET)
    else:
        K = max(occ(A, seed=seed)[2]['k'], 2)
    com = dict(density=dens, seed=seed, blk=blk, budget_k=K)

    def measure(nm, lam, Th, secs, downstream=DOWNSTREAM):
        v = violations(A, Th)
        co = Th @ Th.T; np.fill_diagonal(co, 0)
        kq = Th.shape[1]
        # normalisers, so counts are comparable across densities (which change how many
        # quotient edges exist) -- max_pairs = sum over quotient edges x->y of |X_x|
        lab = Th.argmax(1); sizes = np.bincount(lab, minlength=kq)
        Ab = (A != 0)
        Mq = np.zeros((kq, kq), bool)
        for x in range(kq):
            gx = np.where(lab == x)[0]
            if len(gx):
                reach = Ab[gx].any(0)
                for y in range(kq):
                    if x != y and len(np.where(lab == y)[0]) and reach[lab == y].any():
                        Mq[x, y] = True
        max_pairs = int((Mq.sum(1) * sizes).sum())
        r = dict(method=nm, lam_oc=lam, secs=secs, n_qedges=int(Mq.sum()),
                 max_pairs=max_pairs, intra=float((absA * co).sum() / tot),
                 intra_edges=float((Ab * co).sum() / max(n_edges, 1)),
                 allin_frac=v['allin_pairs'] / max(max_pairs, 1),
                 allin_node_frac=v['allin_nodes'] / P,
                 qcyc=int(not is_dag(Mq.astype(float))), **v, **com)
        if downstream:
            try:
                Ah, Tp, fm = fit_abstraction_auto(A_low=A, E_low=E, T_hard=Th,
                                                  n_restarts=5, seed=seed,
                                                  faithful=False, return_method=True)
                r['fit_method'] = fm
                S = get_S(T=Tp, T_hard=Th, A_low=A)
                r['recon'] = float(((A - Tp @ Ah @ np.linalg.pinv(Tp)) ** 2).sum() / nA)
                r['rae'] = rae(A, Th, Tp, Ah, S, E, su)[0]
                r['marg'] = int((np.abs(Tp).max(1) <= TAU).sum())
                r['norm_Ahigh'] = float(np.linalg.norm(Ah))
            except Exception as e:
                r['recon'] = r['rae'] = np.nan; r['err'] = type(e).__name__
        return r

    saved = {}

    def keep(nm, lam, Th):
        saved['%g|%d|%g|%s|%s' % (dens, seed, blk, nm, lam)] = \
            Th.argmax(1).astype(np.int16)

    Th_gt = (T_gt != 0).astype(float)
    keep('GT partition', np.nan, Th_gt)
    out = []
    if not SKIP_BASE:          # spawn workers re-import, so this MUST be in the job tuple
        out.append(measure('GT partition', np.nan, Th_gt, 0.))
        def _oc_budgeted():
            g, T, _ = oc_closure(A, seed=seed)
            if BUDGET is None or len(g) <= K:
                return T
            try:                      # one-child only, so All-In-preserving merges may not exist
                g2, _ = budget_coarsening(A, g, K, rng=np.random.default_rng(seed))
                T2 = np.zeros((P, len(g2)))
                for j, mem in enumerate(g2):
                    T2[np.asarray(mem), j] = 1.0
                return T2
            except Exception:
                return T

        for nm, fn in (('AOC Closure', lambda: occ(A, seed=seed, k=K)[1]),
                       ('OC Closure', _oc_budgeted),
                       ('Greedy All-In', lambda: allin_agglom(A, min(K, P)))):
            t0 = time.time(); Th = fn(); keep(nm, np.nan, Th)
            out.append(measure(nm, np.nan, Th, time.time() - t0))

    for lam in LAMS:
        t0 = time.time()
        torch.manual_seed(31337 + seed)
        m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
                lambda_info=0., lambda_surj=0, verbose=False)
        Th = m.get_partition()[0].detach().numpy(); keep('GradCAP', lam, Th)
        out.append(measure('GradCAP', lam, Th, time.time() - t0))
    return out, saved


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default='partition_quality')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--downstream', type=int, default=0,
                    help='also fit each partition (adds ~50%% runtime; partitions are saved '
                         'either way, so this can be done later without retraining)')
    ap.add_argument('--seed0', type=int, default=0, help='first seed, to extend a run')
    ap.add_argument('--dens', type=float, nargs='+', default=None,
                    help='override the density grid (e.g. --dens 0.5 to match the fitter table)')
    ap.add_argument('--lambdas', type=float, nargs='+', default=None,
                    help='override the lambda_one_child grid; the non-GradCAP baselines are '
                         'deterministic given the seed, so a single-lambda run can be merged '
                         'into an existing sweep by keeping only its GradCAP rows')
    ap.add_argument('--blks', type=float, nargs='+', default=None,
                    help='override the blockiness grid')
    ap.add_argument('--budget', type=int, default=None,
                    help='fixed cluster-count CEILING for every method (e.g. 20). Default: '
                         "AOC Closure's own closure width, which varies 2-53 by instance.")
    ap.add_argument('--gradcap-only', action='store_true',
                    help='skip the deterministic baselines entirely (for such a merge)')
    args = ap.parse_args()
    dens = args.dens if args.dens else DENS
    lams = args.lambdas if args.lambdas else LAMBDAS
    blks = args.blks if args.blks else BLKS
    jobs = [(d, i, b, bool(args.downstream), lams, bool(args.gradcap_only), args.budget)
            for i in range(args.seed0, args.seed0 + args.n) for d in dens for b in blks]
    print('%d jobs = %d seeds x %d dens %s x %d blk; budget=%s; downstream=%d'
          % (len(jobs), args.n, len(dens), dens, len(blks),
             args.budget if args.budget else 'AOC width', args.downstream), flush=True)
    t0 = time.time(); rows = []; labels = {}
    base = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/%s' % args.out
    path = base + '.csv'
    with mp.Pool(args.procs) as pool:
        for i, (rs, lb) in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); labels.update(lb)
            pd.DataFrame(rows).to_csv(path, index=False)
            np.savez_compressed(base + '_partitions.npz', **labels)
            el = (time.time() - t0) / 60
            print('%d/%d %.1fmin (eta %.0fmin)' % (i, len(jobs), el, el / i * (len(jobs) - i)),
                  flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
