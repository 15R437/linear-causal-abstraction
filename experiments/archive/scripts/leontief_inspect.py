"""Deep dive on ONE Leontief threshold: what clusters are found, and what quotient results.

Prints, for each method:
  * every cluster with its GB sector names, MARGINALISED nodes excluded -- a node whose
    fitted T weight is below tau carries no weight in the abstraction, so listing it as a
    member would overstate what the cluster represents. Marginalised nodes are reported
    separately rather than silently dropped.
  * the high-level graph over the surviving clusters, EMPTY clusters excluded (a budget of
    k=20 does not mean 20 clusters are used).

DATA: EXIOBASE3 IOT_2024_ixi, region GB (Great Britain). 163 sectors + 9 factor inputs.
"""
import os, sys, json, pickle, time, warnings, argparse
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch
from partition_baselines import allin_agglom
from src.models.hail_mary import HailMary, check_conditions
from src.utils.helper import fit_abstraction_auto, get_S, cluster_indices, is_dag
from src.utils.losses import RAE, reconstruction_loss
from occ import occ, oc_closure, budget_coarsening

B = '/Users/issy/Projects/linear-causal-abstraction'
TAU, K, EPOCHS = 1e-2, 20, 3000

ap = argparse.ArgumentParser()
ap.add_argument('--thr', type=float, default=0.05)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--top', type=int, default=12, help='max member names printed per cluster')
ap.add_argument('--methods', nargs='+',
                default=['AOC Closure', 'Greedy All-In', 'gradcap-rand'])
args = ap.parse_args()

A0 = np.load(f'{B}/data/A_low.npy')
L = pickle.load(open(f'{B}/data/A_low_labels.pkl', 'rb'))
names_all = [s for (_, s) in L['sectors']] + list(L['factors'])
A = np.where(np.abs(A0) > args.thr, A0, 0.0)
live = np.where((A != 0).any(0) | (A != 0).any(1))[0]
A = A[np.ix_(live, live)]
names = [names_all[i] for i in live]
p = A.shape[0]
E = np.linalg.inv(np.eye(p) - A); su = np.ones(p)
print(f'threshold {args.thr}: p={p} (from {A0.shape[0]}), density {(A != 0).mean():.3f}, '
      f'acyclic={is_dag(A)}, mass retained {np.abs(A).sum() / np.abs(A0).sum():.3f}\n')


def report(title, Th):
    print('=' * 78)
    print(title)
    print('=' * 78)
    ai, oc, dq = check_conditions(A, cluster_indices(Th))
    Ah, Tp, fm = fit_abstraction_auto(A_low=A, T_hard=Th, E_low=E, n_restarts=5,
                                      seed=args.seed, maxiter=100, faithful=True,
                                      return_method=True)
    S = get_S(T=Tp, T_hard=Th, A_low=A)
    rae = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su, tau=TAU)
    rec = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
    marg = np.abs(Tp).max(1) <= TAU
    idx = cluster_indices(Th)
    used = [j for j, c in enumerate(idx) if len(c)]
    print(f'fitter={fm}  All-In={bool(ai)}  one-child={bool(oc)}  acyclic={bool(dq)}')
    print(f'RAE={rae:.3g}  ReconErr={rec:.3f}  |A_high|={np.linalg.norm(Ah):.3f}')
    print(f'clusters used {len(used)}/{Th.shape[1]}   marginalised nodes {int(marg.sum())}/{p}\n')

    kept = []
    for j in used:
        mem = [i for i in idx[j] if not marg[i]]
        drop = [i for i in idx[j] if marg[i]]
        if not mem:
            print(f'-- cluster {j}: ALL {len(drop)} members marginalised, carries no weight')
            continue
        kept.append(j)
        w = np.abs(Tp[mem, j])
        order = np.argsort(-w)
        mass = float(np.abs(A)[np.ix_(mem, mem)].sum() / np.abs(A).sum())
        print(f'-- cluster {j}: {len(mem)} members'
              + (f' (+{len(drop)} marginalised)' if drop else '')
              + f', intra-mass {mass:.3f}')
        for t in order[:args.top]:
            print(f'      {w[t]:7.3f}  {names[mem[t]]}')
        if len(order) > args.top:
            print(f'      ... {len(order) - args.top} more')
    print()

    # high-level graph over NON-EMPTY clusters only
    sub = Ah[np.ix_(kept, kept)]
    thr_e = 1e-8 * max(np.abs(sub).max(), 1e-30)
    edges = [(a, b, sub[a, b]) for a in range(len(kept)) for b in range(len(kept))
             if a != b and abs(sub[a, b]) > thr_e]
    print(f'high-level graph over the {len(kept)} non-empty clusters: '
          f'{len(edges)} edges, acyclic={is_dag((np.abs(sub) > thr_e).astype(float))}')
    for a, b, v in sorted(edges, key=lambda e: -abs(e[2]))[:25]:
        print(f'      c{kept[a]:<3d} -> c{kept[b]:<3d}   {v:+.4f}')
    if len(edges) > 25:
        print(f'      ... {len(edges) - 25} more edges')
    print()
    return kept


def partition(meth):
    if meth == 'AOC Closure':
        g, T, ci = occ(A, n_draws=12, seed=0, k=K)
        return T, f'closure k={ci["k"]}, repaired={ci["repaired"]}'
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
        return T, 'one-child fixed point'
    if meth == 'Greedy All-In':
        return allin_agglom(A, min(K, p)), 'agglomerative on All-In violations'
    torch.manual_seed(args.seed)
    init = 'occ_ac' if meth.endswith('occ') else 'random'
    t0 = time.time()
    m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), init_type=init)
    m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=0.,
            lambda_info=0., lambda_surj=0, verbose=False)
    return m.get_partition()[0].cpu().detach().numpy(), f'init={init}, {time.time()-t0:.0f}s'


for meth in args.methods:
    T, note = partition(meth)
    report(f'{meth}  (budget k<={K}; {note})', T)
