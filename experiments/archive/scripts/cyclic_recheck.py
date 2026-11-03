"""Re-run the cyclic.py Q2/Q3 cell with the CORRECTED S.

`cyclic.py` scored validity through `process_output` -> `get_S`, which is wrong whenever the
quotient is cyclic -- exactly the cell it was measuring. This re-runs the same instances and
the same partitions, scoring each fit under both S definitions and under all three fitters.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings, time
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp, torch
from src.utils.helper import (fit_abstraction, fit_abstraction_alternative, fit_abstraction_K,
                              get_S, cluster_indices)
from src.utils.losses import observation_loss, intervention_loss

B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('cy', f'{B}/experiments/abstraction_study/scripts/cyclic.py')
cy = importlib.util.module_from_spec(_s); _s.loader.exec_module(cy)


def get_S_M(T, E_low, T_hard):
    """S_i = e_ii^-1 E_low[pi_i, pi_i] t_i, with e_ii read off diag((D^-1 E_high)^-1)."""
    idx = cluster_indices(T_hard); k = T.shape[1]
    U = np.zeros_like(T)
    for j in range(k):
        if len(idx[j]):
            U[:, j] = E_low[:, idx[j]] @ T[idx[j], j]
    M = np.eye(k)
    for i in range(k):
        if not len(idx[i]):
            continue
        ui = U[idx[i], i]; den = ui @ ui
        if den > 1e-30:
            M[i, :] = (ui @ U[idx[i], :]) / den
    d = np.diag(np.linalg.inv(M)).copy()
    S = np.zeros_like(T)
    for i in range(k):
        if len(idx[i]):
            S[idx[i], i] = U[idx[i], i] / d[i]
    return S


def job(a):
    torch.set_num_threads(1)
    p, nh, dens, n_back, seed = a
    A = cy.make_cyclic(p, nh, dens, seed, n_back)
    if A is None:
        return None
    rng = np.random.default_rng(seed)
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = cy.mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        ai, oc, _ = cy.mk.conditions(A, g)
        if ai and oc and len(g) > best:
            best, bg = len(g), g
    if bg is None:
        bg, _ = cy.fb.targeted_merge((A != 0), ba)
    ai, oc, dag = cy.mk.conditions(A, bg)
    if not (ai and oc):
        return None
    K = len(bg)
    T_hard = np.zeros((p, K))
    for j, m in enumerate(bg):
        T_hard[m, j] = 1
    E_low = np.linalg.inv(np.eye(p) - A)
    row = dict(p=p, n_back=n_back, seed=seed, k=K, quotient_acyclic=int(dag),
               low_cyclic=int(cy.is_cyclic(A != 0)))
    for name, fn in (('reg', lambda: fit_abstraction(A_low=A, T_hard=T_hard, E_low=E_low, n_restarts=3)),
                     ('alt', lambda: fit_abstraction_alternative(A_low=A, E_low=E_low, T_hard=T_hard, n_restarts=3)),
                     ('K',   lambda: fit_abstraction_K(A_low=A, E_low=E_low, T_hard=T_hard, n_restarts=3))):
        t0 = time.perf_counter()
        try:
            A_high, T = fn()
            row[f'{name}_time'] = time.perf_counter() - t0
            E_high = np.linalg.inv(np.eye(A_high.shape[0]) - A_high)
            for tag, S in (('repo', get_S(T=T, E_low=E_low, T_hard=T_hard)),
                           ('corr', get_S_M(T, E_low, T_hard))):
                o = float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T))
                i_ = float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T))
                row[f'{name}_valid_{tag}'] = int(o < 1e-8 and i_ < 1e-8)
                row[f'{name}_obs_{tag}'] = o
        except Exception as e:
            row[f'{name}_err'] = type(e).__name__
            for tag in ('repo', 'corr'):
                row[f'{name}_valid_{tag}'] = 0
    return row


if __name__ == '__main__':
    jobs = [(p, 10, 0.5, nb, s) for p in (50, 100)
            for nb in (50, 100, 200, 400, 800) for s in range(10)]
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/cyclic_recheck.csv', index=False)
    print(f'{len(d)} instances in {(time.time()-t0)/60:.1f} min\n')
    pd.set_option('display.width', 250)
    print('=== P(valid abstraction), by quotient acyclicity, repo get_S vs corrected S ===')
    cols = [f'{m}_valid_{t}' for m in ('reg', 'alt', 'K') for t in ('repo', 'corr')]
    print(d.groupby('quotient_acyclic')[cols].agg(['mean', 'size']).round(3).to_string())
