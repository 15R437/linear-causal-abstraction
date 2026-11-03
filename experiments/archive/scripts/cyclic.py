"""THE CYCLIC CASE.

Infrastructure + the three questions that matter.

Generation: take the existing DAG generator, add `n_back` reversed edges to create cycles,
then rescale A so the spectral radius < 1 so the SEM X = AX + U has the unique equilibrium
X = (I-A)^-1 U.

Q1. Does All-In + one-child still imply an ACYCLIC quotient?
    On DAGs it does (an infinite walk is impossible in a finite DAG). That argument needs
    low-level acyclicity, so it should FAIL here. How often?
Q2. Does `fit_abstraction` actually work on cyclic low-level graphs?
    reverse_topo_order runs on the HIGH-level mask, not A_low; low-level cycles enter only
    via (I - W_cc)^-1. So it should work whenever the QUOTIENT is acyclic.
    NB reverse_topo_order has no cycle detection -- on a cyclic quotient it returns a wrong
    order silently, so we must guard.
Q3. Is All-In + one-child + acyclic-quotient sufficient in the cyclic case?
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import observation_loss, intervention_loss
from src.utils.helper import process_output
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)


def make_cyclic(p, nh, density, seed, n_back, rho=0.8, graph_type='random'):
    """DAG from the existing generator + n_back reversed edges, rescaled to rho(A) < 1."""
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=density, shuffle=True, blockiness=0.5,
        heterogeneity=1., internal_density=0., graph_type=graph_type, seed=seed,
        connected=True, max_attempts=400)
    rng = np.random.default_rng(seed + 5000)
    src, dst = np.where(A != 0)
    if len(src) == 0:
        return None
    pick = rng.choice(len(src), size=min(n_back, len(src)), replace=False)
    for e in pick:                      # add the reverse of an existing edge
        i, j = int(src[e]), int(dst[e])
        if A[j, i] == 0:
            A[j, i] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    r = np.max(np.abs(np.linalg.eigvals(A)))
    if r >= rho:
        A = A * (rho / (r + 1e-12))     # ensure the equilibrium exists
    return A


def is_cyclic(Ab):
    n = Ab.shape[0]
    R = Ab.copy()
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        R = R | (R @ R)
    return bool(np.any(np.diag(R)))


def job(a):
    p, nh, dens, n_back, seed = a
    A = make_cyclic(p, nh, dens, seed, n_back)
    if A is None:
        return None
    Ab = (A != 0)
    low_cyclic = is_cyclic(Ab)
    rng = np.random.default_rng(seed)
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        ai, oc, _ = mk.conditions(A, g)          # NOTE: ignore the dag flag on purpose
        if ai and oc and len(g) > best:
            best, bg = len(g), g
    used_fb = int(bg is None)
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, ba)
    ai, oc, dag = mk.conditions(A, bg)
    K = len(bg)
    T = np.zeros((p, K))
    for j, m in enumerate(bg):
        T[m, j] = 1
    row = dict(p=p, true_nhigh=nh, density=dens, n_back=n_back, seed=seed,
               low_cyclic=int(low_cyclic), k=K, allin=int(ai), one_child=int(oc),
               quotient_acyclic=int(dag), used_fallback=used_fb,
               spectral=float(np.max(np.abs(np.linalg.eigvals(A)))))
    # Q2/Q3: does fit_abstraction succeed? Only meaningful when the quotient is acyclic.
    if ai and oc:
        try:
            E_low = np.linalg.inv(np.eye(p) - A)
            S, T_pred, A_high = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
            E_high = np.linalg.inv(np.eye(A_high.shape[0]) - A_high)
            row['obs_loss'] = float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
            row['intv_loss'] = float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
            row['valid_abstraction'] = int(row['obs_loss'] < 1e-8 and row['intv_loss'] < 1e-8)
        except Exception as e:
            row['obs_loss'] = row['intv_loss'] = np.nan
            row['valid_abstraction'] = 0
            row['err'] = type(e).__name__
    return row


if __name__ == '__main__':
    import sys as _sys
    heavy = '--heavy' in _sys.argv
    if heavy:   # push hard into the regime where the QUOTIENT itself goes cyclic
        jobs = [(p, 10, 0.5, nb, s) for p in (50, 100)
                for nb in (50, 100, 200, 400, 800) for s in range(20)]
    else:
        jobs = [(p, 10, 0.5, nb, s) for p in (50, 100) for nb in (0, 2, 5, 10, 25, 50)
                for s in range(10)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows); d.to_csv(f'{B}/experiments/abstraction_study/data/cyclic%s.csv' % ('_heavy' if heavy else ''), index=False)
    print('\n%d instances\n' % len(d))
    print('=== Q1: does All-In + one-child still give an ACYCLIC quotient? ===')
    s = d[(d.allin == 1) & (d.one_child == 1)]
    print(s.groupby('n_back').agg(n=('quotient_acyclic', 'size'),
                                  low_cyclic=('low_cyclic', 'mean'),
                                  quotient_acyclic=('quotient_acyclic', 'mean'),
                                  k=('k', 'mean')).round(3).to_string())
    print('\n=== Q2/Q3: does an abstraction exist, split by quotient acyclicity? ===')
    if 'valid_abstraction' in s.columns:
        print(s.groupby(['n_back', 'quotient_acyclic']).agg(
            n=('valid_abstraction', 'size'), valid=('valid_abstraction', 'mean'),
            obs=('obs_loss', 'median'), intv=('intv_loss', 'median')).round(4).to_string())
        print('\nOVERALL, quotient acyclic:  P(valid abstraction) = %.4f  (n=%d)'
              % (s[s.quotient_acyclic == 1].valid_abstraction.mean(),
                 (s.quotient_acyclic == 1).sum()))
        cy = s[s.quotient_acyclic == 0]
        if len(cy):
            print('OVERALL, quotient CYCLIC :  P(valid abstraction) = %.4f  (n=%d)'
                  % (cy.valid_abstraction.mean(), len(cy)))
