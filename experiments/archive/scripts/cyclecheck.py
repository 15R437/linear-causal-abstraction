"""Does All-In + one-child imply an ACYCLIC high-level graph (low-level a DAG)?

It need not: one-child makes the quotient a functional graph, and functional graphs admit
cycles. Merging can also create a cycle from an acyclic low-level graph. This measures how
often it actually happens, for closure partitions and for the baselines, and checks whether
`success = ai and oc` (the benchmark's definition) ever disagrees with `ai and oc and dag`.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.models.wrapper import hc_fast
from src.utils.generate_data import sample_low_level_graph
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)


def groups_of(T):
    return [g for g in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(g)]


def job(a):
    gt, dens, p, nh, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type=gt, seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return []
    Ab = (A != 0)
    # low-level really is a DAG?
    import numpy.linalg as la
    low_acyclic = int(np.allclose(np.diag(la.matrix_power((Ab).astype(float), 1) * 0), 0)
                      and np.isclose(np.trace(la.inv(np.eye(p) - Ab.astype(float))), p))
    out = []
    rng = np.random.default_rng(seed)
    # (a) closure WITHOUT the dag filter -- keep best valid-on-(allin,onechild) only
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        ai, oc, dag = mk.conditions(A, g)
        if ai and oc and len(g) > best:
            best, bg = len(g), g
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, ba)
    ai, oc, dag = mk.conditions(A, bg)
    out.append(dict(method='closure', allin=int(ai), one_child=int(oc), dag=int(dag),
                    success_2cond=int(ai and oc), success_3cond=int(ai and oc and dag)))
    # (b) a random partition and hc_fast at the same k, as a contrast
    K = len(bg)
    rr = np.random.default_rng(seed + 7)
    T = np.zeros((p, K)); T[np.arange(p), rr.integers(0, K, p)] = 1
    for name, TT in [('random', T)]:
        ai, oc, dag = mk.conditions(A, groups_of(TT))
        out.append(dict(method=name, allin=int(ai), one_child=int(oc), dag=int(dag),
                        success_2cond=int(ai and oc), success_3cond=int(ai and oc and dag)))
    try:
        Th = np.asarray(hc_fast(A_low=A, k=K))
        H = np.zeros_like(Th); H[np.arange(p), np.abs(Th).argmax(1)] = 1
        ai, oc, dag = mk.conditions(A, groups_of(H))
        out.append(dict(method='hc_fast', allin=int(ai), one_child=int(oc), dag=int(dag),
                        success_2cond=int(ai and oc), success_3cond=int(ai and oc and dag)))
    except Exception:
        pass
    for r in out:
        r.update(graph_type=gt, density=dens, p=p, true_nhigh=nh, seed=seed,
                 low_acyclic=low_acyclic)
    return out


if __name__ == '__main__':
    jobs = [(gt, d, p, nh, s)
            for gt, d in [('random', 0.1), ('random', 0.5), ('random', 0.9), ('one_child', 1.0)]
            for p in (50, 100) for nh in (5, 10, 20) for s in range(8) if nh <= max(5, p // 5)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/cyclecheck.csv', index=False)
    print('\n%d rows\n' % len(d))
    print('Among partitions satisfying All-In AND one-child, how many have a CYCLIC high-level graph?')
    s = d[(d.allin == 1) & (d.one_child == 1)]
    print('  n = %d,  acyclic = %.4f,  CYCLIC = %d' % (len(s), s.dag.mean(), (s.dag == 0).sum()))
    print()
    print(d.groupby('method').agg(n=('dag', 'size'), allin=('allin', 'mean'),
                                  one_child=('one_child', 'mean'),
                                  succ_2cond=('success_2cond', 'mean'),
                                  succ_3cond=('success_3cond', 'mean'),
                                  dag_given_2cond=('dag', 'mean')).round(4).to_string())
