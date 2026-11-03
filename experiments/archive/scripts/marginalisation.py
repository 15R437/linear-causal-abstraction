"""Are All-In violations tolerable if fit_abstraction marginalises the offenders?

The two sufficient conditions are NOT necessary: P(valid abstraction | conditions fail) is
0.267 on random and 0.758 on one_child. fit_abstraction zeroes out nodes it cannot fit
(measured: 80/100 on a random partition), so a partition can violate All-In and still yield
an EXACT abstraction over the surviving nodes -- just a less informative one.

Effective information loss must therefore charge for BOTH:
  * edges inside a cluster (destroyed by the partition), and
  * edges incident to a MARGINALISED node (destroyed by the fit).

Arms (all with lambda_graph=10):
  as_is    oc=100, info=10          -- the shipped configuration
  recon    oc=100, recon=10, info=0 -- original recon loss instead of intra_mass
  no_oc    oc=0,   info=10          -- one-child off
  closure  (reference, no training)
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import observation_loss, intervention_loss
from src.utils.helper import process_output
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)

ARMS = {'as_is': dict(lambda_graph=10., lambda_one_child=100., lambda_info=10., lambda_recon=0.),
        'recon': dict(lambda_graph=10., lambda_one_child=100., lambda_info=0., lambda_recon=10.),
        'no_oc': dict(lambda_graph=10., lambda_one_child=0., lambda_info=10., lambda_recon=0.)}


def score(A, T, E_low, absA, tot, name, secs, seed):
    p = A.shape[0]
    groups = [g for g in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(g)]
    ai, oc, dag = mk.conditions(A, groups)
    co = T @ T.T; np.fill_diagonal(co, 0)
    # violating nodes under All-In
    Ab = (A != 0)
    M = np.zeros((len(groups), len(groups)), bool)
    for x in range(len(groups)):
        for y in range(len(groups)):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    bad = set()
    for x in range(len(groups)):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    bad.add(int(i))
    row = dict(method=name, secs=secs, seed=seed, allin=int(ai), one_child=int(oc),
               dag=int(dag), success=int(ai and oc and dag), k_used=len(groups),
               viol_nodes=len(bad), info_intra=(absA * co).sum() / tot)
    try:
        S, T_pred, A_high = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
        E_high = np.linalg.inv(np.eye(A_high.shape[0]) - A_high)
        obs = float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
        intv = float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
        marg = (np.abs(T_pred).sum(1) == 0)
        # effective loss: intra-cluster mass PLUS mass incident to a marginalised node
        lost = co.copy().astype(bool)
        lost[marg, :] = True
        lost[:, marg] = True
        np.fill_diagonal(lost, False)
        row.update(obs_loss=obs, intv_loss=intv,
                   exact=int(obs < 1e-8 and intv < 1e-8),
                   marginalised=int(marg.sum()), marg_frac=float(marg.mean()),
                   info_effective=float((absA * lost).sum() / tot),
                   surviving=int(p - marg.sum()))
    except Exception as e:
        row.update(obs_loss=np.nan, intv_loss=np.nan, exact=0, marginalised=np.nan,
                   marg_frac=np.nan, info_effective=np.nan, surviving=np.nan,
                   err=type(e).__name__)
    return row


def job(a):
    torch.set_num_threads(1)
    gt, dens, p, nh, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type=gt, seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return []          # some sparse cells admit no stable connected model
    Ab = (A != 0); absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(p) - A)
    rng = np.random.default_rng(seed)
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, ba)
    K = len(bg)
    T0 = np.zeros((p, K))
    for j, m in enumerate(bg):
        T0[m, j] = 1
    out = [score(A, T0, E_low, absA, tot, 'closure', 0.0, seed)]
    for name, kw in ARMS.items():
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337 + seed)
        li = torch.from_numpy(T0).float() * 5.0 + torch.randn(p, K, generator=gen) * 0.1
        m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
        m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
        T = m.get_partition()[0].detach().numpy()
        out.append(score(A, T, E_low, absA, tot, name, time.time() - t0, seed))
    for r in out:
        r.update(graph_type=gt, density=dens, p=p, true_nhigh=nh, k=K)
    return out


if __name__ == '__main__':
    jobs = [(gt, d, 100, 10, s)
            for gt, d in [('random', 0.1), ('random', 0.5), ('random', 0.9), ('one_child', 1.0)]
            for s in range(12)]
    print(f'{len(jobs)} instances x 4 arms', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/marginalisation.csv', index=False)
    print('%d rows\n' % len(d))
    order = ['closure', 'as_is', 'recon', 'no_oc']
    print('=== the headline: does an EXACT abstraction survive, and at what effective cost? ===')
    print(d.groupby(['graph_type', 'density', 'method']).agg(
        n=('exact', 'size'), allin=('allin', 'mean'), onechild=('one_child', 'mean'),
        exact=('exact', 'mean'), viol=('viol_nodes', 'mean'),
        marg=('marginalised', 'mean'), surviving=('surviving', 'mean'),
        info_intra=('info_intra', 'mean'), info_eff=('info_effective', 'mean'),
        k=('k_used', 'mean')).reindex(order, level=2).round(3).to_string())
    print('\n=== among runs that VIOLATE All-In: is the abstraction still exact? ===')
    v = d[d.allin == 0]
    if len(v):
        print(v.groupby(['graph_type', 'method']).agg(
            n=('exact', 'size'), exact=('exact', 'mean'), viol=('viol_nodes', 'mean'),
            marg=('marginalised', 'mean'), info_intra=('info_intra', 'mean'),
            info_eff=('info_effective', 'mean')).round(3).to_string())
