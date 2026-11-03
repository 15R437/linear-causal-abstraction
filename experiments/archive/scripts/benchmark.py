"""Head-to-head: existing baselines vs closure vs closure+HailMary.

Scored on what actually matters: exact recovery (obs & intv loss ~ 0), the two sufficient
conditions, information loss, reconstruction with data, and runtime. Every method is given
the SAME k (the closure's K) so the comparison is of the partition, not the budget.

closure here = closure + targeted-merge fallback, i.e. the complete method: it always
returns a certified-valid partition.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.models.wrapper import hc_fast, uca
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.losses import (graph_loss, observation_loss, intervention_loss,
                              reconstruction_loss, reconstruction_loss_with_data)
from src.utils.helper import process_output, reconstruct_adj
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location(
    'fb', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)
OUT = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/benchmark.csv'


def to_hard(T):
    """Column-normalised / soft T -> hard one-hot assignment."""
    p, k = T.shape
    H = np.zeros((p, k))
    H[np.arange(p), np.abs(T).argmax(1)] = 1
    return H


def score(A, T_hard, X, E_low, name, secs, seed):
    Ab = (A != 0)
    groups = [np.where(T_hard[:, j] == 1)[0] for j in range(T_hard.shape[1])]
    groups = [g for g in groups if len(g)]
    ai, oc, dag = mk.conditions(A, groups)
    co = T_hard @ T_hard.T; np.fill_diagonal(co, 0)
    row = dict(method=name, secs=secs, seed=seed, allin=int(ai), one_child=int(oc),
               success=int(ai and oc), used=len(groups),
               info=(np.abs(A) * co).sum() / np.abs(A).sum(),
               max_clust=int(T_hard.sum(0).max()),
               graph_loss=float(graph_loss(A_low=A, T_hard=T_hard)))
    try:
        S, T_pred, A_high = process_output(T_hard=T_hard, A_low=A, E_low=E_low, n_restarts=3)
        E_high = np.linalg.inv(np.eye(A_high.shape[0]) - A_high)
        A_rec = reconstruct_adj(A_high=A_high, T=T_pred)
        row['obs_loss'] = float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
        row['intv_loss'] = float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
        row['recon_loss'] = float(reconstruction_loss(A_low=A, T=T_pred, A_high=A_high, A_hat=A_rec))
        row['recon_data'] = float(reconstruction_loss_with_data(
            A_low=A, A_high=A_high, A_hat=A_rec, X=X, T=T_pred))
        row['high_edges'] = int((np.abs(A_high) > 1e-9).sum())
    except Exception as e:
        for c in ('obs_loss', 'intv_loss', 'recon_loss', 'recon_data', 'high_edges'):
            row[c] = np.nan
        row['err'] = type(e).__name__
    return row


def job(a):
    torch.set_num_threads(1)
    gt, dens, p, nh, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)
    Ab = (A != 0)
    E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed + 999)

    # --- closure (+ targeted-merge fallback): the complete method, defines k ---
    t0 = time.time()
    rng = np.random.default_rng(seed)
    best, bg, best_any, bany = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > best_any:
            best_any, bany = len(g), g
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, bany)
    t_clo = time.time() - t0
    K = len(bg)
    T_clo = np.zeros((p, K))
    for j, m in enumerate(bg):
        T_clo[m, j] = 1

    out = [score(A, T_clo, X, E_low, 'closure', t_clo, seed)]

    # --- closure -> HailMary (keep only if still valid and lower info) ---
    t0 = time.time()
    gen = torch.Generator().manual_seed(31337 + seed)
    li = torch.from_numpy(T_clo).float() * 5.0 + torch.randn(p, K, generator=gen) * 0.1
    m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
    m.train(num_epochs=6000, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T_hm = m.get_partition()[0].detach().numpy()
    gh = [np.where(T_hm[:, j] == 1)[0] for j in range(K)]
    gh = [g for g in gh if len(g)]
    ai, oc, _ = mk.conditions(A, gh)
    coh = T_hm @ T_hm.T; np.fill_diagonal(coh, 0)
    co0 = T_clo @ T_clo.T; np.fill_diagonal(co0, 0)
    keep = ai and oc and ((np.abs(A) * coh).sum() <= (np.abs(A) * co0).sum())
    t_hm = t_clo + time.time() - t0
    out.append(score(A, T_hm if keep else T_clo, X, E_low, 'closure+HM', t_hm, seed))

    # --- baselines, same k ---
    for name, fn in [('hc_fast', hc_fast), ('uca', uca)]:
        t0 = time.time()
        try:
            T = to_hard(np.asarray(fn(A_low=A, k=K)))
            out.append(score(A, T, X, E_low, name, time.time() - t0, seed))
        except Exception as e:
            out.append(dict(method=name, seed=seed, secs=time.time() - t0, err=type(e).__name__))
    rr = np.random.default_rng(seed + 7)
    T = np.zeros((p, K)); T[np.arange(p), rr.integers(0, K, p)] = 1
    out.append(score(A, T, X, E_low, 'random', 0.0, seed))

    for r in out:
        r.update(graph_type=gt, p=p, true_nhigh=nh, k=K, density=dens)
    return out


if __name__ == '__main__':
    jobs = [(gt, d, p, 10, s) for gt, d in [('random', 0.5), ('one_child', 1.0)]
            for p in (50, 100) for s in range(8)]
    print(f'{len(jobs)} instances x 5 methods', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 4 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
    d['exact'] = ((d.obs_loss < 1e-8) & (d.intv_loss < 1e-8)).astype(float)
    print('\n%d rows\n' % len(d))
    print(d.groupby(['graph_type', 'method']).agg(
        n=('success', 'size'), success=('success', 'mean'), exact=('exact', 'mean'),
        info=('info', 'mean'), recon_data=('recon_data', 'median'),
        used=('used', 'mean'), secs=('secs', 'mean')).round(3).to_string())
