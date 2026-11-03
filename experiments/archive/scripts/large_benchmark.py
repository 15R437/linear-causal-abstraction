"""Large-scale benchmark: p x density x n_high x family, all methods.

Three tiers, selected by --tier:
  main    p in {50,100}   all 5 methods            (HailMary is the cost)
  scale   p in {200,400}  closure / hc_fast / random only (HailMary dropped)
  shape   p=100 d=0.5     blockiness x heterogeneity, closure only

Scored on exact recovery (obs & intv loss < 1e-8), the two sufficient conditions,
information loss, reconstruction with data, and runtime. Every method gets the SAME k --
the closure's K -- so the comparison is of the partition, not the budget.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.models.wrapper import hc_fast, uca
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.losses import (graph_loss, observation_loss, intervention_loss,
                              reconstruction_loss, reconstruction_loss_with_data)
from src.utils.helper import process_output, reconstruct_adj
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)


def to_hard(T):
    p, k = T.shape
    H = np.zeros((p, k)); H[np.arange(p), np.abs(T).argmax(1)] = 1
    return H


def score(A, T_hard, X, E_low, name, secs, full=True):
    groups = [np.where(T_hard[:, j] == 1)[0] for j in range(T_hard.shape[1])]
    groups = [g for g in groups if len(g)]
    ai, oc, dag = mk.conditions(A, groups)
    co = T_hard @ T_hard.T; np.fill_diagonal(co, 0)
    row = dict(method=name, secs=secs, allin=int(ai), one_child=int(oc),
               success=int(ai and oc), used=len(groups),
               info=(np.abs(A) * co).sum() / np.abs(A).sum(),
               max_clust=int(T_hard.sum(0).max()))
    if not full:
        return row
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
    gt, dens, p, nh, seed, blk, het, tier = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=blk,
            heterogeneity=het, internal_density=0., graph_type=gt, seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return []
    Ab = (A != 0)
    full = tier != 'shape'
    E_low = np.linalg.inv(np.eye(p) - A) if full else None
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8,
                                             seed=seed + 999) if full else None

    # closure (+ targeted-merge fallback) -- defines k
    t0 = time.time()
    rng = np.random.default_rng(seed)
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    used_fallback = int(bg is None)
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, ba)
    t_clo = time.time() - t0
    K = len(bg)
    T_clo = np.zeros((p, K))
    for j, m in enumerate(bg):
        T_clo[m, j] = 1
    out = [score(A, T_clo, X, E_low, 'closure', t_clo, full)]
    out[0]['used_fallback'] = used_fallback
    out[0]['closure_raw_ok'] = 1 - used_fallback

    if tier == 'main':
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
        out.append(score(A, T_hm if keep else T_clo, X, E_low, 'closure+HM',
                         t_clo + time.time() - t0, full))

    if tier in ('main', 'scale'):
        for name, fn in [('hc_fast', hc_fast)] + ([('uca', uca)] if tier == 'main' else []):
            t0 = time.time()
            try:
                out.append(score(A, to_hard(np.asarray(fn(A_low=A, k=K))), X, E_low,
                                 name, time.time() - t0, full))
            except Exception as e:
                out.append(dict(method=name, secs=time.time() - t0, err=type(e).__name__))
        rr = np.random.default_rng(seed + 7)
        T = np.zeros((p, K)); T[np.arange(p), rr.integers(0, K, p)] = 1
        out.append(score(A, T, X, E_low, 'random', 0.0, full))

    for r in out:
        r.update(graph_type=gt, p=p, density=dens, true_nhigh=nh, seed=seed,
                 blockiness=blk, heterogeneity=het, k=K, tier=tier)
    return out


def build(tier):
    if tier == 'main':
        return [(gt, d, p, nh, s, 0.5, 1.0, tier)
                for gt, d in [('random', 0.1), ('random', 0.3), ('random', 0.5),
                              ('random', 0.7), ('random', 0.9), ('one_child', 1.0)]
                for p in (50, 100) for nh in (5, 10, 20) for s in range(8)
                if nh <= max(5, p // 5)]
    if tier == 'scale':
        return [(gt, d, p, nh, s, 0.5, 1.0, tier)
                for gt, d in [('random', 0.1), ('random', 0.5), ('random', 0.9), ('one_child', 1.0)]
                for p in (200, 400) for nh in (10, 20, 40) for s in range(5)]
    return [(gt, d, 100, 10, s, blk, het, tier)
            for gt, d in [('random', 0.5), ('one_child', 1.0)]
            for blk in (0.0, 0.5, 1.0) for het in (0.5, 1.0, 2.0) for s in range(8)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tier', default='main')
    ap.add_argument('--workers', type=int, default=6)
    t = ap.parse_args().tier
    OUT = f'{B}/experiments/abstraction_study/data/large_{t}.csv'
    jobs = build(t)
    print(f'tier={t}  {len(jobs)} instances', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(ap.parse_args().workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 20 == 0:
                el = (time.time() - t0) / 60
                print(f'{i}/{len(jobs)}  {el:.1f}min  eta {el/i*(len(jobs)-i):.0f}min', flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'done {(time.time()-t0)/60:.1f} min -> {OUT}')
