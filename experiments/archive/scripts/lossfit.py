"""Do the loss conclusions survive the switch to fit_abstraction_alternative?

Every loss conclusion in REPORT.md was measured with `fit_abstraction` (the Massidda
propagation along the structural mask). `process_output` now calls
`fit_abstraction_alternative` instead -- slower, handles cycles, and reaches validity more
easily, typically by marginalising variables (driving their T weight to ~0).

That matters because the loss arms are scored on TWO fit-dependent quantities:

  rel_err   -- the acceptance test of the alpha-hybrid rule
  info_eff  -- retention, which counts mass incident to MARGINALISED nodes as lost

so a fitter that marginalises more is simultaneously easier to accept and scored worse on
retention. Which way the trade lands per arm is exactly what decides the recommendation.

KEY DESIGN POINT: HailMary training reads only A_low and its logits -- the fitter is not in
the loop. So each arm is trained ONCE and the resulting partition scored under BOTH fitters.
The comparison is therefore exactly paired: identical partitions, identical instances, the
fitter as the only difference. No training noise can leak into the fitter contrast.

Arms span the three live loss questions:

  lambda_graph  : graph_only, and lambda_graph in {0, 1, 10, 100} alongside info
                  (post-edge_weight-fix, rerun_fixed.csv put lambda_graph=1 marginally ahead
                  of the recommended 0, but flagged graph_only's 96.5% retention as
                  "suspicious-good ... leans entirely on fit_abstraction marginalising
                  nodes, and the marginalised counts are not in this table" -- they are here)
  utility loss  : info (intra_mass) vs recon, with and without lambda_graph=1
                  (slack_nograph_robust put info ahead at every alpha; the FINDINGS
                  ablation had said recon was essential)
  lambda_util   : lambda_info in {1, 10, 100}   (10 was optimal at every alpha)
  one_child     : info_only drops lambda_one_child, which was "load-bearing" (0/80 without)

rel_err is reported under three noise weightings, because relerr_sensitivity.csv showed the
generator's own weighting is the most favourable one and flips 3 of 4 non-exact instances.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import fit_abstraction, fit_abstraction_alternative, get_S, is_dag
from src.utils.losses import observation_loss, intervention_loss
from occ import occ, conditions

B = '/Users/issy/Projects/linear-causal-abstraction'
N_RESTARTS = 3          # matches every prior study's process_output(n_restarts=3)

#            name            lambda_graph  one_child  info  recon
ARMS = [('graph_only',    dict(lambda_graph=10.,  lambda_one_child=0.,   lambda_info=0.,   lambda_recon=0.)),
        ('info_only',     dict(lambda_graph=0.,   lambda_one_child=0.,   lambda_info=10.,  lambda_recon=0.)),
        ('oc_info',       dict(lambda_graph=0.,   lambda_one_child=100., lambda_info=10.,  lambda_recon=0.)),
        ('oc_info_g1',    dict(lambda_graph=1.,   lambda_one_child=100., lambda_info=10.,  lambda_recon=0.)),
        ('oc_info_g10',   dict(lambda_graph=10.,  lambda_one_child=100., lambda_info=10.,  lambda_recon=0.)),
        ('oc_info_g100',  dict(lambda_graph=100., lambda_one_child=100., lambda_info=10.,  lambda_recon=0.)),
        ('oc_info_u1',    dict(lambda_graph=0.,   lambda_one_child=100., lambda_info=1.,   lambda_recon=0.)),
        ('oc_info_u100',  dict(lambda_graph=0.,   lambda_one_child=100., lambda_info=100., lambda_recon=0.)),
        ('oc_recon',      dict(lambda_graph=0.,   lambda_one_child=100., lambda_info=0.,   lambda_recon=10.)),
        ('oc_recon_g1',   dict(lambda_graph=1.,   lambda_one_child=100., lambda_info=0.,   lambda_recon=10.))]


def recon_data_scores(A, T, A_high, X, su):
    """reconstruction_loss_with_data, plus the normalisations it computes and discards.

    A_hat = T @ A_high @ T_norm.T is built from the FITTED T, so this metric is
    fitter-dependent -- and it charges for marginalisation automatically: a marginalised
    variable gets an all-zero row in A_hat, so its residual is the raw signal and the rmse
    rises. That makes it an independent check on the marginalisation loophole, needing no
    tolerance threshold.

    The raw rmse is not comparable across instances (it carries the noise scale), so the
    oracle ratio rmse/rmse_oracle is the headline: 1.0 = as good as the true A_low, higher
    = worse. Weighted variants use the same three weightings as rel_err, since
    relerr_sensitivity.csv showed the generator's own weighting is the most favourable.
    """
    p = A.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        T_norm = T / np.sum(np.abs(T), axis=0)
    T_norm = np.nan_to_num(T_norm)
    A_hat = T @ A_high @ T_norm.T

    I = np.eye(p)
    resid = (I - A_hat.T) @ X.T
    resid_or = (I - A.T) @ X.T
    ms, ms_or = (resid ** 2).mean(axis=1), (resid_or ** 2).mean(axis=1)

    def w_rmse(w):
        return float(np.sqrt(np.sum(w * ms) / max(np.sum(w), 1e-30)))

    def w_ratio(w):
        return float(np.sqrt(np.sum(w * ms) / max(np.sum(w * ms_or), 1e-30)))

    iso = np.ones(p)
    inv = 1.0 / np.clip(su, 1e-12, None)
    return dict(
        recon_rmse=float(np.sqrt(ms.mean())),
        recon_rmse_oracle=float(np.sqrt(ms_or.mean())),
        recon_ratio=float(np.sqrt(ms.mean() / max(ms_or.mean(), 1e-30))),
        recon_w_gen=w_rmse(su), recon_w_iso=w_rmse(iso), recon_w_inv=w_rmse(inv),
        recon_ratio_gen=w_ratio(su), recon_ratio_iso=w_ratio(iso),
        recon_ratio_inv=w_ratio(inv),
        recon_ratio_worst=max(w_ratio(su), w_ratio(iso), w_ratio(inv)))


def fit_scores(fitter, A, T_hard, E_low, su, absA, tot, co, X=None):
    """Score one partition under one fitter. Returns the fit-dependent columns."""
    p = A.shape[0]
    t0 = time.perf_counter()
    if fitter == 'alt':
        A_high, T = fit_abstraction_alternative(A_low=A, E_low=E_low, T_hard=T_hard,
                                                n_restarts=N_RESTARTS)[:2]
    else:
        A_high, T = fit_abstraction(A_low=A, T_hard=T_hard, E_low=E_low,
                                    n_restarts=N_RESTARTS)[:2]
    secs = time.perf_counter() - t0

    k = A_high.shape[0]
    E_high = np.linalg.inv(np.eye(k) - A_high)
    S = get_S(T=T, E_low=E_low, T_hard=T_hard)
    D = S @ E_high - E_low @ T
    R = E_low @ T

    def rel(w):
        return float(np.sum(w[:, None] * D**2) / max(np.sum(w[:, None] * R**2), 1e-30))

    # info_eff: intra-cluster mass PLUS every edge incident to a marginalised node, since a
    # node with no weight in T contributes nothing to the abstraction.
    #
    # THE TWO FITTERS MARGINALISE IN DIFFERENT NUMBER SYSTEMS, so the report's exact-zero
    # test is not a fair comparison. fit_abstraction propagates t_X ~ W_Xc (I - W_cc)^-1 t_c,
    # so a member of X with no edge into the child c gets an EXACT structural zero -- this is
    # the mechanism the report calls "marginalising the offenders". The alternative optimises
    # every entry freely under a per-cluster unit-norm constraint, so the same nodes come out
    # at 1e-4..1e-2 instead of 0 and the exact test counts NONE of them, scoring the
    # alternative as losslessly retaining mass it has in fact all but discarded.
    #
    # Both fitters normalise each cluster's t-vector to unit norm, so a row's largest
    # magnitude is on a common, scale-free footing across fitters; marginalisation is
    # therefore reported at a ladder of tolerances on it, and info_eff at each.
    rowmax = np.abs(T).max(1) if T.size else np.zeros(p)
    out = {}
    for tag, tol in (('', 0.0), ('_t6', 1e-6), ('_t3', 1e-3), ('_t2', 1e-2)):
        marg = rowmax <= tol
        lost = co.astype(bool).copy()
        lost[marg, :] = True
        lost[:, marg] = True
        np.fill_diagonal(lost, False)
        out[f'marginalised{tag}'] = int(marg.sum())
        out[f'info_eff{tag}'] = float((absA * lost).sum() / tot)
    out['marg_frac'] = out['marginalised'] / p
    out['rowmax_min'] = float(rowmax.min()) if T.size else 0.0
    out['rowmax_p10'] = float(np.percentile(rowmax, 10)) if T.size else 0.0

    iso = np.ones(p)
    inv = 1.0 / np.clip(su, 1e-12, None)
    # is_dag tests A != 0 exactly, so the alternative's numerical dust (off-mask entries
    # at 1e-16..1e-19, verified against high_level_mask) fabricates cycles in an A_high
    # that is acyclic at any sane threshold. Threshold before asking.
    A_clean = np.where(np.abs(A_high) > 1e-9, A_high, 0.0)
    r_gen, r_iso, r_inv = rel(su), rel(iso), rel(inv)
    out.update(
        rel_err=r_gen, rel_err_iso=r_iso, rel_err_inv=r_inv,
        rel_err_worst=max(r_gen, r_iso, r_inv),
        obs=float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T)),
        intv=float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T, T_hard=T_hard)),
        edges=int((np.abs(A_high) > 1e-9).sum()), dag=int(is_dag(A_clean)),
        fit_secs=secs)
    if X is not None:
        out.update(recon_data_scores(A, T, A_high, X, su))
    return out


def score(A, T_hard, E_low, su, absA, tot, X=None, **extra):
    """One row per fitter for this partition. Structural columns are fitter-independent."""
    gs = [x for x in (np.where(T_hard[:, j] == 1)[0] for j in range(T_hard.shape[1])) if len(x)]
    ai, oc, dag_q = conditions(A, gs)
    co = T_hard @ T_hard.T
    np.fill_diagonal(co, 0)
    base = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag_q),
                k_used=len(gs), largest=int(max(len(x) for x in gs)),
                info=float((absA * co).sum() / tot), **extra)
    rows = []
    for fitter in ('alt', 'reg'):
        row = dict(base, fitter=fitter)
        try:
            row.update(fit_scores(fitter, A, T_hard, E_low, su, absA, tot, co, X=X))
            row['fit_failed'] = 0
        except Exception as e:
            row.update(fit_failed=1, fit_err=type(e).__name__, rel_err=np.nan,
                       rel_err_iso=np.nan, rel_err_inv=np.nan, rel_err_worst=np.nan,
                       info_eff=np.nan, info_eff_t6=np.nan, info_eff_t3=np.nan,
                       info_eff_t2=np.nan, marginalised=np.nan)
        rows.append(row)
    return rows


def job(a):
    torch.set_num_threads(1)
    p, dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)

    g0, T0, ci = occ(A, seed=seed)
    K = ci['k']
    k = int(min(2 * K, p))                     # slack 2, as in rerun_fixed.py
    out = score(A, T0, E_low, su, absA, tot, X=X, arm='occ', secs=0.0)
    for name, kw in ARMS:
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337 + seed)
        init_base = np.zeros((p, k)); init_base[:, :min(k, T0.shape[1])] = T0[:, :k]
        init = torch.from_numpy(init_base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
        T_hard = m.get_partition()[0].detach().numpy()
        out += score(A, T_hard, E_low, su, absA, tot, X=X, arm=name, secs=time.time() - t0)
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K, k_offered=k)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default='lossfit.csv')
    a = ap.parse_args()

    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.3, 0.5, 0.7, 0.9)
            for s in range(a.seeds)]
    path = f'{B}/experiments/abstraction_study/data/{a.out}'
    print(f'{len(jobs)} instances x {len(ARMS)+1} arms x 2 fitters', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(a.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            pd.DataFrame(rows).to_csv(path, index=False)
            if i % 3 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time()-t0)/60, path))
