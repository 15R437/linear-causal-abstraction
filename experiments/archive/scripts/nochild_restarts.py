"""Can we drop one_child and just RESTART until we learn a partition that admits an abstraction?

The sufficiency theorem (All-In + one-child + acyclic quotient) is a certificate available
A PRIORI: satisfy the conditions and an abstraction must exist. But the conditions are only
sufficient, and verifying an abstraction directly is cheap -- fit it and measure the residual,
O(p^3), no theorem required. So a different strategy is available: drop lambda_one_child,
restart, and keep any partition whose FITTED residual clears the tolerance. The certificate
becomes a posteriori and per-instance, which is already how the report's alpha-hybrid rule
accepts refinements.

Prior evidence says this fails: "one-child is what makes All-In satisfiable -- 0/80 without
it at any weight", and no_oc scored 0.000 exact on random DAGs in every cell. But all of
that was measured with fit_abstraction and scored All-In, not "admits an abstraction".

THE TRAP THIS SCRIPT IS BUILT AROUND. Without one-child there is nothing stopping the
optimiser from leaving nodes spread over many clusters -- intra-cluster mass ~0, so `info`
reports near-perfect retention -- after which the fitter "succeeds" by handing most nodes a
near-zero weight. What you get is an exact abstraction over a handful of SURVIVING
variables, which is not a useful abstraction of the original system. The report's exact-zero
marginalisation test cannot see this under fit_abstraction_alternative, which never returns
a hard zero. So every retention number here is reported alongside the survivor count, and
info_eff is computed at a tolerance ladder rather than at exact zero.

Arms, all with lambda_graph and lambda_info as noted and NO one-child unless stated:
  oc_info      lambda_one_child=100, info=10   -- the control (current recommendation)
  no_oc_info   one_child=0, info=10
  no_oc_graph  one_child=0, graph=10           -- graph_loss alone, the "suspicious-good" arm
  no_oc_g_info one_child=0, graph=10, info=10
  no_oc_g_recon one_child=0, graph=10, recon=10  -- graph + recon, the arm to restart if you
                also care about retention; recon_loss is co_assign-weighted over PARENT
                profiles, so like graph_loss it separates rather than merges

WHAT DECIDES VIABILITY. P(exact | All-In holds) = 0.73 for graph_only, but that is a
CONDITIONAL rate; unconditionally the arm lands All-In AND exact on ~27% of single draws. So
8 restarts give ~92% if the draws are independent, and 27% forever if success is a property
of the instance rather than the seed. Section 2 of analyse_nochild.py splits the per-draw
rate into never/mixed/always instances, which is the only thing that separates those cases.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
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
N_RESTARTS_FIT = 3
R = 8                      # HailMary restarts per (instance, arm)
NOISES = [0.1, 1.0]        # init diversity: half the restarts at each

ARMS = [('oc_info',      dict(lambda_graph=0.,  lambda_one_child=100., lambda_info=10., lambda_recon=0.)),
        ('no_oc_info',   dict(lambda_graph=0.,  lambda_one_child=0.,   lambda_info=10., lambda_recon=0.)),
        ('no_oc_graph',  dict(lambda_graph=10., lambda_one_child=0.,   lambda_info=0.,  lambda_recon=0.)),
        ('no_oc_g_info', dict(lambda_graph=10., lambda_one_child=0.,   lambda_info=10., lambda_recon=0.)),
        ('no_oc_g_recon',dict(lambda_graph=10., lambda_one_child=0.,   lambda_info=0.,  lambda_recon=10.))]


def score_fit(fitter, A, T_hard, E_low, su, absA, tot, co):
    p = A.shape[0]
    if fitter == 'alt':
        A_high, T = fit_abstraction_alternative(A_low=A, E_low=E_low, T_hard=T_hard,
                                                n_restarts=N_RESTARTS_FIT)[:2]
    else:
        A_high, T = fit_abstraction(A_low=A, T_hard=T_hard, E_low=E_low,
                                    n_restarts=N_RESTARTS_FIT)[:2]
    k = A_high.shape[0]
    E_high = np.linalg.inv(np.eye(k) - A_high)
    S = get_S(T=T, E_low=E_low, T_hard=T_hard)
    D = S @ E_high - E_low @ T; R_ = E_low @ T

    def rel(w):
        return float(np.sum(w[:, None]*D**2) / max(np.sum(w[:, None]*R_**2), 1e-30))

    rowmax = np.abs(T).max(1)
    out = {}
    for tag, tol in (('', 0.0), ('_t3', 1e-3), ('_t2', 1e-2)):
        marg = rowmax <= tol
        lost = co.astype(bool).copy(); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        out[f'marg{tag}'] = int(marg.sum())
        out[f'info_eff{tag}'] = float((absA * lost).sum() / tot)
    # The honest denominator for "is this an abstraction OF the system": how many low-level
    # variables still carry weight, and how many high-level variables they map onto.
    out['survivors'] = int((rowmax > 1e-2).sum())
    out['survivor_frac'] = out['survivors'] / p
    out['k_alive'] = int((np.abs(T).max(0) > 1e-2).sum())
    # is_dag tests A != 0 exactly, so the alternative's numerical dust (off-mask entries
    # at 1e-16..1e-19, verified against high_level_mask) fabricates cycles in an A_high
    # that is acyclic at any sane threshold. Threshold before asking.
    A_clean = np.where(np.abs(A_high) > 1e-9, A_high, 0.0)
    r_gen = rel(su)
    out.update(rel_err=r_gen, rel_err_iso=rel(np.ones(p)),
               rel_err_worst=max(r_gen, rel(np.ones(p)), rel(1.0/np.clip(su, 1e-12, None))),
               obs=float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T)),
               intv=float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T, T_hard=T_hard)),
               edges=int((np.abs(A_high) > 1e-9).sum()), dag=int(is_dag(A_clean)))
    return out


def rows_for(A, T_hard, E_low, su, absA, tot, **extra):
    gs = [x for x in (np.where(T_hard[:, j] == 1)[0] for j in range(T_hard.shape[1])) if len(x)]
    ai, oc, dq = conditions(A, gs)
    co = T_hard @ T_hard.T; np.fill_diagonal(co, 0)
    base = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dq),
                k_used=len(gs), largest=int(max(len(x) for x in gs)),
                info=float((absA * co).sum() / tot), **extra)
    out = []
    for f in ('alt', 'reg'):
        row = dict(base, fitter=f)
        try:
            row.update(score_fit(f, A, T_hard, E_low, su, absA, tot, co)); row['fit_failed'] = 0
        except Exception as e:
            row.update(fit_failed=1, fit_err=type(e).__name__, rel_err=np.nan,
                       rel_err_worst=np.nan, info_eff=np.nan, info_eff_t2=np.nan,
                       survivors=np.nan)
        out.append(row)
    return out


def job(a):
    torch.set_num_threads(1)
    p, dens, seed = a
    A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                   blockiness=0.5, heterogeneity=1., internal_density=0.,
                                   graph_type='random', seed=seed, connected=True, max_attempts=300)
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)
    g0, T0, ci = occ(A, seed=seed); K = ci['k']; k = int(min(2*K, p))

    out = rows_for(A, T0, E_low, su, absA, tot, arm='occ', restart=-1, noise=np.nan, secs=0.)
    for name, kw in ARMS:
        for r in range(R):
            noise = NOISES[r % len(NOISES)]
            t0 = time.time()
            gen = torch.Generator().manual_seed(7000 + 101*seed + r)
            base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
            init = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*noise
            m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
            m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
            out += rows_for(A, m.get_partition()[0].detach().numpy(), E_low, su, absA, tot,
                            arm=name, restart=r, noise=noise, secs=time.time()-t0)
    for row in out:
        row.update(p=p, density=dens, seed=seed, K=K, k_offered=k)
    return out


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(3)]
    print(f'{len(jobs)} instances x {len(ARMS)} arms x {R} restarts x 2 fitters', flush=True)
    t0 = time.time(); rows = []
    path = f'{B}/experiments/abstraction_study/data/nochild_restarts.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    print('done %.1f min -> %s' % ((time.time()-t0)/60, path))
