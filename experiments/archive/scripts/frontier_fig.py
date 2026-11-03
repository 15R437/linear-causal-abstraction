"""Figure 1: the exactness-information frontier.

x = Relative Abstraction Error (RAE), y = fraction of low-level edge mass retained.
One panel per density. HailMary is swept over lambda_one_child (the dial); OCC, UCA and
a random partition are reference points. The composite "HailMary + OCC fallback" line is
traced by varying the acceptance tolerance alpha.

NOISE MODEL: standard normal exogenous noise, Sigma_u = I. Both metrics are then closed
form -- no sampling at all. RAE is the unified metric: max over the observational regime
and the k interventional regimes of tr(D^T D) / tr(R^T R), excluding clusters whose
T-weight has been driven below tau (where do(Y_i) is not well defined and the denominator
vanishes). DRE uses the population form (1/p) tr(Psi^T Psi), Psi = E(I - A_hat).

Retention uses a 1e-2 marginalisation tolerance, NOT exact zero: fit_abstraction_alternative
never returns a hard zero, so an exact-zero test silently stops charging for discarded
variables (understates loss by 0.07-0.29 and reverses which method looks better).
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, check_conditions
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output, cluster_indices
from src.models.wrapper import random as random_partition, uca
from occ import occ

B = '/Users/issy/Projects/linear-causal-abstraction'
LAMBDAS = [0., 0.3, 1., 3., 10., 30., 100., 300., 1000.]
P, EPOCHS, TAU = 100, 3000, 1e-2


def rae(A, T_hard, T, A_high, S, E_low, su, tau=TAU):
    """max over {observational, do(1..k)} of the Sigma-weighted relative error."""
    p, k = T.shape
    I_low, I_high = np.eye(p), np.eye(k)
    E_high = np.linalg.inv(I_high - A_high)

    def ratio(D, R):
        num = float(np.sum(su[:, None] * D ** 2))
        den = float(np.sum(su[:, None] * R ** 2))
        return np.inf if den <= 1e-30 else num / den

    # observational regime
    per = [ratio(S @ E_high - E_low @ T, E_low @ T)]

    # interventional regimes, skipping clusters with negligible T-weight
    F_ii = np.zeros((k, k, k)); idx = np.arange(k); F_ii[idx, idx, idx] = 1.0
    F_pi = (T_hard.T)[..., None] * I_low
    inv_hi = np.linalg.inv(I_high + F_ii @ (E_high - I_high))
    E_hi_do = E_high[None] - (E_high - I_high)[None] @ inv_hi @ F_ii @ E_high[None]
    inv_lo = np.linalg.inv(I_low + F_pi @ (E_low - I_low))
    E_lo_do = E_low - (E_low - I_low) @ inv_lo @ F_pi @ E_low
    colnorm = np.linalg.norm(T, axis=0)
    n_skip = 0
    for i in range(k):
        if colnorm[i] <= tau:
            n_skip += 1
            continue
        R = E_lo_do[i] @ T
        D = R - (S + (T - S) @ F_ii[i]) @ E_hi_do[i]
        per.append(ratio(D, R))
    return float(np.nanmax(per)), n_skip


def dre(A, T, A_high, su):
    """Data Reconstruction Error, population form: (1/p) tr(Sigma^-1 Psi^T Sigma Psi)."""
    p = A.shape[0]
    A_hat = T @ A_high @ np.linalg.pinv(T)
    E = np.linalg.inv(np.eye(p) - A)
    Psi = E @ (np.eye(p) - A_hat)
    num = np.diag(Psi.T @ (su[:, None] * Psi))
    return float(np.mean(num / np.clip(su, 1e-12, None)))


def score(A, T_hard, E_low, su, absA, tot, **extra):
    idx = [c for c in cluster_indices(T_hard) if len(c)]
    ai, oc, dag = check_conditions(A, cluster_indices(T_hard))
    co = T_hard @ T_hard.T; np.fill_diagonal(co, 0)
    row = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag),
               k_used=len(idx), largest=int(max(len(c) for c in idx)),
               intra=float((absA * co).sum() / tot), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T_hard, A_low=A, E_low=E_low)
        r, nskip = rae(A, T_hard, Tp, Ah, S, E_low, su)
        marg = np.abs(Tp).max(1) <= TAU
        lost = co.astype(bool).copy(); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        iem = float((absA * lost).sum() / tot)
        row.update(rae=r, n_skip=nskip, marg=int(marg.sum()),
                   iem=iem, retained=1.0 - iem, dre=dre(A, Tp, Ah, su),
                   norm_Ahigh=float(np.linalg.norm(Ah)))
    except Exception as e:
        row.update(rae=np.nan, retained=np.nan, dre=np.nan, err=type(e).__name__)
    return row


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(P) - A)
    su = np.ones(P)          # standard normal exogenous noise: Sigma_u = I
    g0, T0, ci = occ(A, seed=seed); K = max(ci['k'], 2)

    out = [score(A, T0, E_low, su, absA, tot, method='OCC', lam_oc=np.nan, k=K, seed=seed,
                 density=dens, secs=0.)]
    for nm, fn in (('random', random_partition), ('UCA', uca)):
        t0 = time.time()
        try:
            Tb = fn(A_low=A, k=K)
            Tb = (Tb == Tb.max(axis=1, keepdims=True)).astype(float) if nm == 'UCA' else Tb
            out.append(score(A, Tb, E_low, su, absA, tot, method=nm, lam_oc=np.nan,
                             k=K, seed=seed, density=dens, secs=time.time()-t0))
        except Exception as e:
            print('  %s failed: %s' % (nm, e), flush=True)
    for lam in LAMBDAS:
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337 + seed)
        base = np.zeros((P, K)); base[:, :min(K, T0.shape[1])] = T0[:, :K]
        init = torch.from_numpy(base).float() * 5.0 + torch.randn(P, K, generator=gen) * 0.1
        m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100.,
                lambda_one_child=lam, lambda_info=0., lambda_surj=0, verbose=False)
        out.append(score(A, m.get_partition()[0].detach().numpy(), E_low, su, absA, tot,
                         method='HailMary', lam_oc=lam, k=K, seed=seed, density=dens,
                         secs=time.time()-t0))
    return out


if __name__ == '__main__':
    jobs = [(d, s) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(8)]
    print(f'{len(jobs)} instances x ({len(LAMBDAS)} lambdas + 3 refs), p={P}, {EPOCHS} epochs',
          flush=True)
    t0 = time.time(); rows = []
    path = f'{B}/experiments/abstraction_study/data/frontier2.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time()-t0)/60, path))
