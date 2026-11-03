"""HailMary on the Leontief SCC core, sweeping lambda_one-child.

THE SUBGRAPH. The raw EXIOBASE GB matrix (p=172) splits into one strongly connected
component of 136 sectors, 9 factor sinks, and 26 zero-edge sectors. We keep only the SCC:
p=136, density 0.934, mean out-degree 127. Because it is strongly connected, EVERY
non-trivial partition has a cyclic quotient -- so the acyclic-quotient condition is
unachievable by construction and we are necessarily in the approximate/cyclic regime. This
is the honest setting in which to ask whether relaxing the conditions buys interpretable
structure.

WARM START. AOC Closure returns k=1 on this subgraph (one-child forces total collapse when
every node reaches nearly every other), with information loss 0.900. It is therefore
useless both as a warm start and as a fallback, so we use random initialisation and choose
k ourselves. AOC Closure's k=1 numbers are still recorded as the reference point.

NOISE MODEL. Sigma_u = I (standard normal), as in the synthetic experiments. There is no
canonical exogenous distribution for an input-output table -- the economically natural
choice would weight by sector output or final-demand volatility, which EXIOBASE's A matrix
alone does not give us -- so we take the same convention as the rest of the paper and note
that RAE would change under a sector-size weighting.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, json, pickle, warnings, collections
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from src.models.hail_mary import HailMary, check_conditions
from src.utils.helper import process_output, cluster_indices, is_dag
from src.utils.losses import observation_loss, intervention_loss
from occ import occ

B = '/Users/issy/Projects/linear-causal-abstraction'
TAU = 1e-2
LAMS = [0., 1., 10., 100., 1000.]
KS = [10, 20]
SEEDS = [0, 1, 2]
EPOCHS = 3000


def load_scc():
    A = np.load(f'{B}/data/A_low.npy')
    lab = connected_components(sp.csr_matrix(A != 0), directed=True, connection='strong')[1]
    big = collections.Counter(lab).most_common(1)[0][0]
    idx = np.where(lab == big)[0]
    L = pickle.load(open(f'{B}/data/A_low_labels.pkl', 'rb'))
    names = [s for (_, s) in L['sectors']] + list(L['factors'])
    return A[np.ix_(idx, idx)], [names[i] for i in idx], idx


A_SCC, NAMES, IDX = load_scc()
P = A_SCC.shape[0]


def rae(A, T_hard, T, A_high, S, E_low, su, tau=TAU):
    """max over {observational, do(1..k)} of the Sigma-weighted relative error."""
    p, k = T.shape
    I_low, I_high = np.eye(p), np.eye(k)
    E_high = np.linalg.inv(I_high - A_high)

    def ratio(D, R):
        num = float(np.sum(su[:, None] * D ** 2))
        den = float(np.sum(su[:, None] * R ** 2))
        return np.inf if den <= 1e-30 else num / den

    per = [ratio(S @ E_high - E_low @ T, E_low @ T)]
    F_ii = np.zeros((k, k, k)); ar = np.arange(k); F_ii[ar, ar, ar] = 1.0
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


def score(T_hard, **extra):
    A = A_SCC
    absA = np.abs(A); tot = absA.sum()
    tot_losable = tot - np.abs(np.diag(A)).sum()   # self-loops cannot be clustered away
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    idx = [c for c in cluster_indices(T_hard) if len(c)]
    ai, oc, dq = check_conditions(A, cluster_indices(T_hard))
    co = T_hard @ T_hard.T; np.fill_diagonal(co, 0)
    sizes = sorted((len(c) for c in idx), reverse=True)
    row = dict(allin=int(ai), one_child=int(oc), acyclic=int(dq), k_used=len(idx),
               largest=sizes[0], sizes=json.dumps(sizes),
               intra=float((absA * co).sum() / tot),
               intra_losable=float((absA * co).sum() / tot_losable), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T_hard, A_low=A, E_low=E_low)
        r, nskip = rae(A, T_hard, Tp, Ah, S, E_low, su)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        obs = float(observation_loss(E_low=E_low, E_high=Eh, S=S, T=Tp))
        intv = float(intervention_loss(E_low=E_low, E_high=Eh, S=S, T=Tp, T_hard=T_hard))
        marg = np.abs(Tp).max(1) <= TAU
        lost = co.astype(bool).copy(); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        A_clean = np.where(np.abs(Ah) > 1e-9, Ah, 0.0)
        row.update(rae=r, n_skip=nskip, marg=int(marg.sum()),
                   iem=float((absA * lost).sum() / tot),
                   iem_losable=float((absA * lost).sum() / tot_losable), obs=obs, intv=intv,
                   exact=int(obs < 1e-8 and intv < 1e-8),
                   norm_Ahigh=float(np.linalg.norm(Ah)),
                   max_Ahigh=float(np.abs(Ah).max()),
                   edges_Ahigh=int((np.abs(Ah) > 1e-9).sum()),
                   dag_Ahigh=int(is_dag(A_clean)))
    except Exception as e:
        row.update(rae=np.nan, iem=np.nan, err=type(e).__name__)
    return row


def job(a):
    torch.set_num_threads(1)
    k, lam, seed = a
    t0 = time.time()
    torch.manual_seed(seed)
    m = HailMary(n_low=P, n_high=k, A_low=torch.from_numpy(A_SCC).float(), init_type='random')
    m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
            lambda_info=0., lambda_surj=0, verbose=False)
    T = m.get_partition()[0].cpu().detach().numpy()
    row = score(T, method='HailMary', k=k, lam_oc=lam, seed=seed, secs=time.time() - t0)
    # keep the assignment so cluster composition can be inspected afterwards
    row['assign'] = json.dumps([int(x) for x in np.argmax(T, axis=1)])
    return row


if __name__ == '__main__':
    print(__doc__.split('\n')[0], flush=True)
    print('SCC: p=%d density=%.3f' % (P, (A_SCC != 0).mean()), flush=True)
    g, T0, ci = occ(A_SCC, n_draws=12, seed=0)
    occ_row = score(T0, method='OCC', k=ci['k'], lam_oc=np.nan, seed=0, secs=0.)
    occ_row['assign'] = json.dumps([int(x) for x in np.argmax(T0, axis=1)])
    print('OCC reference: k=%d iem=%.3f rae=%.4f' % (occ_row['k_used'], occ_row['iem'], occ_row['rae']), flush=True)

    jobs = [(k, lam, s) for k in KS for lam in LAMS for s in SEEDS]
    print('%d HailMary runs (k in %s, lambda_oc in %s, %d seeds)\n' % (len(jobs), KS, LAMS, len(SEEDS)), flush=True)
    t0 = time.time(); rows = [occ_row]
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            pd.DataFrame(rows).to_csv(f'{B}/experiments/abstraction_study/data/leontief_scc.csv', index=False)
            print('%d/%d  %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{B}/experiments/abstraction_study/data/leontief_scc.csv', index=False)
    with open(f'{B}/experiments/abstraction_study/data/leontief_scc_names.json', 'w') as f:
        json.dump(NAMES, f)
    pd.set_option('display.width', 250)
    print('\ndone %.1f min\n' % ((time.time()-t0)/60))
    h = d[d.method == 'HailMary']
    print(h.groupby(['k', 'lam_oc']).agg(
        n=('iem', 'size'), k_used=('k_used', 'mean'), largest=('largest', 'mean'),
        intra=('intra', 'mean'), iem=('iem', 'mean'),
        iem_los=('iem_losable', 'mean'), rae=('rae', 'median'),
        marg=('marg', 'mean'), normA=('norm_Ahigh', 'mean'),
        allin=('allin', 'mean'), secs=('secs', 'mean')).round(4).to_string())
