"""HailMary vs AOC Closure across the full range of coefficient thresholds.

20 INDEPENDENT restarts per threshold at a fixed lambda_one-child = 0, to settle the
jaggedness seen with 3. Restarts here are genuinely independent: AOC Closure returns k=1 on
this graph, so initialisation is regularised_surjective_init with a distinct torch seed per
run -- unlike the synthetic experiments, where init is 5.0*onehot(pi_0) + 0.1*randn and the
"restarts" are jitter around a single AOC Closure partition.

Original docstring follows.

Does thresholding negligible input coefficients make the Leontief graph partitionable?

The raw matrix is density 0.615 with a 136-node SCC, and on it neither AOC Closure nor
HailMary finds anything: IEM ~ 1/k for any partition, seed agreement is chance-level, and a
random balanced partition matches the learned one. That is a density problem.

Thresholding small coefficients is standard in input-output analysis (a coefficient of
0.001 means a sector draws 0.1% of its inputs from another) and is already used for the
illustrative figure in this paper. A screen shows the giant SCC shattering at
threshold ~0.02 (53 -> 5 nodes) and the graph becoming acyclic at 0.05, while retaining
85% / 75% of the total coefficient mass.

This tests whether the structure becomes REAL at those thresholds, using the diagnostics
that failed on the dense graph:
  * seed agreement (adjusted Rand index) -- is the partition reproducible?
  * comparison against a random balanced partition at matched k -- is learning doing work?
Isolated nodes (no edges after thresholding) are dropped: they are free singletons that
inflate k and carry no mass.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, json, pickle, warnings, collections, itertools
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from sklearn.metrics import adjusted_rand_score
from src.models.hail_mary import HailMary, check_conditions
from src.utils.helper import process_output, cluster_indices
from src.utils.losses import observation_loss, intervention_loss
from occ import occ

B = '/Users/issy/Projects/linear-causal-abstraction'
TAU = 1e-2
THRS = [0.001, 0.003, 0.005, 0.008, 0.012, 0.016, 0.02, 0.03, 0.05, 0.07, 0.10]
LAMS = [0.]
SEEDS = list(range(20))
K, EPOCHS = 20, 3000


def build(thr):
    A0 = np.load(f'{B}/data/A_low.npy')
    L = pickle.load(open(f'{B}/data/A_low_labels.pkl', 'rb'))
    names = [s for (_, s) in L['sectors']] + list(L['factors'])
    A = np.where(np.abs(A0) > thr, A0, 0.0)
    live = np.where((A != 0).any(0) | (A != 0).any(1))[0]      # drop isolated nodes
    return A[np.ix_(live, live)], [names[i] for i in live]


def rae_of(A, T_hard, T, A_high, S, E_low, su, tau=TAU):
    p, k = T.shape
    I_low, I_high = np.eye(p), np.eye(k)
    E_high = np.linalg.inv(I_high - A_high)

    def ratio(D, R):
        num = float(np.sum(su[:, None] * D ** 2)); den = float(np.sum(su[:, None] * R ** 2))
        return np.inf if den <= 1e-30 else num / den

    per = [ratio(S @ E_high - E_low @ T, E_low @ T)]
    F_ii = np.zeros((k, k, k)); ar = np.arange(k); F_ii[ar, ar, ar] = 1.0
    F_pi = (T_hard.T)[..., None] * I_low
    E_hi_do = E_high[None] - (E_high - I_high)[None] @ np.linalg.inv(I_high + F_ii @ (E_high - I_high)) @ F_ii @ E_high[None]
    E_lo_do = E_low - (E_low - I_low) @ np.linalg.inv(I_low + F_pi @ (E_low - I_low)) @ F_pi @ E_low
    cn = np.linalg.norm(T, axis=0)
    for i in range(k):
        if cn[i] <= tau:
            continue
        R = E_lo_do[i] @ T
        per.append(ratio(R - (S + (T - S) @ F_ii[i]) @ E_hi_do[i], R))
    return float(np.nanmax(per))


def score(A, T_hard, **extra):
    p = A.shape[0]
    absA = np.abs(A); tot = absA.sum(); tot_los = tot - np.abs(np.diag(A)).sum()
    E_low = np.linalg.inv(np.eye(p) - A); su = np.ones(p)
    idx = [c for c in cluster_indices(T_hard) if len(c)]
    ai, oc, dq = check_conditions(A, cluster_indices(T_hard))
    co = T_hard @ T_hard.T; np.fill_diagonal(co, 0)
    sizes = sorted((len(c) for c in idx), reverse=True)
    row = dict(allin=int(ai), one_child=int(oc), acyclic=int(dq), k_used=len(idx),
               largest=sizes[0], sizes=json.dumps(sizes), p=p,
               intra=float((absA * co).sum() / tot), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T_hard, A_low=A, E_low=E_low)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        marg = np.abs(Tp).max(1) <= TAU
        lost = co.astype(bool).copy(); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        row.update(rae=rae_of(A, T_hard, Tp, Ah, S, E_low, su), marg=int(marg.sum()),
                   iem=float((absA * lost).sum() / tot),
                   iem_losable=float((absA * lost).sum() / tot_los),
                   obs=float(observation_loss(E_low=E_low, E_high=Eh, S=S, T=Tp)),
                   norm_Ahigh=float(np.linalg.norm(Ah)),
                   frac_neg=float((Tp[T_hard == 1] < 0).mean()))
    except Exception as e:
        row.update(rae=np.nan, iem=np.nan, err=type(e).__name__)
    return row


def job(a):
    torch.set_num_threads(1)
    thr, kind, lam, seed = a
    A, names = build(thr)
    p = A.shape[0]
    if kind == 'random':
        rng = np.random.default_rng(1000 + seed)
        asg = np.tile(np.arange(K), p // K + 1)[:p]; rng.shuffle(asg)
        T = np.zeros((p, K)); T[np.arange(p), asg] = 1.
        r = score(A, T, method='random_balanced', thr=thr, lam_oc=np.nan, seed=seed, secs=0.)
    else:
        t0 = time.time()
        torch.manual_seed(seed)
        m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
                lambda_info=0., lambda_surj=0, verbose=False)
        T = m.get_partition()[0].cpu().detach().numpy()
        r = score(A, T, method='HailMary', thr=thr, lam_oc=lam, seed=seed, secs=time.time()-t0)
    r['assign'] = json.dumps([int(x) for x in np.argmax(T, axis=1)])
    return r


if __name__ == '__main__':
    rows = []
    for thr in THRS:
        A, names = build(thr)
        g, T0, ci = occ(A, n_draws=12, seed=0)
        r = score(A, T0, method='OCC', thr=thr, lam_oc=np.nan, seed=0, secs=0.)
        r['assign'] = json.dumps([int(x) for x in np.argmax(T0, axis=1)])
        rows.append(r)
        print('thr=%g: p=%d density=%.3f | OCC k=%d iem=%.3f repaired=%s'
              % (thr, A.shape[0], (A != 0).mean(), ci['k'], r['iem'], ci['repaired']), flush=True)
        json.dump(names, open(f'{B}/experiments/abstraction_study/data/leontief_r20_thr{thr}_names.json', 'w'))

    jobs = [(t, 'HailMary', l, s) for t in THRS for l in LAMS for s in SEEDS]
    jobs += [(t, 'random', np.nan, s) for t in THRS for s in SEEDS]
    print('\n%d runs\n' % len(jobs), flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/leontief_thresh_r20.csv', index=False)
    pd.set_option('display.width', 250)
    print('\ndone %.1f min\n' % ((time.time()-t0)/60))
    print(d.groupby(['thr', 'method', 'lam_oc'], dropna=False).agg(
        n=('iem', 'size'), k_used=('k_used', 'mean'), largest=('largest', 'mean'),
        intra=('intra', 'mean'), iem=('iem', 'mean'), rae=('rae', 'median'),
        marg=('marg', 'mean'), allin=('allin', 'mean'), fneg=('frac_neg', 'mean'),
        normA=('norm_Ahigh', 'mean')).round(4).to_string())
    print('\nSEED AGREEMENT (adjusted Rand index) -- is the partition reproducible?')
    for (thr, meth, lam), g in d[d.method != 'OCC'].groupby(['thr', 'method', 'lam_oc'], dropna=False):
        As = [np.array(json.loads(x)) for x in g["assign"]]
        if len(As) > 1:
            aris = [adjusted_rand_score(x, y) for x, y in itertools.combinations(As, 2)]
            print('   thr=%g %-15s lam=%-6s mean ARI %+.3f   pairs %s'
                  % (thr, meth, lam, np.mean(aris), ' '.join('%+.3f' % v for v in aris)))
