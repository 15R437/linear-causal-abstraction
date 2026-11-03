"""HailMary on the Leontief (EXIOBASE GB) graph -- the RAW CYCLIC matrix.

NO enforce_dag: the Leontief matrix is genuinely cyclic (8219 reciprocal pairs among 172
nodes) and forcing it acyclic throws away half the edges (density 0.615 -> 0.327).
fit_abstraction_alternative handles cyclic low-level graphs, which is the reason to use it
here; only the QUOTIENT must be acyclic for the sufficiency theorem, and with cycles this
dense that means every cycle has to sit INSIDE a cluster.

Two arms, random init:
  their lambdas : recon=1, graph=100, one_child=1000
  graph_only    : graph=100 alone (scale is immaterial when it is the only term)
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
from src.utils.helper import process_output, cluster_indices
from src.utils.losses import observation_loss, intervention_loss
from src.utils.generate_data import sample_by_noise_adjustment_any_order
from occ import occ

ARMS = {'their_lam': dict(lambda_recon=1., lambda_graph=100., lambda_one_child=1000., lambda_info=0.),
        'graph_only': dict(lambda_recon=0., lambda_graph=100., lambda_one_child=0., lambda_info=0.)}
A_LOW = np.load('/Users/issy/Projects/linear-causal-abstraction/data/A_low.npy')  # RAW, cyclic
P = A_LOW.shape[0]


def score(T, E_low, absA, tot, **extra):
    A = A_LOW
    idx = [x for x in cluster_indices(T) if len(x)]
    ai, oc, dag = check_conditions(A, cluster_indices(T))
    co = T @ T.T; np.fill_diagonal(co, 0)
    row = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag),
               k_used=len(idx), largest=int(max(len(x) for x in idx)),
               intra=float((absA * co).sum() / tot), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        obs = float(observation_loss(E_low=E_low, E_high=Eh, S=S, T=Tp))
        intv = float(intervention_loss(E_low=E_low, E_high=Eh, S=S, T=Tp, T_hard=T))
        rowmax = np.abs(Tp).max(1)
        marg = rowmax <= 1e-2
        lost = co.astype(bool).copy(); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        row.update(obs=obs, intv=intv, exact=int(obs < 1e-8 and intv < 1e-8),
                   marg_t2=int(marg.sum()),
                   info_eff=float((absA * co.astype(bool)).sum() / tot),
                   info_eff_t2=float((absA * lost).sum() / tot))
    except Exception as e:
        row.update(obs=np.nan, intv=np.nan, exact=0, marg_t2=np.nan,
                   info_eff=np.nan, info_eff_t2=np.nan, err=type(e).__name__)
    return row


def job(a):
    torch.set_num_threads(1)
    arm, k, seed = a
    absA = np.abs(A_LOW); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A_LOW)
    t0 = time.time()
    torch.manual_seed(seed)
    m = HailMary(n_low=P, n_high=k, A_low=torch.from_numpy(A_LOW).float(), init_type='random')
    m.train(num_epochs=6000, lambda_surj=0, verbose=False, **ARMS[arm])
    T = m.get_partition()[0].cpu().detach().numpy()
    return score(T, E_low, absA, tot, arm=arm, k=k, seed=seed, secs=time.time()-t0)


if __name__ == '__main__':
    absA = np.abs(A_LOW); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A_LOW)
    g, T0, ci = occ(A_LOW, n_draws=12, seed=0)
    print(f'Leontief: p={P}  density={(A_LOW!=0).mean():.3f}  max out-degree={(A_LOW!=0).sum(1).max()}')
    print(f"OCC: K={ci['k']}  repaired={ci['repaired']}  merges={ci['n_merges']}  "
          f"conditions=({ci['allin']},{ci['one_child']},{ci['acyclic']})  "
          f"sizes[:6]={ci['cluster_sizes'][:6]}")
    occ_row = score(T0, E_low, absA, tot, arm='occ', k=ci['k'], seed=0, secs=0.)
    jobs = [(a, k, s) for a in ARMS for k in (6, 10) for s in (0, 1, 2)]
    print(f'\nHailMary: random init, 6000 epochs, {len(jobs)} runs\n', flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = pool.map(job, jobs)
    d = pd.DataFrame(rows + [occ_row])
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/leontief_hm.csv', index=False)
    pd.set_option('display.width', 250)
    print('done %.1f min\n' % ((time.time()-t0)/60))
    cols = ['arm','k','seed','k_used','largest','allin','one_child','valid','exact',
            'obs','intv','intra','info_eff_t2','marg_t2','secs']
    print(d[cols].round(4).to_string(index=False))
    print('\nby arm x k (mean of 3 random inits):')
    h = d[d.arm!='occ']
    print(h.groupby(['arm','k']).agg(n=('valid','size'), k_used=('k_used','mean'), largest=('largest','mean'),
        allin=('allin','mean'), one_child=('one_child','mean'), valid=('valid','mean'),
        exact=('exact','mean'), intra=('intra','mean'), info_eff_t2=('info_eff_t2','mean'),
        marg_t2=('marg_t2','mean'), secs=('secs','mean')).round(3).to_string())
    # what structure emerged: cluster sizes, and sector names for the non-trivial clusters
    import pickle
    try:
        with open('/Users/issy/Projects/linear-causal-abstraction/data/A_low_labels.pkl','rb') as f:
            lab = pickle.load(f)
        names = [s_ for (_, s_) in lab['sectors']] + list(lab['factors'])
    except Exception:
        names = [f'node{i}' for i in range(P)]
    print('\n=== structure of the best graph_only run (largest k_used) ===')
    go = d[d.arm=='graph_only']
    if len(go):
        best = go.loc[go.k_used.idxmax()]
        torch.manual_seed(int(best.seed))
        m = HailMary(n_low=P, n_high=int(best.k), A_low=torch.from_numpy(A_LOW).float(), init_type='random')
        m.train(num_epochs=6000, lambda_surj=0, verbose=False, **ARMS['graph_only'])
        T = m.get_partition()[0].cpu().detach().numpy()
        cl = [x for x in cluster_indices(T) if len(x)]
        cl.sort(key=len, reverse=True)
        print(f'k={int(best.k)} seed={int(best.seed)}  cluster sizes: {[len(c) for c in cl]}')
        for j, c in enumerate(cl):
            if len(c) > 25:
                print(f'  cluster {j} (n={len(c)}): [too large to list]')
            else:
                print(f'  cluster {j} (n={len(c)}): ' + '; '.join(names[i][:38] for i in c))
    o = d[d.arm=='occ'].iloc[0]
    print(f"\nOCC reference: k_used={o.k_used}  largest={o.largest}  valid={o.valid}  "
          f"exact={o.exact}  intra={o.intra:.3f}  info_eff_t2={o.info_eff_t2:.3f}  marg={o.marg_t2}")
