"""Leontief (EXIOBASE3, region GB): all methods at a few thresholds, current metrics.

Only two things vary: the coefficient THRESHOLD (a preprocessing choice) and the METHOD.
Everything else is one fixed graph. AOC Closure, OC Closure and Greedy All-In are
deterministic and run once per threshold; GradCAP and the random control are stochastic and
get RESTARTS each, which is what makes the reproducibility (ARI) comparison possible.

GradCAP is run under BOTH initialisations:
  gradcap-rand   regularised_surjective_init      (what the old Leontief script used, because
                                                   AOC returned k=1 on the unbudgeted graph)
  gradcap-occ    occ_ac_logits, i.e. seeded from AOC Closure  (what the synthetic experiments
                                                   use; viable here now the budget gives k=20)

Metrics are the canonical ones from src/utils/losses.py as of 2026-09-24: RAE returns
sqrt(num/den), ReconErr is reconstruction_loss (a ratio of Frobenius NORMS). Fitting is
fit_abstraction_auto(faithful=True).

Acyclicity of the quotient is RECORDED but is NOT a validity condition: the abstraction
equation (I-A)S = T(I-A_high) never invokes a topological order, and a cyclic ground-truth
quotient is exact (REPORT.md). It matters only operationally, since it routes the fitter to
the _K branch.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, json, pickle, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, check_conditions
from src.utils.helper import fit_abstraction_auto, get_S, cluster_indices, is_dag
from src.utils.losses import RAE, reconstruction_loss
from occ import occ, oc_closure, budget_coarsening
from partition_baselines import allin_agglom

B = '/Users/issy/Projects/linear-causal-abstraction'
TAU, K, EPOCHS = 1e-2, 20, 3000
THRS = [0.02, 0.05, 0.1]


def build(thr):
    A0 = np.load(f'{B}/data/A_low.npy')
    L = pickle.load(open(f'{B}/data/A_low_labels.pkl', 'rb'))
    names = [s for (_, s) in L['sectors']] + list(L['factors'])
    A = np.where(np.abs(A0) > thr, A0, 0.0)
    live = np.where((A != 0).any(0) | (A != 0).any(1))[0]
    return A[np.ix_(live, live)], [names[i] for i in live]


def score(A, Th, **extra):
    p = A.shape[0]
    absA = np.abs(A); tot = absA.sum()
    E = np.linalg.inv(np.eye(p) - A); su = np.ones(p)
    ai, oc, dq = check_conditions(A, cluster_indices(Th))
    idx = [c for c in cluster_indices(Th) if len(c)]
    co = Th @ Th.T; np.fill_diagonal(co, 0)
    sizes = sorted((len(c) for c in idx), reverse=True)
    row = dict(allin=int(ai), one_child=int(oc), acyclic=int(dq), k_used=len(idx),
               largest=sizes[0] if sizes else 0, p=p,
               intra=float((absA * co).sum() / tot),
               intra_edges=float(((A != 0) * co).sum() / max(int((A != 0).sum()), 1)),
               **extra)
    try:
        Ah, Tp, fm = fit_abstraction_auto(A_low=A, T_hard=Th, E_low=E, n_restarts=5,
                                          seed=0, maxiter=100, faithful=True,
                                          return_method=True)
        S = get_S(T=Tp, T_hard=Th, A_low=A)
        row.update(rae=RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su,
                           tau=TAU),
                   recon=reconstruction_loss(A_low=A, T=Tp, A_high=Ah),
                   fit_method=fm, marg=int((np.abs(Tp).max(1) <= TAU).sum()),
                   norm_Ahigh=float(np.linalg.norm(Ah)))
    except Exception as e:
        row.update(rae=np.nan, recon=np.nan, err=type(e).__name__)
    row['assign'] = json.dumps([int(x) for x in np.argmax(Th, axis=1)])
    return row


def job(a):
    torch.set_num_threads(1)
    thr, meth, seed = a
    A, _ = build(thr)
    p = A.shape[0]
    t0 = time.time()
    if meth == 'AOC Closure':
        Th = occ(A, n_draws=12, seed=0, k=K)[1]
    elif meth == 'OC Closure':
        g, Th, _ = oc_closure(A, n_draws=12, seed=0)
        if len(g) > K:
            try:
                g2, _ = budget_coarsening(A, g, K, rng=np.random.default_rng(0))
                Th = np.zeros((p, len(g2)))
                for j, mem in enumerate(g2):
                    Th[np.asarray(mem), j] = 1.0
            except Exception:
                pass
    elif meth == 'Greedy All-In':
        Th = allin_agglom(A, min(K, p))
    elif meth == 'random balanced':
        rng = np.random.default_rng(1000 + seed)
        asg = np.tile(np.arange(K), p // K + 1)[:p]; rng.shuffle(asg)
        Th = np.zeros((p, K)); Th[np.arange(p), asg] = 1.
    else:                                   # gradcap-rand / gradcap-occ
        torch.manual_seed(seed)
        init = 'occ_ac' if meth.endswith('occ') else 'random'
        m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), init_type=init)
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=0.,
                lambda_info=0., lambda_surj=0, verbose=False)
        Th = m.get_partition()[0].cpu().detach().numpy()
    return score(A, Th, method=meth, thr=thr, seed=seed, secs=time.time() - t0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, default=5)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--out', default='leontief_quick')
    args = ap.parse_args()
    DET = ['AOC Closure', 'OC Closure', 'Greedy All-In']
    STO = ['gradcap-rand', 'gradcap-occ', 'random balanced']
    jobs = [(t, m, 0) for t in THRS for m in DET]
    jobs += [(t, m, s) for t in THRS for m in STO for s in range(args.restarts)]
    print('%d jobs = %d thresholds x (%d deterministic + %d stochastic x %d restarts)'
          % (len(jobs), len(THRS), len(DET), len(STO), args.restarts), flush=True)
    path = f'{B}/experiments/abstraction_study/data/{args.out}.csv'
    t0 = time.time(); rows = []
    with mp.Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r); pd.DataFrame(rows).to_csv(path, index=False)   # checkpoint
            el = (time.time() - t0) / 60
            print('%d/%d %.1fmin (eta %.0fmin)' % (i, len(jobs), el, el / i * (len(jobs) - i)),
                  flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
