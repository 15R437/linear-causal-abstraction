"""Does normalising graph/recon by k stop the collapse to 2 clusters at large k?

THE SYMPTOM (reported): lambda_recon=1, lambda_graph=10, lambda_one_child=100 works
reasonably, but at large k it collapses to 2 clusters.

THE MECHANISM (measured, no training needed): one_child_loss is an unnormalised COUNT --
exactly mean_i d_i(d_i-1), where d_i is the number of distinct other clusters node i points
into -- and it grows ~k^1.0 as d_i -> out-degree. graph_loss and recon_loss are RATES over
all p^2 node pairs, but only Sigma(co_assign) = p(p/k - 1) pairs are co-assigned, so they
fall ~k^-0.75. The ratio one_child/graph therefore drifts as k^+1.8: at fixed lambdas the
merging term wins at large k, and 2 clusters is where it lands (one-child holds trivially).

THE FIXES TESTED
  none    current behaviour
  k       multiply graph and recon by k -- removes exactly one power (drift k^+1.8 -> k^+0.8)
  comass  divide by the DETACHED co-assignment mass -> drift k^+0.45; exact at all k, where
          x k is only the leading-order approximation and breaks down as k -> p
  full    comass, plus one_child divided by a detached d_bar(d_bar-1) -> targets full
          k-invariance

Any T_soft-dependent denominator MUST be detached: otherwise the optimiser lowers the loss
by increasing co-assignment mass, which is a spurious merging pressure -- the very pathology
being removed.
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
from src.utils.generate_data import sample_low_level_graph
from occ import occ, conditions

B = '/Users/issy/Projects/linear-causal-abstraction'
NORMS = ['none', 'k', 'comass', 'full']
SLACKS = [1.0, 2.0, 4.0]
LAM = dict(lambda_recon=1., lambda_graph=10., lambda_one_child=100., lambda_info=0.)


class NormHailMary(HailMary):
    """HailMary with the graph/recon/one_child terms put on a k-invariant footing.

    Rescaling forward()'s returned terms is equivalent to rescaling the lambdas, since
    train_step only multiplies and sums -- so no optimiser code changes, and every factor
    is detached so it contributes no gradient of its own.
    """

    def __init__(self, *a, norm='none', **kw):
        super().__init__(*a, **kw)
        self.norm = norm

    def forward(self, temp=1., compute_recon=True, compute_graph=True, compute_one_child=True):
        rec, gr, oc, su, en, inf = super().forward(temp, compute_recon, compute_graph,
                                                   compute_one_child)
        if self.norm == 'none':
            return rec, gr, oc, su, en, inf
        p = self.n_low
        _, T_soft = self.get_partition(temp)
        with torch.no_grad():
            if self.norm == 'k':
                pair_scale = torch.tensor(float(self.n_high), dtype=gr.dtype)
            else:
                co = (T_soft @ T_soft.T) * (1. - torch.eye(p, device=self.device))
                pair_scale = (p * p) / co.sum().clamp(min=1e-9)
            oc_scale = torch.tensor(1.0, dtype=oc.dtype)
            if self.norm == 'full':
                # d_bar from the same H used by one_child_loss, so the divisor is the
                # violation opportunity actually available at this k
                H = torch.max(self.A_low_binary[:, :, None, :] * T_soft.T[None, None, :, :],
                              dim=-1)[0] * (1 - T_soft)
                dbar = H.sum(dim=-1).mean()
                oc_scale = 1.0 / torch.clamp(dbar * (dbar - 1.), min=1e-3)
        return rec * pair_scale, gr * pair_scale, oc * oc_scale, su, en, inf


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
    absA = np.abs(A); tot = absA.sum()
    g0, T0, ci = occ(A, seed=seed); K = ci['k']
    out = []
    for norm in NORMS:
        for sl in SLACKS:
            k = int(np.clip(round(sl * K), 2, p))
            t0 = time.time()
            gen = torch.Generator().manual_seed(31337 + seed)
            base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
            init = torch.from_numpy(base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
            m = NormHailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(),
                             logits_init=init, norm=norm)
            m.train(num_epochs=3000, lambda_surj=0., verbose=False, **LAM)
            T = m.get_partition()[0].detach().numpy()
            gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(k)) if len(x)]
            ai, oc_, dq = conditions(A, gs)
            co = T @ T.T; np.fill_diagonal(co, 0)
            out.append(dict(p=p, density=dens, seed=seed, K=K, norm=norm, slack=sl,
                            k_offered=k, k_used=len(gs),
                            collapsed=int(len(gs) <= 2),
                            largest=int(max(len(x) for x in gs)),
                            allin=int(ai), one_child=int(oc_), valid=int(ai and oc_ and dq),
                            intra=float((absA * co).sum() / tot), secs=time.time() - t0))
    return out


if __name__ == '__main__':
    jobs = [(50, d, s) for d in (0.2, 0.5) for s in range(3)]
    print(f'{len(jobs)} instances x {len(NORMS)} norms x {len(SLACKS)} slacks', flush=True)
    t0 = time.time(); rows = []
    path = f'{B}/experiments/abstraction_study/data/knorm.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    d = pd.DataFrame(rows); d.to_csv(path, index=False)
    pd.set_option('display.width', 240)
    print('\ndone %.1f min\n' % ((time.time()-t0)/60))
    print('COLLAPSE RATE (k_used <= 2) by normalisation x slack:')
    print(d.pivot_table(index='norm', columns='slack', values='collapsed').reindex(NORMS).round(2).to_string())
    print('\nk_used by normalisation x slack (k_offered in brackets):')
    print(d.pivot_table(index='norm', columns='slack', values='k_used').reindex(NORMS).round(1).to_string())
    print(d.pivot_table(index='norm', columns='slack', values='k_offered').reindex(NORMS).round(0).to_string())
    print('\nAll-In / one_child / intra by normalisation:')
    print(d.groupby('norm').agg(allin=('allin','mean'), one_child=('one_child','mean'),
        valid=('valid','mean'), k_used=('k_used','mean'), largest=('largest','mean'),
        intra=('intra','mean'), collapsed=('collapsed','mean')).reindex(NORMS).round(3).to_string())
