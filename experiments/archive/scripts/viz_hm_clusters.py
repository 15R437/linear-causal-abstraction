"""Cluster-size distributions: OCC vs the recommended HailMary refinement.

Recommended config: closure init at slack 2K, lambda_one_child=100, lambda_info=10,
lambda_graph=0; accept when measured relative error <= alpha (default 0.05) AND info_eff improves,
otherwise keep OCC's. The second clause is essential: without it the rule accepts a WORSE
partition on 12-21% of instances (e.g. p=100 d=0.1 where OCC is already lossless).

Prints concrete example partitions and writes a figure comparing the size distributions.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import process_output
from occ import occ, conditions
B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
ALPHA = 0.05


def job(a):
    torch.set_num_threads(1)
    p, dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return None
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)
    g0, T0, ci = occ(A, seed=seed)
    K = ci['k']; k = int(min(round(2 * K), p))
    gen = torch.Generator().manual_seed(31337 + seed)
    base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
    init = torch.from_numpy(base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
    m.train(num_epochs=6000, lambda_graph=0., lambda_one_child=100., lambda_info=10.,
            lambda_recon=0., lambda_surj=0., verbose=False)
    T1 = m.get_partition()[0].detach().numpy()
    g1 = [x for x in (np.where(T1[:, j] == 1)[0] for j in range(k)) if len(x)]

    def ev(T, gs):
        co = T @ T.T; np.fill_diagonal(co, 0)
        try:
            S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
            Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
            Dm = S @ Eh - E_low @ Tp; R = E_low @ Tp
            rel = float(np.sum(su[:, None]*Dm**2)/max(np.sum(su[:, None]*R**2), 1e-30))
            marg = (np.abs(Tp).sum(1) == 0)
            lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
            np.fill_diagonal(lost, False)
            ie = float((absA*lost).sum()/tot)
        except Exception:
            rel, ie = np.nan, float((absA*co).sum()/tot)
        return rel, ie
    rel0, ie0 = ev(T0, g0)
    rel1, ie1 = ev(T1, g1)
    accepted = bool(rel1 <= ALPHA and ie1 < ie0)   # error within alpha AND actually better
    return dict(p=p, density=dens, seed=seed, K=K, k_offered=k, accepted=accepted,
                occ_sizes=';'.join(str(len(x)) for x in sorted(g0, key=len, reverse=True)),
                hm_sizes=';'.join(str(len(x)) for x in sorted(g1, key=len, reverse=True)),
                occ_info=ie0, hm_info=ie1, hm_rel=rel1,
                occ_largest=max(len(x) for x in g0), hm_largest=max(len(x) for x in g1),
                occ_k=len(g0), hm_k=len(g1))


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(5)]
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows); d.to_csv(f'{B}/data/viz_hm_clusters.csv', index=False)
    print('%d instances, accepted (rel_err <= %.2f): %.2f\n' % (len(d), ALPHA, d.accepted.mean()))
    print('=== CONCRETE EXAMPLES (sizes, largest first) ===')
    for _, r in d[d.accepted].groupby(['p','density']).head(1).iterrows():
        o = [int(x) for x in r.occ_sizes.split(';')]
        h = [int(x) for x in r.hm_sizes.split(';')]
        print('\np=%d d=%.1f seed=%d   (rel_err %.4f)' % (r.p, r.density, r.seed, r.hm_rel))
        print('  OCC       k=%2d  largest %3d (%.0f%%)  sizes %s' % (
            len(o), o[0], 100*o[0]/r.p, o[:10]))
        print('  HailMary  k=%2d  largest %3d (%.0f%%)  sizes %s' % (
            len(h), h[0], 100*h[0]/r.p, h[:10]))
        print('  info_eff  %.3f -> %.3f' % (r.occ_info, r.hm_info))

    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':130,'savefig.bbox':'tight'})
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.2))
    for i, p in enumerate((50, 100)):
        sub = d[(d.p == p) & d.accepted]
        o = np.concatenate([np.array(s.split(';'), int) for s in sub.occ_sizes])
        h = np.concatenate([np.array(s.split(';'), int) for s in sub.hm_sizes])
        bins = np.arange(0.5, max(o.max(), h.max()) + 2)
        ax[i,0].hist(o, bins=bins, alpha=.65, label='OCC', color='#1f77b4')
        ax[i,0].hist(h, bins=bins, alpha=.65, label='HailMary', color='#d62728')
        ax[i,0].set_yscale('log'); ax[i,0].set_xlabel('cluster size')
        ax[i,0].set_ylabel(f'p={p}\ncount'); ax[i,0].legend(fontsize=7)
        ax[i,0].set_title('size distribution' if i==0 else '', fontsize=10)
        ax[i,1].scatter(sub.occ_largest/p, sub.hm_largest/p, s=42, color='#2ca02c')
        lim=[0,1.02]; ax[i,1].plot(lim,lim,'k--',lw=1)
        ax[i,1].set_xlim(lim); ax[i,1].set_ylim(lim)
        ax[i,1].set_xlabel('OCC largest / p'); ax[i,1].set_ylabel('HailMary largest / p')
        ax[i,1].set_title('the blob' if i==0 else '', fontsize=10)
        ax[i,2].scatter(sub.occ_info, sub.hm_info, s=42, color='#9467bd')
        lim=[0,1.02]; ax[i,2].plot(lim,lim,'k--',lw=1)
        ax[i,2].set_xlim(lim); ax[i,2].set_ylim(lim)
        ax[i,2].set_xlabel('OCC info_eff'); ax[i,2].set_ylabel('HailMary info_eff')
        ax[i,2].set_title('information retained' if i==0 else '', fontsize=10)
    fig.suptitle('What HailMary changes about OCC\'s partition (accepted runs, alpha=0.05)', y=1.01)
    fig.savefig(f'{B}/figures/16_hm_cluster_sizes.png'); plt.close(fig)
    print('\nwrote figures/16_hm_cluster_sizes.png')
    print('\n=== summary ===')
    print(d[d.accepted].groupby(['p','density']).agg(n=('occ_k','size'),
        occ_k=('occ_k','mean'), hm_k=('hm_k','mean'),
        occ_largest=('occ_largest','mean'), hm_largest=('hm_largest','mean'),
        occ_info=('occ_info','mean'), hm_info=('hm_info','mean')).round(2).to_string())
