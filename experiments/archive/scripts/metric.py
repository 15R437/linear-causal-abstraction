"""Does minimising intra-cluster edge mass track recon_loss_data?

Within-instance ladder: from the one-child closure, apply greedy splits one at a time,
recording every intermediate partition. Each rung is CERTIFIED valid (All-In + one-child
+ acyclic) and each split strictly lowers intra-cluster edge mass, so `info` sweeps
downward on a fixed graph and fixed data. Score every rung on:
    info          intra-cluster |edge| mass fraction  (what HailMary optimises)
    recon         Frobenius reconstruction loss       (suspected gameable)
    recon_data    rmse of (I-A_hat')X                 (raw, as losses.py returns it)
    recon_ratio   that rmse / the oracle rmse         (scale-free; losses.py computes
                                                       rmse_oracle then discards it)
Then correlate, within instance, via Spearman.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import fit_abstraction, reconstruct_adj
from src.utils.losses import reconstruction_loss, reconstruction_loss_with_data, graph_loss

_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def candidate_partitions(A, rng, n_orders=20, n_merge=25, max_c=30):
    """A SPREAD of certified-valid partitions for one graph, to sweep `info` within
    instance. The split-ladder alone is degenerate on random DAGs (the closure is already
    at max k), so candidates come from three sources:
      * the one-child closure under many random orders (it is order-dependent),
      * greedy splitting of each (raises k, lowers info),
      * random merges of two clusters (lowers k, raises info),
    every one re-checked against All-In + one-child + acyclicity.
    """
    cands = {}

    def add(g):
        if all(mk.conditions(A, g)):
            cands.setdefault(len(g), [list(x) for x in g])
            return True
        return False

    seeds = []
    for _ in range(n_orders):
        g = mk.closure(A, rng)
        if add(g):
            seeds.append(g)
    for g in seeds[:4]:
        add(mk.refine_by_splitting(A, g, rng))
    for g in seeds[:8]:
        cur = [list(x) for x in g]
        for _ in range(n_merge):
            if len(cur) < 3:
                break
            i, j = rng.choice(len(cur), 2, replace=False)
            m = cur[:min(i, j)] + cur[min(i, j) + 1:max(i, j)] + cur[max(i, j) + 1:]
            m = m + [cur[i] + cur[j]]
            if add(m):
                cur = m
            else:
                break
    ks = sorted(cands)
    if len(ks) > max_c:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, max_c).round().astype(int)]
    return [cands[k] for k in sorted(set(ks))]


def job(a):
    gt, d, p, nh, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=d, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)
    rng = np.random.default_rng(seed)
    parts = candidate_partitions(A, rng)
    if not parts:
        return []
    X = sample_by_noise_adjustment_any_order(A, n_samples=10000, gennorm_param=1.8, seed=seed)
    I = np.eye(p)
    U = (I - A.T) @ X.T                     # oracle residual == the noise
    rmse_oracle = np.sqrt(np.mean(U ** 2))
    null_std = None
    sd = U.std(axis=1); sd[sd < 1e-12] = 1e-12   # per-node oracle residual SD
    null_std = np.sqrt(np.mean((X.T / sd[:, None]) ** 2))   # A_hat = 0 baseline

    out = []
    for r, groups in enumerate(parts):
        T = np.zeros((p, len(groups)))
        for j, m in enumerate(groups):
            T[m, j] = 1
        co = T @ T.T; np.fill_diagonal(co, 0)
        info = (np.abs(A) * co).sum() / np.abs(A).sum()
        try:
            A_high, T_pred = fit_abstraction(A, T, n_restarts=3, seed=seed, maxiter=60)
            A_recon = reconstruct_adj(A_high=A_high, T=T_pred)
            rec = reconstruction_loss(A_low=A, T=T_pred, A_high=A_high, A_hat=A_recon)
            rd = reconstruction_loss_with_data(A_low=A, A_high=A_high, A_hat=A_recon,
                                               X=X, T=T_pred)
            R = (I - A_recon.T) @ X.T
            rd_std = np.sqrt(np.mean((R / sd[:, None]) ** 2))   # pinv A_hat, equal node weight
            with np.errstate(divide='ignore', invalid='ignore'):
                Tn = np.nan_to_num(T_pred / np.sum(np.abs(T_pred), axis=0))
            A_l1 = T_pred @ A_high @ Tn.T          # the L1 construction, far better behaved
            R1 = (I - A_l1.T) @ X.T
            rd_l1 = np.sqrt(np.mean((R1 / sd[:, None]) ** 2))
        except Exception:
            rec, rd, rd_std, rd_l1 = np.nan, np.nan, np.nan, np.nan
        out.append(dict(graph_type=gt, p=p, true_nhigh=nh, seed=seed, rung=r,
                        k=len(groups), info=info, recon=rec, recon_data=rd,
                        recon_ratio=rd / rmse_oracle, recon_std=rd_std, recon_l1=rd_l1, rmse_oracle=rmse_oracle, null_std=null_std,
                        graph_ok=int(graph_loss(A_low=A, T_hard=T) < 1e-8)))
    return out


if __name__ == '__main__':
    # Focused on the RANDOM family, and on the high-n_high cells where the ceiling is
    # small and the closure starts with real information loss -- easy instances give a
    # degenerate one-rung ladder. one_child p=50 kept only as a control.
    jobs = [('random', 0.5, p, nh, s)
            for p in (50, 100, 200) for nh in (10, 20, 40) for s in range(8)
            if nh <= p // 5]
    jobs += [('one_child', 1.0, 50, nh, s) for nh in (5, 10) for s in range(3)]
    print(f'{len(jobs)} instances', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(r)
            if i % 4 == 0:
                print(f'{i}/{len(jobs)}  {(time.time()-t0)/60:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{SP}/metric.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{SP}/metric.csv', index=False)
    print(f'done {(time.time()-t0)/60:.1f} min, {len(rows)} rungs')
