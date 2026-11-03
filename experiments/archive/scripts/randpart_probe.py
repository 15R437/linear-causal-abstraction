"""How do random partitions reach RAE = 0? Node marginalisation + edge cancellation.

Adds the diagnostics fit3cmp.py did not carry: what fraction of T entries are driven to
zero, whether whole clusters are skipped by RAE, and what reconstruction pays for it.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, warnings, time
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp, torch
B = '/Users/issy/Projects/linear-causal-abstraction'
exec(open(f'{B}/experiments/abstraction_study/scripts/fit3cmp.py').read().split("if __name__")[0])
from src.utils.losses import reconstruction_loss, info_loss


def job(a):
    torch.set_num_threads(1)
    kind, p, k, part, trial = a
    inst = gen_instance(kind, p, k, 0.6, trial)
    if inst is None:
        return None
    A_low, T_gt, A_gt = inst
    E_low = np.linalg.inv(np.eye(p) - A_low)
    T_hard = (T_gt != 0).astype(float) if part == 'gt' else \
             random_partition(p, k, np.random.default_rng(10_000 + trial))
    block = block_structure(A_low, T_hard)
    row = dict(kind=kind, p=p, k=k, part=part, trial=trial, block_edges=int(block.sum()),
               info_loss=float(info_loss(A_low=A_low, T_hard=T_hard)),
               # reconstruction of the ZERO predictor, i.e. "no high-level edges at all"
               recon_null=float(np.linalg.norm(A_low, 'fro') / np.linalg.norm(A_low, 'fro')))
    if part == 'gt':
        row['recon_oracle'] = float(reconstruction_loss(A_low=A_low, T=T_gt, A_high=A_gt))
    for name, fn in (('reg', lambda: fit_abstraction(A_low=A_low, T_hard=T_hard, E_low=E_low, n_restarts=10, seed=trial)),
                     ('alt', lambda: fit_abstraction_alternative(A_low=A_low, E_low=E_low, T_hard=T_hard, n_restarts=10, seed=trial)),
                     ('K',   lambda: fit_abstraction_K(A_low=A_low, E_low=E_low, T_hard=T_hard, n_restarts=10, seed=trial))):
        try:
            A_h, T = fn()
        except Exception:
            continue
        S = get_S_local(T, T_hard, A_low)
        rae, nsk = RAE(A_low=A_low, T=T, A_high=A_h, S=S, E_low=E_low,
                       T_hard=T_hard, return_skipped=True)
        nz = T[T_hard != 0]
        # a node is MARGINALISED if its T weight is negligible against its own cluster's scale
        colmax = np.abs(T).max(axis=0); colmax[colmax == 0] = 1.
        rel = np.abs(T) / colmax[None, :]
        marg = ((T_hard != 0) & (rel < 1e-6)).sum()
        absA = np.abs(A_h).copy(); np.fill_diagonal(absA, 0.)
        mx = absA.max() if absA.size else 1.
        row.update({f'{name}_rae': float(rae), f'{name}_exact': int(rae < 1e-6),
                    f'{name}_skipped': int(nsk),
                    f'{name}_marg_frac': float(marg / max(len(nz), 1)),
                    f'{name}_tmin_rel': float((np.abs(nz) / np.abs(nz).max()).min()),
                    f'{name}_recon': float(reconstruction_loss(A_low=A_low, T=T, A_high=A_h)),
                    f'{name}_edges_rel': int((absA > 1e-6 * mx).sum()),
                    f'{name}_recall_rel': float(((absA > 1e-6 * mx) & block).sum() / max(int(block.sum()), 1)),
                    # attenuation: median |A_high| on blocks that DO carry low-level edges
                    f'{name}_med_edge': float(np.median(absA[block]) if block.sum() else np.nan)})
    return row


if __name__ == '__main__':
    jobs = [(kind, p, k, part, t) for kind in ('acyclic', 'cyclic')
            for (p, k) in ((24, 4), (40, 6)) for part in ('gt', 'random') for t in range(15)]
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/randpart_probe.csv', index=False)
    pd.set_option('display.width', 260)
    print(f'{len(d)} instances in {(time.time()-t0)/60:.1f} min\n')
    for m in ('reg', 'alt', 'K'):
        print(f'=== {m} ===')
        print(d.groupby(['kind', 'p', 'part'])[
            [f'{m}_exact', f'{m}_skipped', f'{m}_marg_frac', f'{m}_recall_rel',
             f'{m}_edges_rel', f'{m}_recon', f'{m}_med_edge']].mean().round(3).to_string())
        print()
    print('=== context: block_edges, info_loss, oracle reconstruction ===')
    print(d.groupby(['kind', 'p', 'part'])[['block_edges', 'info_loss', 'recon_oracle']].mean().round(3).to_string())
