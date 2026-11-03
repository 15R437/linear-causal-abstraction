"""Does dropping recon_loss bring back edge-less / disconnected high-level DAGs?

Three arms:
  old        lambda_recon=10 with the ORIGINAL hard-count surjective term (no gradient),
             i.e. exactly what the code did before this session.
  recon_off  lambda_recon=0 with the new expected-empty-columns term (the recommendation).
  both       lambda_recon=10 with the new term, to separate "recon" from "surjective".
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output, is_dag
from src.utils.losses import observation_loss, intervention_loss, graph_loss

OUT = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/disconnect.csv'


class OldSurjective(HailMary):
    """The pre-change surjective term: a hard count with no gradient path."""
    def forward(self, temp=1., compute_recon=True):
        recon, graph, one_child, _, entropy = super().forward(temp, compute_recon)
        T_hard, _ = self.get_partition(temp)
        surj = (self.n_high - torch.sum(T_hard.sum(dim=0) > 0)) / self.n_high
        return recon, graph, one_child, surj, entropy


ARMS = {
    'old (recon=10, hard count)':   (OldSurjective, dict(lambda_recon=10., lambda_surj=1.)),
    'new (recon=0, empty cols)':    (HailMary,      dict(lambda_recon=0.,  lambda_surj=1.)),
    'both (recon=10, empty cols)':  (HailMary,      dict(lambda_recon=10., lambda_surj=1.)),
}


def job(args):
    torch.set_num_threads(1)
    arm, trial, n_low, n_high, density = args
    k = n_high
    A_low, T_gt, _, A_high_gt = sample_low_level_graph(
        n_high=k, n_low=n_low, density=density, shuffle=True, blockiness=0.5,
        heterogeneity=1.0, internal_density=0.0, graph_type='one_child', seed=trial)
    cls, kw = ARMS[arm]
    torch.manual_seed(1000 * trial + 7)
    m = cls(n_low=n_low, n_high=k, A_low=torch.from_numpy(A_low).float())
    m.train(num_epochs=1000 * int(np.log((n_low - k) * k + 1) + 1), verbose=False, **kw)
    T_hard = m.get_partition()[0].detach().numpy()

    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    S, T, A_high = process_output(T_hard=T_hard, A_low=A_low, E_low=E_low)
    E_high = np.linalg.inv(np.eye(k) - A_high)
    obs = observation_loss(E_low=E_low, E_high=E_high, S=S, T=T)
    intv = intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T)
    gl = graph_loss(A_low=A_low, T_hard=T_hard)

    mask = np.abs(A_high) > 1e-9
    deg = mask.sum(0) + mask.sum(1)
    used = int((T_hard.sum(0) > 0).sum())
    # Isolation among the clusters that actually have members: an empty cluster is a
    # surjectivity failure, already counted, not a disconnection failure.
    nonempty = T_hard.sum(0) > 0
    iso_nonempty = int(((deg == 0) & nonempty).sum())
    return {'arm': arm, 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': density,
            'edges_high': int(mask.sum()), 'edges_high_gt': int((A_high_gt != 0).sum()),
            'empty_dag': int(mask.sum() == 0),
            'isolated_all': int((deg == 0).sum()),
            'isolated_nonempty': iso_nonempty,
            'any_isolated_nonempty': int(iso_nonempty > 0),
            'clusters_used': used, 'surjective': int(used == k),
            'valid': int((obs < 1e-8) and (intv < 1e-8)), 'dag': int(is_dag(A_high)),
            'graph_loss': float(gl), 'graph_ok': int(gl < 1e-8),
            # The honest bar: admissible, non-degenerate, and structurally sound.
            'all_ok': int((obs < 1e-8) and (intv < 1e-8) and (gl < 1e-8)
                          and used == k and mask.sum() > 0 and iso_nonempty == 0)}


if __name__ == '__main__':
    jobs = [(arm, t, nl, nh, d) for arm in ARMS for t in range(10)
            for (nl, nh) in [(50, 5), (100, 10)] for d in [0.4, 1.0]]
    t0 = time.time()
    with mp.Pool(6) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 240)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min\n")
    g = df.groupby(['density', 'n_low', 'arm'])[
        ['edges_high', 'edges_high_gt', 'empty_dag', 'any_isolated_nonempty',
         'surjective', 'valid', 'graph_ok', 'all_ok', 'graph_loss']].mean()
    print(g.round(3).to_string())
