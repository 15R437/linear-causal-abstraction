"""When the learned high-level DAG is edge-less, was the low-level graph actually
disconnected -- or are real cross-cluster edges being lost in post-processing?"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output, high_level_mask
from src.utils.losses import observation_loss, intervention_loss, graph_loss

OUT = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/connectivity.csv'


def job(args):
    torch.set_num_threads(1)
    trial, n_low, n_high, density, lrec = args
    k = n_high
    A_low, T_gt, _, A_high_gt = sample_low_level_graph(
        n_high=k, n_low=n_low, density=density, shuffle=True, blockiness=0.5,
        heterogeneity=1.0, internal_density=0.0, graph_type='one_child', seed=trial)
    torch.manual_seed(1000 * trial + 7)
    m = HailMary(n_low=n_low, n_high=k, A_low=torch.from_numpy(A_low).float())
    m.train(num_epochs=1000 * int(np.log((n_low - k) * k + 1) + 1),
            lambda_recon=lrec, verbose=False)
    T_hard = m.get_partition()[0].detach().numpy()

    Ab = (A_low != 0).astype(float)
    # Weak connectivity of the low-level graph.
    n_comp, comp = connected_components(sp.csr_matrix(Ab), directed=True, connection='weak')
    # Components that actually contain an edge (a lone node is its own component).
    sizes = np.bincount(comp)
    n_comp_nontrivial = int((sizes > 1).sum())
    n_isolated_nodes = int((sizes == 1).sum())

    # Cross-cluster low-level edges under the LEARNED partition.
    cross = T_hard.T @ Ab @ T_hard
    np.fill_diagonal(cross, 0)
    n_cross_edges = float(cross.sum())

    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    S, T, A_high = process_output(T_hard=T_hard, A_low=A_low, E_low=E_low)
    obs = observation_loss(E_low=E_low, E_high=np.linalg.inv(np.eye(k) - A_high), S=S, T=T)
    intv = intervention_loss(E_low=E_low, E_high=np.linalg.inv(np.eye(k) - A_high), S=S, T=T)
    gl = graph_loss(A_low=A_low, T_hard=T_hard)

    edges_high = int((np.abs(A_high) > 1e-9).sum())
    # What the structural mask says BEFORE the abstraction fit, for comparison.
    try:
        mask_edges = int(high_level_mask(A_low=A_low, T_hard=T_hard).sum())
    except Exception:
        mask_edges = -1

    used = int((T_hard.sum(0) > 0).sum())
    return {'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': density,
            'lambda_recon': lrec,
            'low_edges': int(Ab.sum()),
            'n_weak_components': int(n_comp),
            'n_nontrivial_components': n_comp_nontrivial,
            'n_isolated_low_nodes': n_isolated_nodes,
            'cross_cluster_edges': n_cross_edges,
            'edges_high_fitted': edges_high,
            'edges_high_mask': mask_edges,
            'edges_high_gt': int((A_high_gt != 0).sum()),
            'empty_dag': int(edges_high == 0),
            'surjective': int(used == k),
            'valid': int((obs < 1e-8) and (intv < 1e-8)),
            'graph_ok': int(gl < 1e-8)}


if __name__ == '__main__':
    jobs = [(t, nl, nh, d, lr) for t in range(10) for (nl, nh) in [(50, 5), (100, 10)]
            for d in [0.4, 1.0] for lr in [0., 10.]]
    t0 = time.time()
    with mp.Pool(6) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min\n")

    print("=== the decisive question: when the fitted high-level DAG is edge-less, "
          "are there cross-cluster low-level edges? ===")
    e = df[df.empty_dag == 1]
    print(f"edge-less runs: {len(e)} of {len(df)}")
    if len(e):
        print(e[['n_low', 'density', 'lambda_recon', 'trial', 'low_edges',
                 'n_nontrivial_components', 'n_isolated_low_nodes', 'cross_cluster_edges',
                 'edges_high_mask', 'surjective', 'valid', 'graph_ok']].to_string(index=False))
        print()
        print(f"  edge-less AND zero cross-cluster edges (genuinely disconnected): "
              f"{int((e.cross_cluster_edges == 0).sum())}/{len(e)}")
        print(f"  edge-less DESPITE cross-cluster edges (post-processing lost them): "
              f"{int((e.cross_cluster_edges > 0).sum())}/{len(e)}")
    print()
    print("=== low-level connectivity by regime (independent of any run) ===")
    print(df.groupby(['density', 'n_low'])[
        ['low_edges', 'n_weak_components', 'n_nontrivial_components',
         'n_isolated_low_nodes', 'edges_high_gt']].mean().round(2).to_string())
