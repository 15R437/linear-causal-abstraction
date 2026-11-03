"""Show, concretely, what HailMary changes about OCC's partition on a dense graph.

Prints OCC's clusters, then HailMary's (recon loss, closure init, k=K), then exactly which
nodes moved and what it bought. Dense graphs (d=0.9) are where the refinement helps.

    venv/bin/python .../show_refinement.py --p 40 --density 0.9 --seed 2
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output
from src.utils.losses import observation_loss, intervention_loss
from occ import occ, conditions, high_level_graph

ap = argparse.ArgumentParser()
ap.add_argument('--p', type=int, default=40)
ap.add_argument('--density', type=float, default=0.9)
ap.add_argument('--nhigh', type=int, default=10)
ap.add_argument('--seed', type=int, default=2)
a = ap.parse_args()

A, *_ = sample_low_level_graph(n_high=a.nhigh, n_low=a.p, density=a.density, shuffle=True,
                               blockiness=0.5, heterogeneity=1., internal_density=0.,
                               graph_type='random', seed=a.seed, connected=True,
                               max_attempts=300)
Ab = (A != 0); absA = np.abs(A); tot = absA.sum()
E_low = np.linalg.inv(np.eye(a.p) - A)
print(f'\np={a.p}  density={a.density}  seed={a.seed}   {Ab.sum()} edges, '
      f'max out-degree {Ab.sum(1).max()}')

groups0, T0, ci = occ(A, seed=a.seed)
K = ci['k']

gen = torch.Generator().manual_seed(31337 + a.seed)
li = torch.from_numpy(T0).float() * 5.0 + torch.randn(a.p, K, generator=gen) * 0.1
m = HailMary(n_low=a.p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
m.train(num_epochs=6000, lambda_graph=10., lambda_one_child=100., lambda_info=0.,
        lambda_recon=10., lambda_surj=0., verbose=False)
T1 = m.get_partition()[0].detach().numpy()
groups1 = [g for g in (np.where(T1[:, j] == 1)[0] for j in range(K)) if len(g)]


def report(groups, T, label):
    M = high_level_graph(Ab, groups)
    ai, oc, dag = conditions(A, groups)
    co = T @ T.T; np.fill_diagonal(co, 0)
    intra = (absA * co).sum() / tot
    try:
        S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        obs = float(observation_loss(E_low=E_low, E_high=Eh, S=S, T=Tp))
        intv = float(intervention_loss(E_low=E_low, E_high=Eh, S=S, T=Tp))
        marg = np.where(np.abs(Tp).sum(1) == 0)[0]
        lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        eff = (absA * lost).sum() / tot
    except Exception:
        obs = intv = eff = float('nan'); marg = np.array([], int)
    print(f'\n{label}   k={len(groups)}   All-In={ai} one-child={oc} acyclic={dag}')
    order = sorted(range(len(groups)), key=lambda j: -len(groups[j]))
    for j in order:
        kid = np.where(M[j])[0]
        mem = ', '.join(str(int(x)) for x in sorted(groups[j]))
        if len(mem) > 72:
            mem = mem[:69] + '...'
        print(f'   [{len(groups[j]):3d}] {{{mem}}}' + (f'  -> cluster of size {len(groups[kid[0]])}'
                                                       if len(kid) else '   (sink)'))
    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f'   sizes {sizes}     largest {sizes[0]}/{a.p} = {sizes[0]/a.p:.0%}')
    print(f'   marginalised by fit_abstraction: {len(marg)} nodes'
          + (f' {sorted(marg.tolist())}' if 0 < len(marg) <= 12 else ''))
    print(f'   exact abstraction: {obs < 1e-8 and intv < 1e-8}   (obs {obs:.2e}, intv {intv:.2e})')
    print(f'   information: intra-cluster {intra:.4f}   EFFECTIVE {eff:.4f}')
    return eff, set(map(int, marg))


e0, m0 = report(groups0, T0, 'OCC (closure)')
e1, m1 = report(groups1, T1, 'OCC -> HailMary (recon, k=K)')

lab0 = np.zeros(a.p, int); lab1 = np.zeros(a.p, int)
for j, g in enumerate(groups0):
    lab0[g] = j
for j, g in enumerate(groups1):
    lab1[g] = j
moved = [i for i in range(a.p)
         if set(np.where(lab0 == lab0[i])[0]) != set(np.where(lab1 == lab1[i])[0])]
print(f'\nWHAT CHANGED')
print(f'   {len(moved)} of {a.p} nodes have a different cluster-mate set')
print(f'   effective information loss {e0:.4f} -> {e1:.4f}   ({(e0-e1)/e0:+.1%})')
big0 = max(len(g) for g in groups0); big1 = max(len(g) for g in groups1)
print(f'   largest cluster {big0} -> {big1}      clusters {len(groups0)} -> {len(groups1)}')
if m1 - m0:
    print(f'   newly marginalised: {sorted(m1 - m0)}')
