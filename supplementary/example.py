"""End-to-end example: generate a model, learn a partition, fit T, score it.

    python example.py            # synthetic, acyclic and cyclic
    python example.py leontief   # the Leontief economy in data/

Expected on the synthetic acyclic case: every structurally valid partition reaches
RAE ~ 1e-16, and the partitions are separated by reconstruction error, not by RAE.
On the cyclic case the eigen fitter is exact while the iterative fitters are not.
"""
import sys
import numpy as np

from abstraction import (sample_low_level_graph, generate_random_T,
                         T_concretisation_sampling, occ, allin_agglom,
                         fit_abstraction_eigen, fit_abstraction_K, get_S,
                         RAE, reconstruction_loss)

P, K, SEED = 100, 10, 0


def score(name, A, T_hard, fitter):
    E = np.linalg.inv(np.eye(A.shape[0]) - A)
    if fitter == 'eigen':
        A_high, T = fit_abstraction_eigen(A_low=A, T_hard=T_hard, seed=SEED, n_restarts=5)[:2]
    else:
        A_high, T = fit_abstraction_K(A_low=A, E_low=E, T_hard=T_hard, n_restarts=5,
                                      seed=SEED, maxiter=100, faithful=False)
    S = get_S(T=T, T_hard=T_hard, A_low=A)
    print('  %-16s %-8s RAE %.2e   ReconErr %.4f   ||A_high|| %7.3f'
          % (name, fitter,
             RAE(A_low=A, T=T, A_high=A_high, S=S, E_low=E, T_hard=T_hard),
             reconstruction_loss(A_low=A, T=T, A_high=A_high),
             np.linalg.norm(A_high)))


def cyclic_high_level(k, density, rho, rng):
    """A high-level graph guaranteed to contain a cycle, rescaled to spectral radius rho."""
    for _ in range(500):
        M = (rng.random((k, k)) < density).astype(float)
        np.fill_diagonal(M, 0.)
        A = M * rng.uniform(0.5, 2.0, (k, k)) * rng.choice([-1, 1], (k, k))
        r = np.max(np.abs(np.linalg.eigvals(A)))
        if r < 1e-9:
            continue
        R = (A != 0); Rc = R.copy()
        for _ in range(k):
            Rc = Rc | (Rc @ R)
        if np.any(np.diag(Rc)):
            return A * (rho / r)
    raise RuntimeError('no cyclic draw found')


def synthetic():
    print('ACYCLIC quotient  (p=%d, k=%d, density 0.5, blockiness 1)' % (P, K))
    A, T_gt, S_gt, A_high_gt = sample_low_level_graph(
        n_high=K, n_low=P, density=0.5, shuffle=True, blockiness=1.0, heterogeneity=0.,
        internal_density=0., graph_type='random', seed=SEED, connected=True, max_attempts=400)
    print('  %-16s %-8s RAE %.2e   ReconErr %.4f   ||A_high|| %7.3f   <- oracle'
          % ('ground truth', '-',
             RAE(A_low=A, T=T_gt, A_high=A_high_gt, S=S_gt,
                 T_hard=(T_gt != 0).astype(float)),
             reconstruction_loss(A_low=A, T=T_gt, A_high=A_high_gt),
             np.linalg.norm(A_high_gt)))
    for nm, Th in (('GT partition', (T_gt != 0).astype(float)),
                   ('AOC Closure', occ(A, seed=SEED, k=K)[1]),
                   ('Greedy All-In', allin_agglom(A, K))):
        score(nm, A, Th, 'eigen')

    print('\nCYCLIC quotient   (same size; cycles placed in the high-level graph)')
    rng = np.random.default_rng(10_000 + SEED)
    A_high_c = cyclic_high_level(K, 0.5, 0.7, rng)
    T_c, _ = generate_random_T(P, K, (1., -2.), (2., -1.), seed=SEED)
    A_c, _ = T_concretisation_sampling(A_high=A_high_c, T=T_c, internal_density=0.,
                                       blockiness=1.0, heterogeneity=0., seed=SEED)
    Th_c = (T_c != 0).astype(float)
    for fitter in ('eigen', 'K'):
        score('GT partition', A_c, Th_c, fitter)


def leontief(thr=0.05, k=5):
    A0 = np.load('data/leontief_A_low.npy')
    A = np.where(np.abs(A0) > thr, A0, 0.0)
    print('LEONTIEF economy  (%d sectors, threshold %.2f, density %.3f)'
          % (A.shape[0], thr, (A != 0).mean()))
    for nm, Th in (('AOC Closure', occ(A, seed=SEED, k=k)[1]),
                   ('Greedy All-In', allin_agglom(A, k))):
        score(nm, A, Th, 'eigen')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'leontief':
        leontief()
    else:
        synthetic()
