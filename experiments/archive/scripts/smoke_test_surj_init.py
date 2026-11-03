"""Smoke test: does HailMary do better with a surjective loss in place of the
reconstruction loss, and does the initialisation strategy matter?

Two sweeps, both paired on the seed (every config sees the same A_low per trial):

  A. surjective loss   -- lambda_recon = 0 and a surjective penalty switched on,
                          across several formulations and several lambda_surj.
                          Controls: the current default (recon on, no real surjective
                          pressure) and recon = 0 with no surjective term at all.
  B. initialisation    -- the default loss weights, varying only the initial logits.

  C. interaction       -- the surjective variants that survived A, crossed with the
                          initialisations that struggled in B (built from A/B results).

Run:  venv/bin/python experiments/abstraction_study/scripts/smoke_test_surj_init.py
"""
import itertools
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output, is_dag
from src.utils.losses import (abstraction_residual, graph_loss, intervention_loss,
                              observation_loss, rand_index, reconstruction_loss)

# ----------------------------------------------------------------------------
# Surjective loss formulations
# ----------------------------------------------------------------------------
# Each returns a scalar that is 0 when every high-level column is claimed by at
# least one low-level node, and grows as columns are abandoned. They are scaled to
# roughly O(1) so that a given lambda_surj means something comparable across them.

def surjective_loss(mode, T_soft, T_hard, n_high):
    if mode == 'none':
        return torch.zeros((), device=T_soft.device)

    if mode == 'hard_count':
        # What the model currently does: count of unclaimed columns. A boolean
        # comparison sits between the parameters and the loss, so this has no gradient.
        return (n_high - torch.sum(T_hard.sum(dim=0) > 0)) / n_high

    if mode == 'hinge_max':
        # Every column needs at least one node assigning it > 0.5 mass.
        return F.relu(0.5 - T_soft.max(dim=0).values).mean()

    if mode == 'topk_barrier':
        # The variant last used in this repo (see git history): a log-barrier gated
        # by how far the two best claimants of a column are from a majority.
        top_k, _ = torch.topk(T_soft, k=min(2, T_soft.shape[0]), dim=0)
        col_deficit = F.relu(0.5 - top_k.mean(dim=0))
        return (col_deficit * (-torch.log(T_soft + 1e-8).mean(dim=0))).sum()

    if mode == 'marginal_entropy':
        # KL(column marginal || uniform): 0 iff clusters are equally loaded.
        p = T_soft.mean(dim=0)
        return (p * torch.log(p + 1e-8)).sum() + math.log(n_high)

    if mode == 'empty_prob':
        # Expected number of empty columns under an independence approximation,
        # computed in log space so the product does not underflow.
        log_empty = torch.log1p(-T_soft.clamp(max=1.0 - 1e-6)).sum(dim=0)
        return torch.exp(log_empty).sum() / n_high

    if mode == 'colmass_l2':
        # Squared deviation of the column mass from uniform, scaled to [0, 1]:
        # 0 when every cluster carries 1/k of the mass, 1 when one cluster carries it all.
        q = T_soft.mean(dim=0)
        return ((q - 1.0 / n_high) ** 2).sum() / (1.0 - 1.0 / n_high)

    if mode == 'max_share':
        # Penalise only the largest cluster's share. Blunter than colmass_l2 -- the gradient
        # touches one column at a time -- but it targets the blob directly.
        q = T_soft.mean(dim=0)
        return (q.max() - 1.0 / n_high) / (1.0 - 1.0 / n_high)

    if mode == 'hinge_colsum':
        # Soft "at least one member per cluster" on expected cluster sizes.
        return F.relu(1.0 - T_soft.sum(dim=0)).sum() / n_high

    raise ValueError(f"Unknown surjective mode: {mode}")


SURJ_MODES = ['none', 'hard_count', 'hinge_max', 'topk_barrier',
              'marginal_entropy', 'empty_prob', 'hinge_colsum',
              'colmass_l2', 'max_share']


class HailMaryVariant(HailMary):
    """HailMary with a swappable surjective term. Everything else is inherited, so
    the reconstruction/graph/one-child terms are exactly the current implementation."""

    def __init__(self, *args, surj_mode='hard_count', info_mode='recon', norm='old',
                 intra_scale='plain', **kwargs):
        self.surj_mode = surj_mode
        # 'recon'      - the existing in-edge All-In term (what lambda_recon weighted)
        # 'intra_mass' - fraction of |A_low| edge mass hidden inside clusters, i.e. the
        #                edge mass the abstraction destroys. Occupies the same slot, so
        #                lambda_recon weights whichever is selected.
        # 'none'       - nothing in that slot
        self.info_mode = info_mode
        # 'old' - graph_loss averaged over all p^2 entries, one_child as sum(H)^2 - sum(H^2).
        #         Both drift with problem size: graph shrinks ~5x from p=50 to p=200 while
        #         one_child grows ~3x, so their balance moves 15x and no single lambda pair
        #         transfers across sizes.
        # 'new'  - graph_loss per co-assigned pair (flat in p), one_child as an inverse
        #         participation ratio (0 when a node feeds exactly one cluster). Balance
        #         drifts ~1.9x instead.
        self.norm = norm
        self.intra_scale = intra_scale
        super().__init__(*args, **kwargs)

    def forward(self, temp=1., compute_recon=True):
        need_recon = compute_recon and self.info_mode in ('recon', 'recon_const')
        # HailMary.forward returns a 6th value (its own info term) since the overnight
        # change; the harness supplies its own in the recon slot, so that one is discarded
        # here and lambda_info is pinned to 0 below.
        out = super().forward(temp, need_recon)
        recon, graph, one_child, surj, entropy = out[:5]
        T_hard, T_soft = self.get_partition(temp)
        # The base class now computes expected-empty-columns; override it for every other
        # variant so all arms go through one code path.
        if self.surj_mode != 'empty_prob':
            surj = surjective_loss(self.surj_mode, T_soft, T_hard, self.n_high)
        if self.norm == 'graph_const':
            # Same scale correction as dividing by the co-assignment mass, but with a
            # CONSTANT denominator p^2/k instead of one that depends on T_soft. Dividing
            # by the live co-assignment mass lets the model shrink the loss by growing
            # clusters -- a blob incentive -- because the denominator counts all
            # co-assigned pairs while the numerator counts only violating ones.
            # sum/(p^2/k) == k * (sum/p^2) == k * the current mean.
            graph = graph * self.n_high

        if self.norm == 'new':
            T_h2, T_s2 = T_hard, T_soft
            Ab = self.A_low_binary
            co = T_s2 @ T_s2.T
            ew = self.A_low.abs() @ T_s2
            dm = 1.0 - torch.eye(self.n_low, device=self.device)
            H = torch.max(Ab[:, :, None, :] * T_s2.T[None, None, :, :], dim=-1)[0] * (1 - T_s2)
            Hs = torch.clamp(H, 1e-6, 1 - 1e-6)
            Hl = torch.logit(Hs)
            ti = Hs.detach()[:, :, None, :]
            lj = Hl[:, None, :, :]
            ce = ti * F.softplus(-lj) + (1.0 - ti) * F.softplus(lj)
            Hp = (ce * ew[:, None, :, :]).sum(dim=-1)
            m = co * dm
            graph = (m * Hp).sum() / m.sum().clamp(min=1.0)
            s1 = torch.sum(H, dim=-1)
            s2 = torch.sum(H ** 2, dim=-1)
            active = (s1 > 1e-3).float()
            one_child = ((active * (1.0 - s2 / (s1 ** 2).clamp(min=1e-8))).sum()
                         / active.sum().clamp(min=1.0))

        if self.info_mode == 'intra_mass':
            # intra_mass sits in [0, ~1/k]: a random balanced partition scores 1/k and the
            # generating partition scores 0, so the term's whole dynamic range shrinks with k
            # while being independent of p. Multiplying by k restores a constant range so one
            # lambda transfers across k.
            W = self.A_low.abs()                                   # (batch, n_low, n_low)
            co = T_soft @ T_soft.T
            diag = 1.0 - torch.eye(self.n_low, device=self.device)
            recon = (W * (co * diag)[None, :, :]).sum() / W.sum().clamp(min=1e-12)
            if self.intra_scale == 'times_k':
                recon = recon * self.n_high
        elif self.info_mode == 'recon_const':
            # The in-edge All-In term, renormalised the same way as graph_loss: its .mean()
            # also divides by p^2 while only co-assigned pairs contribute, so multiplying by
            # k restores a per-co-assigned-pair scale.
            recon = recon * self.n_high
        elif self.info_mode == 'none':
            recon = torch.zeros((), device=self.device)
        if len(out) > 5:
            return recon, graph, one_child, surj, entropy, torch.zeros((), device=self.device)
        return recon, graph, one_child, surj, entropy


# ----------------------------------------------------------------------------
# Initialisation strategies
# ----------------------------------------------------------------------------

def one_child_closure(A_low, rng=None):
    """Finest partition satisfying the one-child condition: start from singletons and
    repeatedly merge the children of any cluster that has more than one. ~1ms at p=400.

    NOTE this enforces one-child only, NOT All-In -- a merged cluster holding a sink
    alongside non-sinks inherits an outgoing edge the sink does not participate in.
    It is also order-dependent, which is why each restart can draw a different one.
    """
    p = A_low.shape[0]
    Ab = (A_low != 0)
    lab = list(range(p))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            lab[ry] = rx

    changed = True
    while changed:
        changed = False
        groups = {}
        for i in range(p):
            groups.setdefault(find(i), []).append(i)
        roots = list(groups)
        if rng is not None:
            rng.shuffle(roots)
        for r in roots:
            mem = groups.get(r)
            if mem is None:
                continue
            kids = {find(j) for i in mem for j in np.where(Ab[i])[0] if find(j) != find(r)}
            if len(kids) > 1:
                kl = sorted(kids)
                if rng is not None:
                    rng.shuffle(kl)
                for o in kl[1:]:
                    union(kl[0], o)
                changed = True
    out = {}
    for i in range(p):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def coarsen_to_k(groups, A_low, k):
    """Reduce a partition to exactly k clusters by folding the smallest cluster into the
    cluster it feeds (preserving the chain where possible), else into the largest."""
    groups = [list(g) for g in groups]
    Ab = (A_low != 0)
    while len(groups) > k:
        order = sorted(range(len(groups)), key=lambda j: len(groups[j]))
        src = order[0]
        child = None
        for j in range(len(groups)):
            if j != src and Ab[np.ix_(groups[src], groups[j])].any():
                child = j
                break
        if child is None:
            child = max((j for j in range(len(groups)) if j != src), key=lambda j: len(groups[j]))
        groups[child] = groups[child] + groups[src]
        groups.pop(src)
    return groups

def make_logits(strategy, n_low, n_high, generator):
    """All strategies return an (n_low, n_high) logit matrix."""
    if strategy == 'reg_surjective':
        # Current default: low-variance noise plus a margin injected at randomly
        # chosen targets that cover every column at least once.
        logits = torch.randn(n_low, n_high, generator=generator) * 0.1
        targets = torch.zeros(n_low, dtype=torch.long)
        targets[:n_high] = torch.arange(n_high)
        if n_low > n_high:
            targets[n_high:] = torch.randint(0, n_high, (n_low - n_high,), generator=generator)
        targets = targets[torch.randperm(n_low, generator=generator)]
        margin = math.log(max(1, n_high - 1)) + 0.5 + 0.1
        logits[torch.arange(n_low), targets] += margin
        return logits

    if strategy == 'random_normal':
        # Plain unstructured random logits: no coverage guarantee.
        return torch.randn(n_low, n_high, generator=generator)

    if strategy == 'random_hard':
        # Random hard assignment (again no coverage guarantee), same margin as
        # reg_surjective so the only difference is the guaranteed cover.
        logits = torch.randn(n_low, n_high, generator=generator) * 0.1
        targets = torch.randint(0, n_high, (n_low,), generator=generator)
        margin = math.log(max(1, n_high - 1)) + 0.5 + 0.1
        logits[torch.arange(n_low), targets] += margin
        return logits

    if strategy == 'uniform':
        # Exactly uniform softmax. Ties in argmax break to column 0 for every row.
        return torch.zeros(n_low, n_high)

    if strategy == 'near_uniform':
        # Uniform up to tiny noise, so argmax ties are broken randomly.
        return torch.randn(n_low, n_high, generator=generator) * 1e-2

    if strategy.startswith('biased_'):
        # Everyone starts in cluster 0 with probability p.
        p = float(strategy.split('_')[1])
        logits = torch.randn(n_low, n_high, generator=generator) * 0.1
        logits[:, 0] += math.log(p * (n_high - 1) / (1 - p))
        return logits

    raise ValueError(f"Unknown init strategy: {strategy}")


INIT_STRATEGIES = ['reg_surjective', 'random_normal', 'random_hard', 'uniform',
                   'near_uniform', 'biased_0.5', 'biased_0.8', 'biased_0.95']


# ----------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------

DEFAULT_JOB = {
    'sweep': 'X', 'trial': 0, 'n_low': 50, 'n_high': 5, 'density': 1.0,
    'graph_type': 'one_child', 'init': 'reg_surjective', 'surj_mode': 'empty_prob',
    'lambda_surj': 10.0, 'lambda_recon': 0.0,
    'lambda_graph': 10.0, 'lambda_one_child': 100.0, 'info_mode': 'recon', 'norm': 'old', 'intra_scale': 'plain', 'init_seed': None,
   'restart': 0, 'epochs': None, 'arm_name': '',
}


def run_one(job):
    """Train one model on one graph and score it. Returns a flat dict of metrics."""
    torch.set_num_threads(1)
    # A dict rather than a positional tuple: the sweeps kept growing new factors, and
    # every one of them meant touching every job builder.
    cfg = dict(DEFAULT_JOB)
    cfg.update(job)
    sweep, trial = cfg['sweep'], cfg['trial']
    n_low, n_high, density = cfg['n_low'], cfg['n_high'], cfg['density']
    graph_type, init_strategy, surj_mode = cfg['graph_type'], cfg['init'], cfg['surj_mode']
    lambda_surj, lambda_recon = cfg['lambda_surj'], cfg['lambda_recon']
    lambda_graph, lambda_one_child = cfg['lambda_graph'], cfg['lambda_one_child']
    k = n_high

    # connected=True: a disconnected draw is two unrelated systems sharing an index, and
    # an edge-less high-level DAG is the correct abstraction of it, so such draws cannot
    # score structure recovery. For one_child this is only satisfiable at density 1.0.
    A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(
        n_high=n_high, n_low=n_low, density=density, shuffle=True, blockiness=0.5,
        heterogeneity=1.0, internal_density=0.0, graph_type=graph_type, seed=trial,
        connected=True, max_attempts=200)

    init_seed = cfg['init_seed'] if cfg['init_seed'] is not None else 1000 * trial + 7
    generator = torch.Generator().manual_seed(init_seed)
    if init_strategy == 'closure':
        g = one_child_closure(A_low, rng=np.random.default_rng(init_seed))
        g = coarsen_to_k(g, A_low, k) if len(g) > k else g
        T0 = np.zeros((n_low, k))
        for j, mem in enumerate(g[:k]):
            T0[mem, j] = 1.0
        logits_init = (torch.from_numpy(T0).float() * 5.0
                       + torch.randn(n_low, k, generator=generator) * 0.1)
    else:
        logits_init = make_logits(init_strategy, n_low, k, generator)

    model = HailMaryVariant(n_low=n_low, n_high=k, A_low=torch.from_numpy(A_low).float(),
                            logits_init=logits_init, surj_mode=surj_mode,
                            info_mode=cfg['info_mode'], norm=cfg['norm'],
                            intra_scale=cfg['intra_scale'])
    num_epochs = cfg['epochs'] if cfg['epochs'] is not None else 1000 * int(np.log((n_low - k) * k + 1) + 1)

    t0 = time.perf_counter()
    if num_epochs > 0:
        try:
            model.train(num_epochs=num_epochs, lambda_recon=lambda_recon,
                        lambda_graph=lambda_graph, lambda_one_child=lambda_one_child,
                        lambda_surj=lambda_surj, lambda_info=0.0, verbose=False)
        except TypeError:          # pre-overnight HailMary has no lambda_info
            model.train(num_epochs=num_epochs, lambda_recon=lambda_recon,
                        lambda_graph=lambda_graph, lambda_one_child=lambda_one_child,
                        lambda_surj=lambda_surj, verbose=False)
    runtime = time.perf_counter() - t0

    T_hard = model.get_partition()[0].detach().cpu().numpy()
    n_used = int((T_hard.sum(axis=0) > 0).sum())

    row = {'sweep': sweep, 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'k': k,
           'density': density, 'graph_type': graph_type,
           'init': init_strategy, 'surj_mode': surj_mode,
           'lambda_surj': lambda_surj, 'lambda_recon': lambda_recon,
           'lambda_graph': lambda_graph, 'lambda_one_child': lambda_one_child,
           'info_mode': cfg['info_mode'], 'norm': cfg['norm'],
           'intra_scale': cfg['intra_scale'], 'restart': cfg['restart'],
           'epochs': num_epochs, 'arm_name': cfg['arm_name'], 'runtime': runtime,
           # Instance difficulty: these generators produce wildly varying graphs at a
           # fixed density, so record what was actually drawn.
           'n_edges_low': int((A_low != 0).sum()),
           'n_edges_high': int((A_high_gt != 0).sum()),
           'min_block': int((T_gt != 0).sum(axis=0).min()),
           'n_clusters_used': n_used, 'coverage': n_used / k,
           # Degeneracy in the other direction: surjectivity is satisfied by singletons,
           # which also make the All-In constraint vacuous. Compare against the generating
           # partition's own size spread rather than assuming a unique correct answer.
           'max_cluster': int(T_hard.sum(axis=0).max()),
           'min_cluster': int(T_hard.sum(axis=0).min()),
           'n_singletons': int((T_hard.sum(axis=0) == 1).sum()),
           'blob_frac': float(T_hard.sum(axis=0).max() / n_low),
           'gt_max_cluster': int((np.abs(T_gt) > 0).sum(axis=0).max()),
           'gt_blob_frac': float((np.abs(T_gt) > 0).sum(axis=0).max() / n_low),
           'size_entropy': float(-(lambda q: (q * np.log(q + 1e-12)).sum())(
               T_hard.sum(axis=0) / n_low) / np.log(k)),
           'surjective': int(n_used == k),
           'rand_index': rand_index(T_gt=T_gt, T_pred=T_hard),
           'graph_loss': graph_loss(A_low=A_low, T_hard=T_hard),
           # Edge mass destroyed by collapsing clusters (0 for the generating partition).
           'intra_mass': float((np.abs(A_low) * (T_hard @ T_hard.T
                                                 * (1 - np.eye(n_low)))).sum()
                               / max(np.abs(A_low).sum(), 1e-12)),
           # How much of A_low survives the round trip through the high-level graph.
           'block_pred_err': float(
               np.abs((A_low != 0).astype(float)
                      - T_hard @ (((T_hard.T @ (A_low != 0).astype(float) @ T_hard) > 0)
                                  .astype(float) * (1 - np.eye(k))) @ T_hard.T).sum()
               / max((A_low != 0).sum(), 1e-12))}

    # Post-processing needs every cluster populated; a collapsed partition is scored
    # as a failure rather than crashing the sweep.
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    try:
        S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard, A_low=A_low, E_low=E_low)
        E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)
        obs = observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
        intv = intervention_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
        edge_mask = np.abs(A_high_pred) > 1e-9
        deg = edge_mask.sum(0) + edge_mask.sum(1)
        row.update({'obs_loss': obs, 'intv_loss': intv,
                    'edges_high': int(edge_mask.sum()),
                    'edges_high_gt': int((A_high_gt != 0).sum()),
                    'empty_high_dag': int(edge_mask.sum() == 0),
                    'isolated_nonempty': int(((deg == 0) & (T_hard.sum(0) > 0)).sum()),
                    # The other sufficient condition: at most one child per cluster.
                    'max_children': int(edge_mask.sum(axis=1).max()) if k else 0,
                    'one_child_ok': int(edge_mask.sum(axis=1).max() <= 1) if k else 1,
                    'abstraction_residual': abstraction_residual(A_low=A_low, T=T_pred, A_high=A_high_pred),
                    'recon_loss': reconstruction_loss(A_low=A_low, T=T_pred, A_high=A_high_pred),
                    'dag': int(is_dag(A_high_pred)),
                    'exact': int((obs < 1e-8) and (intv < 1e-8)),
                    'postprocess_failed': 0})
    except Exception:
        row.update({'obs_loss': np.nan, 'intv_loss': np.nan, 'abstraction_residual': np.nan,
                    'recon_loss': np.nan, 'dag': 0, 'exact': 0, 'postprocess_failed': 1,
                    'edges_high': -1, 'edges_high_gt': int((A_high_gt != 0).sum()),
                    'empty_high_dag': -1, 'isolated_nonempty': -1,
                    'max_children': -1, 'one_child_ok': -1})

    return row


# ----------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------

def sweep_a_jobs(trials, sizes, densities, lambdas):
    """Surjective loss formulations, with the reconstruction loss switched off."""
    jobs = []
    for trial, (n_low, n_high), d in itertools.product(range(trials), sizes, densities):
        # Controls: the current default, then recon-only, then neither term.
        jobs.append({'sweep': 'A', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': 'reg_surjective', 'surj_mode': 'hard_count', 'lambda_surj': 1.0, 'lambda_recon': 10.0, 'graph_type': 'one_child'})
        jobs.append({'sweep': 'A', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': 'reg_surjective', 'surj_mode': 'none', 'lambda_surj': 0.0, 'lambda_recon': 10.0, 'graph_type': 'one_child'})
        jobs.append({'sweep': 'A', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': 'reg_surjective', 'surj_mode': 'none', 'lambda_surj': 0.0, 'lambda_recon': 0.0, 'graph_type': 'one_child'})
        for mode in SURJ_MODES:
            if mode == 'none':
                continue
            for lam in lambdas:
                jobs.append({'sweep': 'A', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': 'reg_surjective', 'surj_mode': mode, 'lambda_surj': lam, 'lambda_recon': 0.0, 'graph_type': 'one_child'})
    return jobs


def sweep_b_jobs(trials, sizes, densities):
    """Initialisation strategies under the current default loss weights."""
    return [{'sweep': 'B', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': init, 'surj_mode': 'hard_count', 'lambda_surj': 1.0, 'lambda_recon': 10.0, 'graph_type': 'one_child'}
            for trial, (n_low, n_high), d, init
            in itertools.product(range(trials), sizes, densities, INIT_STRATEGIES)]


def sweep_c_jobs(trials, sizes, densities, arms, inits):
    """Cross the winning loss configurations with the initialisations, to see whether
    the recommendation from sweep A survives a bad starting point.

    arms: list of (surj_mode, lambda_surj, lambda_recon).
    """
    return [{'sweep': 'C', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': init, 'surj_mode': mode, 'lambda_surj': lam, 'lambda_recon': lrec, 'graph_type': 'one_child'}
            for trial, (n_low, n_high), d, init, (mode, lam, lrec)
            in itertools.product(range(trials), sizes, densities, inits, arms)]


# one_child can only be connected at density 1.0 (a node has at most one child, so the graph
# has at most k-1 edges and connectivity needs all of them). random DAGs reach connectivity
# comfortably at 0.5, and there density really is a sparsity knob.
GRAPH_FAMILIES = [('one_child', 1.0), ('random', 0.5)]


def sweep_d_jobs(trials, sizes, arms, families=GRAPH_FAMILIES):
    """The recon question, on connected graphs. arms: (surj_mode, lambda_surj, lambda_recon)."""
    return [{'sweep': 'D', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d, 'init': 'reg_surjective', 'surj_mode': mode, 'lambda_surj': lam, 'lambda_recon': lrec, 'graph_type': gt}
            for trial, (n_low, n_high), (gt, d), (mode, lam, lrec)
            in itertools.product(range(trials), sizes, families, arms)]


D_ARMS = [
    ('hard_count', 1.0, 10.0),      # the original code, verbatim
    ('none', 0.0, 10.0),            # recon only
    ('none', 0.0, 0.0),             # neither
    ('empty_prob', 1.0, 0.0),
    ('empty_prob', 10.0, 0.0),
    ('empty_prob', 100.0, 0.0),
    ('topk_barrier', 10.0, 0.0),
    ('hinge_max', 10.0, 0.0),
    ('empty_prob', 1.0, 10.0),      # keep recon AND fix the surjective term
    ('empty_prob', 10.0, 10.0),
]


def sweep_e_jobs(trials, sizes, arms, families=GRAPH_FAMILIES):
    """All-In (lambda_graph) against one-child (lambda_one_child): the two sufficient
    conditions, and how the weight split between them affects each.
    arms: (lambda_graph, lambda_one_child)."""
    return [{'sweep': 'E', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'lambda_graph': lg, 'lambda_one_child': loc}
            for trial, (n_low, n_high), (gt, d), (lg, loc)
            in itertools.product(range(trials), sizes, families, arms)]


E_ARMS = [(10.0, 0.0), (10.0, 1.0), (10.0, 10.0), (10.0, 100.0), (10.0, 1000.0),
          (100.0, 10.0), (100.0, 100.0)]


def sweep_f_jobs(trials, sizes, families=GRAPH_FAMILIES):
    """Does the settled configuration hold at a size it was not chosen on?
    Only the original configuration and the settled one, so the comparison is cheap
    enough to run at p=200/k=20."""
    arms = [
        # the code as it stood before any of this
        {'surj_mode': 'hard_count', 'lambda_surj': 1.0, 'lambda_recon': 10.0,
         'lambda_graph': 10.0, 'lambda_one_child': 100.0},
        # what sweeps D and E settled on
        {'surj_mode': 'empty_prob', 'lambda_surj': 10.0, 'lambda_recon': 0.0,
         'lambda_graph': 10.0, 'lambda_one_child': 100.0},
    ]
    return [dict({'sweep': 'F', 'trial': trial, 'n_low': n_low, 'n_high': n_high,
                  'density': d, 'graph_type': gt}, **arm)
            for trial, (n_low, n_high), (gt, d), arm
            in itertools.product(range(trials), sizes, families, arms)]


G_ARMS = ([('none', 0.0)] +
           [('empty_prob', 10.0)] +
           [(m, l) for m in ('marginal_entropy', 'colmass_l2', 'max_share')
            for l in (0.1, 1.0, 10.0, 100.0)])


def sweep_g_jobs(trials, sizes, arms=G_ARMS, families=GRAPH_FAMILIES):
    """Cluster-balance terms against the coverage-only term, with recon off.
    Balance is a strictly stronger ask than surjectivity: it rules out the blob as well
    as the empty cluster, so the question is whether it fights valid abstractions."""
    return [{'sweep': 'G', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': mode, 'lambda_surj': lam, 'lambda_recon': 0.0}
            for trial, (n_low, n_high), (gt, d), (mode, lam)
            in itertools.product(range(trials), sizes, families, arms)]


H_ARMS = [
    ('none', 0.0),          # the sweep-D settled config: surjective term only
    ('recon', 10.0),        # the original in-edge All-In term, for reference
    ('intra_mass', 0.1),
    ('intra_mass', 1.0),
    ('intra_mass', 10.0),
    ('intra_mass', 100.0),
]


def sweep_h_jobs(trials, sizes, arms=H_ARMS, families=GRAPH_FAMILIES):
    """Does penalising destroyed edge mass fight the All-In condition?
    Surjective term held at empty_prob lambda=10 throughout."""
    return [{'sweep': 'H', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': 10.0,
             'info_mode': mode, 'lambda_recon': lam}
            for trial, (n_low, n_high), (gt, d), (mode, lam)
            in itertools.product(range(trials), sizes, families, arms)]


I_ARMS = [(loc, im) for loc in (100.0, 0.0) for im in (0.0, 100.0)]


def sweep_i_jobs(trials, sizes, arms=I_ARMS, families=(('random', 0.5),)):
    """On random DAGs the generating partition is not one-child, so the tree term makes
    blob-plus-singletons the canonical solution. Does dropping it recover real structure?"""
    return [{'sweep': 'I', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': 10.0,
             'lambda_one_child': loc,
             'info_mode': 'intra_mass' if im > 0 else 'none', 'lambda_recon': im}
            for trial, (n_low, n_high), (gt, d), (loc, im)
            in itertools.product(range(trials), sizes, families, arms)]


J_ARMS = [('none', 0.0), ('intra_mass', 1.0), ('intra_mass', 10.0), ('intra_mass', 100.0)]


def sweep_j_jobs(trials, sizes, arms=J_ARMS, families=GRAPH_FAMILIES):
    """Does the intra_mass setting hold across problem sizes and p/k ratios?
    One-child stays at 100 throughout -- without it low graph_loss does not translate
    into low obs/intv loss."""
    return [{'sweep': 'J', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': 10.0,
             'lambda_one_child': 100.0,
             'info_mode': mode, 'lambda_recon': lam}
            for trial, (n_low, n_high), (gt, d), (mode, lam)
            in itertools.product(range(trials), sizes, families, arms)]


K_ARMS = [(ls, im) for ls in (0.0, 10.0) for im in (0.0, 10.0, 100.0)]


def sweep_k_jobs(trials, sizes, arms=K_ARMS, families=GRAPH_FAMILIES):
    """Is the surjective term still needed once intra_mass is penalising the blob?
    intra_mass attacks the degeneracy surjectivity cannot see, so it may subsume it."""
    return [{'sweep': 'K', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': ls,
             'lambda_one_child': 100.0,
             'info_mode': 'intra_mass' if im > 0 else 'none', 'lambda_recon': im}
            for trial, (n_low, n_high), (gt, d), (ls, im)
            in itertools.product(range(trials), sizes, families, arms)]


L_ARMS = [('old', 0.0), ('old', 100.0),
          ('new', 0.0), ('new', 10.0), ('new', 100.0)]


def sweep_l_jobs(trials, sizes, arms=L_ARMS, families=(('one_child', 1.0),)):
    """Does renormalising graph_loss and one_child_loss make one lambda set transfer
    across problem sizes? lambda_graph=10 / lambda_one_child=100 throughout, which under
    the new normalisation keeps their ratio near 6 at every size."""
    return [{'sweep': 'L', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': 10.0,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0, 'norm': nm,
             'info_mode': 'intra_mass' if im > 0 else 'none', 'lambda_recon': im}
            for trial, (n_low, n_high), (gt, d), (nm, im)
            in itertools.product(range(trials), sizes, families, arms)]


M_ARMS = [('old', 100.0), ('graph_const', 100.0)]


def sweep_m_jobs(trials, sizes, arms=M_ARMS, families=(('one_child', 1.0),), trial_start=0):
    """Isolates the moving-denominator question: graph_const applies the same k-factor
    scale correction as the per-pair normalisation, but through a constant, so there is
    no gradient path through the denominator. one_child is left alone in both arms."""
    return [{'sweep': 'M', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': 10.0,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0, 'norm': nm,
             'info_mode': 'intra_mass', 'lambda_recon': im}
            for trial, (n_low, n_high), (gt, d), (nm, im)
            in itertools.product(range(trial_start, trial_start + trials),
                                 sizes, families, arms)]


N_LAMBDAS = [0.0, 1.0, 3.0, 10.0, 30.0]


def sweep_n_jobs(trials, sizes, lambdas=N_LAMBDAS):
    """Random DAGs, varying k. Forcing one-child on a k=10 random DAG collapses it to ~3.6
    clusters, so k=10 is over-specified and blob-plus-singletons is the honest answer to the
    constraints. Does a k matched to the one-child structure behave sensibly? intra_mass is
    scaled by k here so one lambda is comparable across the k values."""
    return [{'sweep': 'N', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': 0.5,
             'graph_type': 'random', 'surj_mode': 'empty_prob', 'lambda_surj': 10.0,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0, 'norm': 'graph_const',
             'intra_scale': 'times_k',
             'info_mode': 'intra_mass' if lam > 0 else 'none', 'lambda_recon': lam}
            for trial, (n_low, n_high), lam
            in itertools.product(range(trials), sizes, lambdas)]


def sweep_o_jobs(trials, sizes, restarts=10, surjs=(0.0, 10.0),
                 families=(('one_child', 1.0), ('random', 0.5))):
    """Restarts on the SAME problem with different initialisations, crossed with the
    surjective term on/off. The point is to separate init-dependent failure (restarts help)
    from problem-intrinsic failure (they do not). Selection at inference would use
    graph_loss then intra_mass, both computable without ground truth."""
    return [{'sweep': 'O', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': ls,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0, 'norm': 'graph_const',
             'intra_scale': 'times_k', 'info_mode': 'intra_mass', 'lambda_recon': 10.0,
             'restart': r, 'init_seed': 900000 + 1000 * trial + r}
            for trial, (n_low, n_high), (gt, d), ls, r
            in itertools.product(range(trials), sizes, families, surjs, range(restarts))]


def sweep_p_jobs(trials, sizes, restarts=5, families=(('random', 0.5),)):
    """intra_mass vs the in-edge All-In term (recon), both renormalised by k, as the sole
    utility term alongside graph_loss and one_child. Surjectivity off."""
    arms = [('intra_mass', 10.0), ('recon_const', 10.0)]
    return [{'sweep': 'P', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'surj_mode': 'empty_prob', 'lambda_surj': 0.0,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0, 'norm': 'graph_const',
             'intra_scale': 'times_k', 'info_mode': mode, 'lambda_recon': lam,
             'restart': r, 'init_seed': 700000 + 1000 * trial + r}
            for trial, (n_low, n_high), (gt, d), (mode, lam), r
            in itertools.product(range(trials), sizes, families, arms, range(restarts))]


def sweep_q_jobs(trials, sizes, restarts=5,
                 families=(('one_child', 1.0), ('random', 0.5))):
    """Closure as a cheap baseline (no training), against random-init and closure-init
    training. Coherence term is recon, surjectivity off, graph renormalised."""
    base = dict(surj_mode='empty_prob', lambda_surj=0.0, lambda_graph=10.0,
                lambda_one_child=100.0, norm='graph_const', intra_scale='times_k',
                info_mode='recon_const', lambda_recon=10.0)
    arms = [('closure', 0), ('reg_surjective', None), ('closure', None)]
    return [dict(base, sweep='Q', trial=trial, n_low=n_low, n_high=n_high, density=d,
                 graph_type=gt, init=init, epochs=ep, restart=r,
                 init_seed=500000 + 1000 * trial + r)
            for trial, (n_low, n_high), (gt, d), (init, ep), r
            in itertools.product(range(trials), sizes, families, arms, range(restarts))]


R_ARMS = [
    # name, init, epochs, info_mode, lambda_recon, surj_mode, lambda_surj, norm, intra_scale
    ('ORIGINAL',      'reg_surjective', None, 'recon',       10.0, 'hard_count', 1.0,  'old',         'plain'),
    ('REC_recon',     'reg_surjective', None, 'recon_const', 10.0, 'empty_prob', 0.0,  'graph_const', 'plain'),
    ('REC_recon_clo', 'closure',        None, 'recon_const', 10.0, 'empty_prob', 0.0,  'graph_const', 'plain'),
    ('REC_intra_clo', 'closure',        None, 'intra_mass',  10.0, 'empty_prob', 0.0,  'graph_const', 'times_k'),
    ('CLOSURE_ONLY',  'closure',        0,    'none',        0.0,  'empty_prob', 0.0,  'graph_const', 'plain'),
]


def sweep_r_jobs(trials, sizes, restarts=5,
                 families=(('one_child', 1.0), ('random', 0.5))):
    """The confirmatory comparison: every candidate configuration on identical seeds."""
    return [{'sweep': 'R', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'arm_name': name, 'init': init, 'epochs': ep,
             'info_mode': im, 'lambda_recon': lr, 'surj_mode': sm, 'lambda_surj': ls,
             'norm': nm, 'intra_scale': isc,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0,
             'restart': r, 'init_seed': 300000 + 1000 * trial + r}
            for trial, (n_low, n_high), (gt, d), (name, init, ep, im, lr, sm, ls, nm, isc), r
            in itertools.product(range(trials), sizes, families, R_ARMS, range(restarts))]


S_ARMS = [
    # isolate each ingredient of the winning configuration, one change at a time
    ('FULL',          'closure',        'intra_mass', 10.0, 0.0,  'graph_const', 'times_k'),
    ('minus_closure', 'reg_surjective', 'intra_mass', 10.0, 0.0,  'graph_const', 'times_k'),
    ('minus_gconst',  'closure',        'intra_mass', 10.0, 0.0,  'old',         'times_k'),
    ('minus_timesk',  'closure',        'intra_mass', 10.0, 0.0,  'graph_const', 'plain'),
    ('plus_surj',     'closure',        'intra_mass', 10.0, 10.0, 'graph_const', 'times_k'),
    ('swap_recon',    'closure',        'recon_const', 10.0, 0.0, 'graph_const', 'plain'),
]


def sweep_s_jobs(trials, sizes, restarts=5, families=(('one_child', 1.0),)):
    """Ablation of the winning configuration: one ingredient removed at a time."""
    return [{'sweep': 'S', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'arm_name': name, 'init': init, 'epochs': None,
             'info_mode': im, 'lambda_recon': lr, 'surj_mode': 'empty_prob',
             'lambda_surj': ls, 'norm': nm, 'intra_scale': isc,
             'lambda_graph': 10.0, 'lambda_one_child': 100.0,
             'restart': r, 'init_seed': 300000 + 1000 * trial + r}
            for trial, (n_low, n_high), (gt, d), (name, init, im, lr, ls, nm, isc), r
            in itertools.product(range(trials), sizes, families, S_ARMS, range(restarts))]


T_ARMS = [
    ('with_gconst', 'graph_const'),
    ('no_gconst',   'old'),
]


def sweep_t_jobs(trials, sizes, restarts=3, families=(('one_child', 1.0),)):
    """Does graph_const help or hurt once intra_mass x k is in place? The ablation said it
    hurts at p=100; check whether that holds at p=50 and p=200 before dropping it."""
    return [{'sweep': 'T', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'arm_name': name, 'init': 'closure', 'epochs': None,
             'info_mode': 'intra_mass', 'lambda_recon': 10.0, 'surj_mode': 'empty_prob',
             'lambda_surj': 0.0, 'norm': nm, 'intra_scale': 'times_k',
             'lambda_graph': 10.0, 'lambda_one_child': 100.0,
             'restart': r, 'init_seed': 300000 + 1000 * trial + r}
            for trial, (n_low, n_high), (gt, d), (name, nm), r
            in itertools.product(range(trials), sizes, families, T_ARMS, range(restarts))]


def sweep_u_jobs(trials, sizes, restarts=12,
                 families=(('one_child', 1.0), ('random', 0.5))):
    """Many restarts of the shipped configuration, so the success-vs-restarts curve can be
    read off directly rather than extrapolated."""
    return [{'sweep': 'U', 'trial': trial, 'n_low': n_low, 'n_high': n_high, 'density': d,
             'graph_type': gt, 'arm_name': 'SHIPPED', 'init': 'closure', 'epochs': None,
             'info_mode': 'intra_mass', 'lambda_recon': 10.0, 'intra_scale': 'times_k',
             'surj_mode': 'empty_prob', 'lambda_surj': 0.0, 'norm': 'old',
             'lambda_graph': 10.0, 'lambda_one_child': 100.0,
             'restart': r, 'init_seed': 610000 + 1000 * trial + r}
            for trial, (n_low, n_high), (gt, d), r
            in itertools.product(range(trials), sizes, families, range(restarts))]


def run_jobs(jobs, out_csv, n_workers=6):
    import multiprocessing as mp
    t0 = time.time()
    done = 0
    header_written = out_csv.exists()
    with mp.Pool(n_workers) as pool:
        for row in pool.imap_unordered(run_one, jobs, chunksize=1):
            pd.DataFrame([row]).to_csv(out_csv, mode='a', index=False, header=not header_written)
            header_written = True
            done += 1
            if done % 10 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(jobs) - done)
                print(f"  {done}/{len(jobs)} done, {elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min left",
                      flush=True)
    return pd.read_csv(out_csv)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', default='AB', help="which sweeps to run: any of A, B, C")
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--restarts', type=int, default=5)
    parser.add_argument('--trial-start', type=int, default=0,
                        help="first seed index, so more seeds can be added to a finished sweep")
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--sizes', default='50x5,100x10', help="comma-separated n_low x n_high")
    parser.add_argument('--densities', default='0.4,1.0')
    # Each arm is "mode:lambda_surj:lambda_recon".
    parser.add_argument('--c-arms',
                        default='hard_count:1:10,empty_prob:10:0,topk_barrier:10:0')
    parser.add_argument('--c-inits',
                        default='reg_surjective,random_normal,uniform,biased_0.8')
    args = parser.parse_args()

    sizes = [tuple(int(v) for v in s.split('x')) for s in args.sizes.split(',')]
    densities = [float(v) for v in args.densities.split(',')]
    lambdas = [0.1, 1.0, 10.0, 100.0]

    outdir = Path(args.outdir) if args.outdir else Path(
        f"results/smoke_tests/{time.strftime('%Y%m%d-%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {outdir}", flush=True)

    if 'A' in args.sweep:
        jobs = sweep_a_jobs(args.trials, sizes, densities, lambdas)
        print(f"Sweep A: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_a.csv', args.workers)
    if 'B' in args.sweep:
        jobs = sweep_b_jobs(args.trials, sizes, densities)
        print(f"Sweep B: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_b.csv', args.workers)
    if 'U' in args.sweep:
        jobs = sweep_u_jobs(args.trials, sizes, restarts=args.restarts)
        print(f"Sweep U: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_u.csv', args.workers)
    if 'T' in args.sweep:
        jobs = sweep_t_jobs(args.trials, sizes, restarts=args.restarts)
        print(f"Sweep T: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_t.csv', args.workers)
    if 'S' in args.sweep:
        jobs = sweep_s_jobs(args.trials, sizes, restarts=args.restarts)
        print(f"Sweep S: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_s.csv', args.workers)
    if 'R' in args.sweep:
        jobs = sweep_r_jobs(args.trials, sizes, restarts=args.restarts)
        print(f"Sweep R: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_r.csv', args.workers)
    if 'Q' in args.sweep:
        jobs = sweep_q_jobs(args.trials, sizes)
        print(f"Sweep Q: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_q.csv', args.workers)
    if 'P' in args.sweep:
        jobs = sweep_p_jobs(args.trials, sizes)
        print(f"Sweep P: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_p.csv', args.workers)
    if 'O' in args.sweep:
        jobs = sweep_o_jobs(args.trials, sizes)
        print(f"Sweep O: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_o.csv', args.workers)
    if 'N' in args.sweep:
        jobs = sweep_n_jobs(args.trials, sizes)
        print(f"Sweep N: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_n.csv', args.workers)
    if 'M' in args.sweep:
        jobs = sweep_m_jobs(args.trials, sizes, trial_start=args.trial_start)
        print(f"Sweep M: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_m.csv', args.workers)
    if 'L' in args.sweep:
        jobs = sweep_l_jobs(args.trials, sizes)
        print(f"Sweep L: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_l.csv', args.workers)
    if 'K' in args.sweep:
        jobs = sweep_k_jobs(args.trials, sizes)
        print(f"Sweep K: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_k.csv', args.workers)
    if 'J' in args.sweep:
        jobs = sweep_j_jobs(args.trials, sizes)
        print(f"Sweep J: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_j.csv', args.workers)
    if 'I' in args.sweep:
        jobs = sweep_i_jobs(args.trials, sizes)
        print(f"Sweep I: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_i.csv', args.workers)
    if 'H' in args.sweep:
        jobs = sweep_h_jobs(args.trials, sizes)
        print(f"Sweep H: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_h.csv', args.workers)
    if 'G' in args.sweep:
        jobs = sweep_g_jobs(args.trials, sizes)
        print(f"Sweep G: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_g.csv', args.workers)
    if 'F' in args.sweep:
        jobs = sweep_f_jobs(args.trials, sizes)
        print(f"Sweep F: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_f.csv', args.workers)
    if 'E' in args.sweep:
        jobs = sweep_e_jobs(args.trials, sizes, E_ARMS)
        print(f"Sweep E: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_e.csv', args.workers)
    if 'D' in args.sweep:
        jobs = sweep_d_jobs(args.trials, sizes, D_ARMS)
        print(f"Sweep D: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_d.csv', args.workers)
    if 'C' in args.sweep:
        arms = [(p[0], float(p[1]), float(p[2]))
                for p in (spec.split(':') for spec in args.c_arms.split(','))]
        jobs = sweep_c_jobs(args.trials, sizes, densities, arms, args.c_inits.split(','))
        print(f"Sweep C: {len(jobs)} runs", flush=True)
        run_jobs(jobs, outdir / 'sweep_c.csv', args.workers)
    print("Done.", flush=True)
