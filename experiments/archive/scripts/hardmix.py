"""SMOKE TEST: does adding a HARD (T_hard) copy of graph/recon loss to the soft one help?

Hypothesis: the soft losses can settle in fuzzy partitions that score well under T_soft but
decode badly under argmax. Adding the same penalty evaluated at the straight-through T_hard
gives a "shock" that is only zero when the DISCRETE partition is good.

No source change: subclass HailMary and override forward(). T_hard is already the
straight-through estimator (T_soft + (onehot - T_soft).detach()), so the hard term is
differentiable and its gradient flows through T_soft. The hard term is ADDED to the soft
one inside the returned graph_loss / recon_loss, so it inherits the same lambda.

Arms: none (baseline) / graph / recon / both, at w=1.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from occ import occ
from frontier_fig import score, P

EPOCHS = 2000
LAMS = [0., 1000.]
W = float(os.environ.get('HARDMIX_W', '1.0'))   # weight on the hard copy
TAG = os.environ.get('HARDMIX_TAG', '')         # output suffix, e.g. '_w0.1'
ARMS = [('none', 0., 0.), ('graph', W, 0.), ('recon', 0., W), ('both', W, W)]


def _pen(H_soft, co_assign, diag_mask, n_high):
    """The shared 'pairwise CE between rows, weighted by co-assignment' penalty."""
    H = torch.clamp(H_soft, min=1e-6, max=1. - 1e-6)
    tgt = H.detach()[:, :, None, :]
    lg = torch.logit(H)[:, None, :, :]
    ce = tgt * F.softplus(-lg) + (1.0 - tgt) * F.softplus(lg)
    return (co_assign * diag_mask * ce.sum(-1)).mean() * n_high


class HardMix(HailMary):
    w_graph_hard = 0.
    w_recon_hard = 0.

    def forward(self, temp=1., compute_recon=True, compute_graph=True, compute_one_child=True):
        T_hard, T_soft = self.get_partition(temp)
        Ab = self.A_low_binary
        diag_mask = 1 - torch.eye(self.n_low, device=self.device)
        zero = torch.zeros((), device=self.device)

        def recon_term(T):
            H = torch.max(Ab.transpose(-1, -2)[:, :, None, :] * T.T[None, None, :, :], dim=-1)[0]
            return _pen(H, T @ T.T, diag_mask, self.n_high)

        def graph_H(T):
            return torch.max(Ab[:, :, None, :] * T.T[None, None, :, :], dim=-1)[0] * (1 - T)

        recon_loss = zero
        if compute_recon:
            recon_loss = recon_term(T_soft)
            if self.w_recon_hard:
                recon_loss = recon_loss + self.w_recon_hard * recon_term(T_hard)

        H_soft_graph = graph_H(T_soft) if (compute_graph or compute_one_child) else None
        graph_loss = zero
        if compute_graph:
            graph_loss = _pen(H_soft_graph, T_soft @ T_soft.T, diag_mask, self.n_high)
            if self.w_graph_hard:
                graph_loss = graph_loss + self.w_graph_hard * _pen(
                    graph_H(T_hard), T_hard @ T_hard.T, diag_mask, self.n_high)

        one_child_loss = zero
        if compute_one_child:
            one_child_loss = torch.mean(H_soft_graph.sum(-1) ** 2 - (H_soft_graph ** 2).sum(-1))

        entropy_loss = -torch.sum(T_soft * torch.log(T_soft + 1e-8), dim=-1).mean()
        co = T_soft @ T_soft.T
        info_loss = ((self.A_low.abs() * (co * diag_mask)[None]).sum()
                     / self.A_low.abs().sum().clamp(min=1e-12)) * self.n_high
        surjective_loss = (self.n_high - torch.sum(T_hard.sum(dim=0) > 0)) / self.n_high
        return recon_loss, graph_loss, one_child_loss, surjective_loss, entropy_loss, info_loss


def job(a):
    torch.set_num_threads(1)
    dens, inst, lam, arm, wg, wr = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=inst, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    g0, T0, ci = occ(A, seed=inst); K = max(ci['k'], 2)
    out = []
    if lam == LAMS[0] and arm == 'none':
        out.append(score(A, T0, E_low, su, absA, tot, method='OCC', lam_oc=np.nan,
                         k=K, seed=inst, density=dens, arm='OCC', secs=0.))
    t0 = time.time()
    torch.manual_seed(10_000 * inst + int(lam))     # SAME init across arms -> paired
    m = HardMix(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
    m.w_graph_hard, m.w_recon_hard = wg, wr
    m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
            lambda_info=0., lambda_surj=0, verbose=False)
    out.append(score(A, m.get_partition()[0].detach().numpy(), E_low, su, absA, tot,
                     method='HailMary', lam_oc=lam, k=K, seed=inst, density=dens,
                     arm=arm, secs=time.time() - t0))
    return out


if __name__ == '__main__':
    jobs = [(d, i, l, a, wg, wr) for d in (0.1, 0.5, 0.9) for i in range(3)
            for l in LAMS for a, wg, wr in ARMS]
    print('%d runs (%d epochs, w=%g): 3 densities x 3 inst x %d lam x %d arms'
          % (len(jobs), EPOCHS, W, len(LAMS), len(ARMS)), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/hardmix%s.csv' % TAG
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 12 == 0:
                pd.DataFrame(rows).to_csv(path, index=False)
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(path, index=False)
    print('\ndone %.1f min -> %s' % ((time.time() - t0) / 60, path))
