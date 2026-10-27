import numpy as np
import torch

from .clustering import hierarchical_clustering, heuristic_loss
from .gradcap import GradCAP
from .greedy import aoc_closure
from .uca import uca_model, uca_train
from src.utils.helper import regularised_surjective_init, get_device


"""Thin wrappers exposing each algorithm through a common
`fn(A_low, k, **kwds) -> T_hard` interface."""


def random(A_low, k, **kwds):
    n_low = A_low.shape[0]
    logits = regularised_surjective_init(n_low=n_low, n_high=k)
    indices = np.argmax(logits, axis=1)
    T_hard = np.zeros((n_low, k))
    T_hard[np.arange(n_low), indices] = 1.
    return T_hard


def uca(A_low, k, num_epochs=6000, **kwds):
    # num_epochs is a fixed budget of GRADIENT STEPS (see uca_train). The previous
    # size-dependent heuristic, 1000 * int(log((p-k)*k + 1) + 1), gave 5000-8000 and was
    # moot while uca_train applied no gradient at all.
    model = uca_model(A_low=A_low, k=k)
    ca = uca_train(model, num_epochs)
    T = ca.tau_map_weights.weight.cpu().detach().numpy().T

    # Column-normalise
    T = T / (np.linalg.norm(T, axis=0, keepdims=True) + 1e-8)
    return T


def hail_mary_graph(A_low, k, n_restarts=1, seed=0, **kwds):
    """Best-of-n_restarts GradCAP.

    Each restart starts from a different randomised one-child closure and is scored on the
    two sufficient conditions (All-In via graph_loss, plus one child per cluster); among
    those that satisfy both, the one destroying the least edge mass is returned. Both the
    filter and the tie-break are functions of A_low and T_hard only, so this is an
    inference-time procedure and not oracle selection.

    Falls back to the best-by-graph_loss partition if no restart satisfies both conditions.

    n_restarts defaults to 1: with the closure initialisation the per-run success rate on
    one_child graphs is 1.000, so restarts buy nothing there. On random graphs the curve
    saturates around 6 (0.264 -> 0.326) and 67% of problems fail on every restart, so extra
    attempts are mostly wasted there too.
    """
    from src.utils.losses import graph_loss as _graph_loss

    device = get_device()
    n_low = A_low.shape[0]
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32).to(device)
    num_epochs = 1000 * int(np.log((n_low - k) * k + 1) + 1)

    best, best_key = None, None
    for r in range(n_restarts):
        partition_init = aoc_closure(A_low, k, seed=seed * 1000 + r)
        model = GradCAP(A_low_tensor, k, partition_init, seed=seed * 1000 + r).to(device)
        model.train(num_epochs=num_epochs, verbose=False)
        T_hard = model.get_partition()[0].cpu().detach().numpy()

        gl = _graph_loss(A_low=A_low, T_hard=T_hard)
        mask = (T_hard.T @ (A_low != 0).astype(float) @ T_hard) > 0
        np.fill_diagonal(mask, False)
        one_child = mask.sum(axis=1).max() <= 1 if k else True
        co = T_hard @ T_hard.T
        np.fill_diagonal(co, 0)
        info = (np.abs(A_low) * co).sum() / max(np.abs(A_low).sum(), 1e-12)

        # rank: satisfying both conditions first, then least information destroyed
        key = (0 if (gl < 1e-8 and one_child) else 1, info if (gl < 1e-8 and one_child) else gl)
        if best_key is None or key < best_key:
            best, best_key = T_hard, key
    return best


def hc_fast(A_low, k, init_assignment=None, **kwds):
    _, T_dict, _ = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[heuristic_loss])
    return T_dict[k]
