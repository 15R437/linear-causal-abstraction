"""Local causal abstraction of a ReLU MLP.

Within the activation region containing x0 a ReLU network is EXACTLY a linear SCM over its units:
inputs are root variables (X_i = U_i, varying around x0), and each hidden/output unit i in layer l has
parents in layer l-1 with weights D_l[i] W_l[i, :] and a constant exogenous term D_l[i] b_l[i], where
D_l = diag(1[z_l(x0) > 0]) is the activation pattern. The region is certified by the radius
r(x0) = min_i |z_i(x0)| / ||dz_i/dx||, inside which no unit changes state.

We build that SCM, check it reproduces the network, then abstract it: partition -> fit_abstraction_eigen
-> RAE and reconstruction error. For the layerwise partition the fitter's collinearity condition
t_l ~ (D_{l+1} W_{l+1})^T t_{l+1} is backpropagation, so each column of T should be the gradient of the
output with respect to that layer's activations; we check that too.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.algos.T_fitter import fit_abstraction_eigen
from src.algos.greedy import aoc_closure, greedy_allin, random_partition, partition_to_matrix, all_in_violation, cluster_edge_counts
from src.utils.helper import get_S
from src.utils.losses import RAE, reconstruction_loss


def target(x):
    """Compositional task: z1 = sin(x1) + x2, z2 = cos(x3) * x4, y = z1 * z2."""
    z1 = torch.sin(x[:, 0]) + x[:, 1]
    z2 = torch.cos(x[:, 2]) * x[:, 3]
    return (z1 * z2)[:, None]


def train_mlp(widths, n_steps=4000, seed=0):
    torch.manual_seed(seed)
    layers = []
    for d_in, d_out in zip(widths[:-1], widths[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    mlp = nn.Sequential(*layers[:-1])  # linear output layer
    optimiser = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for step in range(n_steps):
        x = torch.rand(256, widths[0]) * 4 - 2
        loss = ((mlp(x) - target(x)) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    x_test = torch.rand(5000, widths[0]) * 4 - 2
    with torch.no_grad():
        test_mse = ((mlp(x_test) - target(x_test)) ** 2).mean().item()
        baseline = target(x_test).var().item()
    return mlp, test_mse, baseline


def linear_layers(mlp):
    return [m for m in mlp if isinstance(m, nn.Linear)]


def forward_with_activations(mlp, x):
    """Returns the post-activations of every layer (input first, output last) and every pre-activation."""
    post, pre = [x], []
    h = x
    for m in mlp:
        h = m(h)
        if isinstance(m, nn.Linear):
            pre.append(h)
        if isinstance(m, nn.ReLU) or m is linear_layers(mlp)[-1]:
            post.append(h)
    return post, pre


def local_scm(mlp, x0):
    """Exact linear SCM of the MLP in the activation region of x0.

    Returns A (A[i, j] is the edge i -> j), the exogenous vector u (x0 for inputs, gated biases otherwise),
    the layer of each node, and the certified radius of the region.
    """
    Ws = [m.weight.detach().numpy().astype(float) for m in linear_layers(mlp)]
    bs = [m.bias.detach().numpy().astype(float) for m in linear_layers(mlp)]
    widths = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]
    offsets = np.cumsum([0] + widths)
    p = offsets[-1]

    x0_t = torch.tensor(x0, dtype=torch.float32)[None]
    _, pre = forward_with_activations(mlp, x0_t)
    gates = [(z[0].detach().numpy() > 0).astype(float) for z in pre[:-1]] + [np.ones(widths[-1])]  # output layer is linear

    A = np.zeros((p, p))
    u = np.zeros(p)
    u[:widths[0]] = x0
    layer_of = np.repeat(np.arange(len(widths)), widths)
    for l, (W, b, D) in enumerate(zip(Ws, bs, gates)):
        parents = slice(offsets[l], offsets[l + 1])
        children = slice(offsets[l + 1], offsets[l + 2])
        A[parents, children] = (D[:, None] * W).T
        u[children] = D * b

    # certified radius: distance from x0 to the nearest hyperplane where a hidden unit switches
    jac = torch.autograd.functional.jacobian(lambda x: torch.cat([z[0] for z in forward_with_activations(mlp, x[None])[1][:-1]]), x0_t[0])
    z0 = torch.cat([z[0] for z in pre[:-1]]).detach().numpy()
    radius = float(np.min(np.abs(z0) / np.linalg.norm(jac.numpy(), axis=1).clip(min=1e-12)))
    return A, u, layer_of, radius, widths


def scm_values(A, u):
    """Solves x = A^T x + u."""
    return np.linalg.solve(np.eye(len(u)) - A.T, u)


def network_values(mlp, x):
    post, _ = forward_with_activations(mlp, torch.tensor(x, dtype=torch.float32)[None])
    return np.concatenate([h[0].detach().numpy() for h in post])


def layer_gradients(mlp, x0):
    """Gradient of the (scalar) output with respect to each layer's post-activations, input first."""
    post, _ = forward_with_activations(mlp, torch.tensor(x0, dtype=torch.float32, requires_grad=True)[None])
    for h in post[:-1]:
        h.retain_grad()
    post[-1].sum().backward()
    return [h.grad[0].numpy() for h in post[:-1]] + [np.ones(post[-1].shape[1])]


def abstract(A, T_hard, sigma_u, seed=0):
    E_low = np.linalg.inv(np.eye(len(A)) - A)
    t0 = time.time()
    A_high, T = fit_abstraction_eigen(A, T_hard, seed=seed)
    fit_time = time.time() - t0
    S = get_S(T=T, E_low=E_low, T_hard=T_hard, A_low=A)
    rae = {name: RAE(A_low=A, T=T, A_high=A_high, S=S, E_low=E_low, T_hard=T_hard, sigma_u=s) for name, s in sigma_u.items()}
    return T, A_high, rae, reconstruction_loss(A_low=A, T=T, A_high=A_high), fit_time


if __name__ == "__main__":
    widths = [4, 32, 32, 1]
    mlp, test_mse, baseline = train_mlp(widths)
    print(f"MLP {widths}: test MSE {test_mse:.4f} (target variance {baseline:.3f})")

    rng = np.random.default_rng(0)
    x0 = rng.uniform(-2, 2, widths[0])
    A, u, layer_of, radius, widths = local_scm(mlp, x0)
    p = len(A)
    n_active = [int(np.any(A[:, layer_of == l] != 0, axis=0).sum()) for l in range(1, len(widths))]
    print(f"\nx0 = {np.round(x0, 3)}: p = {p} nodes, active units per layer {n_active}, "
          f"{int((A != 0).sum())} edges, certified radius r = {radius:.4f}")

    # the SCM reproduces the network exactly inside the region and breaks outside it
    for scale in [0, 0.99, 3, 10]:
        errs = []
        for _ in range(20):
            d = rng.standard_normal(widths[0])
            x = x0 + scale * radius * d / np.linalg.norm(d)
            A_x, u_x = A.copy(), u.copy()
            u_x[:widths[0]] = x
            errs.append(np.abs(scm_values(A_x, u_x) - network_values(mlp, x)).max())
        print(f"  |SCM - network| at distance {scale:>4} r: max {max(errs):.2e}")

    # exogenous variances: inputs vary around x0, biases are constant
    sigma_u = {'ones': np.ones(p), 'inputs': np.where(layer_of == 0, 1.0, 1e-8)}

    layer_partition = [list(np.where(layer_of == l)[0]) for l in range(len(widths))]
    partitions = {
        'layerwise (k=4)': layer_partition,
        'aoc_closure (k=4)': aoc_closure(A, 4, seed=0),
        'aoc_closure (k=8)': aoc_closure(A, 8, seed=0),
        'greedy_allin (k=4)': greedy_allin(A, 4),
        'greedy_allin (k=8)': greedy_allin(A, 8),
        'random (k=8)': random_partition(p, 8, seed=0),
    }

    print(f"\n{'partition':20s} {'K':>3s} {'sizes':28s} {'All-In':>6s} {'1-child':>7s} {'RAE(ones)':>10s} {'RAE(inputs)':>11s} {'recon':>7s} {'fit':>6s}")
    results = {}
    for name, partition in partitions.items():
        partition = [list(c) for c in partition if len(c) > 0]
        T_hard = partition_to_matrix(partition)
        T, A_high, rae, recon, fit_time = abstract(A, T_hard, sigma_u)
        results[name] = (partition, T, A_high)
        A_partition = cluster_edge_counts(A, partition) > 0
        sizes = str(sorted((len(c) for c in partition), reverse=True))
        print(f"{name:20s} {len(partition):>3d} {sizes:28s} {str(all_in_violation(A, partition) == 0):>6s} "
              f"{str(A_partition.sum(axis=1).max() <= 1):>7s} {rae['ones']:>10.2e} {rae['inputs']:>11.2e} {recon:>7.3f} {fit_time:>5.2f}s")

    # semantics of T for the layerwise abstraction: each column should be d(output)/d(layer activations)
    partition, T, A_high = results['layerwise (k=4)']
    grads = layer_gradients(mlp, x0)
    print("\nlayerwise T vs backpropagated gradient d y / d h_l (|cosine|):")
    for l, (clst, g) in enumerate(zip(partition, grads)):
        t = T[clst, l]
        cos = abs(t @ g) / (np.linalg.norm(t) * np.linalg.norm(g) + 1e-12)
        print(f"  layer {l} ({len(clst):2d} units): {cos:.6f}")
    print("high-level weights along the chain:", np.round([A_high[l, l + 1] for l in range(len(widths) - 1)], 4))
