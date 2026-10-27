"""
=== Theory ===
We abstract a hidden layer of a trained MLP into k mutually orthogonal subspaces, each corresponding to a high-level
variable. Subspace i is spanned by the orthonormal columns of T_i in R^{p x d_i}, and an activation h is projected onto
the subspace by z = T_i^T h in R^{d_i}.

The high-level variable i takes one of m_i values, determined by m_i affine forms which carve the subspace into convex regions.

    s_c(z) = A_c . z + a_c      (A_c in R^{d_i}, a_c in R)      for c in {1, ..., m_i},

    tau_i(h) = argmax_c s_c(T_i^T h) in {1, ..., m_i}.

The high-level model is the MLP itself up to the chosen layer, where the high-level variables are given by tau_i(h)
and then these have directed arrows into the output variables only. Essentially, every subsequent layer of the MLP
is marginalised. The high-level output is computed from tau(h) by a majority vote over the intervened activations
used for fitting, taking the network's predicted class (not the true label) as the target.

(Low-Level): Input --> [MLP up to chosen layer] --> h in R^p --> [Rest of MLP] --> Output
(High-Level): Input --> [MLP up to chosen layer] --> tau(h) in {1, ..., m_1} x ... x {1, ..., m_k} --> Output

=== Soft Version ===
We replace the argmax with the differentiable softmax and an inverse temperature kappa so that a soft version of the
high-level model computes the probability distribution over the m_i values for each high-level variable i 
(rather than just picking the maximum). In this soft version, we have

    q_i(h) = softmax(kappa * (s_1(T_i^T h), ..., s_{m_i}(T_i^T h)))

Let (c_1, c_2, ..., c_k) in {1, ..., m_1} x ... x {1, ..., m_k} be a joint (hard) assignment over the k high-level variables.
Each assignment defines a probability distribution (note that conditional on h these are independent, since each
variable reads its own disjoint subspace) and a learnable logit vector:

    p(c_1,...,c_k) = q_1(h)_{c_1} x q_2(h)_{c_2} x ... x q_k(h)_{c_k}

    M_{(c_1,...,c_k)} = (logit_1, ..., logit_C)

where C is the number of classes for the task. The high-level output is then computed as

    Output = sum_{(c_1,...,c_k)} [ p(c_1,...,c_k) . M_{(c_1,...,c_k)} ]

Learnable Parameters:
    - T_i in R^{p x d_i} for i in {1,...,k}

    - A^i_c in R^{d_i} and a^i_c in R, for i in {1,...,k} and c in {1,...,m_i}   (sum_i m_i forms, one set per variable)

    - M_{(c_1,...,c_k)} in R^{C} for (c_1,...,c_k) in {1, ..., m_1} x ... x {1, ..., m_k}
      (learned during training, then replaced by the majority vote above)

=== Interchange Intervention Training ===
Fix the following hyper parameters given a layer of the MLP with dimension dim(h):
- k (the number of high-level variables) < dim(h);
- m_1,...,m_k the number of different values each variable takes;
- d_1,...,d_k (these can in theory be learned using Boundless DAS) and 
- d_max = sum_i d_i (the total dimension budget)

Given a dataset D = {x_1, ..., x_N} we sample a base input b, a source input s, and a subset Theta of the
high-level variables. Writing h_b and h_s for their activations at the chosen layer, the interchange
intervention patches the source's components of the targeted subspaces into the base and leaves the rest
alone. With P_i = T_i T_i^T the orthogonal projector onto subspace i, and P_Theta = sum_{i in Theta} P_i,

    h_patched = (I - P_Theta) h_b + P_Theta h_s
              = h_b + sum_{i in Theta} T_i T_i^T (h_s - h_b).

Because the subspaces are mutually orthogonal, P_i P_j = 0 for i != j, so the patch is exactly a
substitution of values at the high level:

    tau_i(h_patched) = tau_i(h_s)  for i in Theta,     tau_i(h_patched) = tau_i(h_b)  otherwise.

Orthogonality of the subspaces allows us to intervene on each high-level variable independently since
intervening on one variable leaves every other variable's value untouched. The loss is the cross-entropy
between the high-level prediction (sum_c p(c) . M_c) and the network's predicted class at h_patched (treated as a
constant label).
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

INTERCHANGE_FAMILIES = ("obs", "single", "all")
FAMILIES = INTERCHANGE_FAMILIES + ("dense",)


def split_sequential(model, cut):
    """Split an nn.Sequential at module index `cut` into (to_layer, from_layer), so h = to_layer(x) is the input of
    model[cut] and from_layer(h) finishes the forward pass."""
    head, tail = model[:cut], model[cut:]
    return (lambda x: head(x)), (lambda h: tail(h))


def parse_specs(specs):
    """(m_i, d_i) per high-level variable. An int k is shorthand for k binary variables on one dimension each, which
    is the Boolean case m_i = 2, d_i = 1."""
    if isinstance(specs, int):
        return [(2, 1)] * specs
    out = [(int(m), int(d)) for m, d in specs]
    for m, d in out:
        if m < 2 or d < 1:
            raise ValueError(f"need m >= 2 and d >= 1, got {(m, d)}")
    return out


class MLPAbstraction(nn.Module):
    """k high-level variables on mutually orthogonal blocks of one hidden layer.

    Variable i owns d_i of the D = sum_i d_i columns of T and takes one of m_i values, read as the argmax of m_i
    affine forms on its block. The mechanism M is a table over the prod_i m_i joint assignments.

    The readout is gauge-fixed. The argmax is invariant to A_c -> A_c + v for a common v, and the softmax sharpness
    is set by the scale of the pairwise differences A_c - A_c', so kappa would otherwise be redundant with ||A|| and
    the model would learn its own temperature. We centre over values and scale so the mean squared pairwise distance
    is 1, i.e. ||A||_F = sqrt((m_i - 1) / 2); at m_i = 2 this reduces to ||t|| = 1.
    """

    def __init__(self, p, specs, n_classes, kappa=8.0, scale=1.0, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.specs = parse_specs(specs)
        self.sizes = [m for m, _ in self.specs]
        self.dims = [d for _, d in self.specs]
        self.k, self.p, self.n_classes = len(self.specs), p, n_classes
        self.D = sum(self.dims)
        self.n_cells = int(np.prod(self.sizes))
        if self.D > p:
            raise ValueError(f"dimension budget {self.D} exceeds the layer width {p}")
        # weight is (D, p); its rows are an orthonormal basis, so T = weight.T has orthonormal columns
        self.proj = nn.utils.parametrizations.orthogonal(nn.Linear(p, self.D, bias=False))
        self.A_raw = nn.ParameterList([nn.Parameter(torch.randn(m, d) * 0.1) for m, d in self.specs])
        self.a_raw = nn.ParameterList([nn.Parameter(torch.zeros(m)) for m, _ in self.specs])
        self.M = nn.Parameter(torch.zeros(self.n_cells, n_classes))
        self.register_buffer("kappa", torch.tensor(float(kappa)))
        self.register_buffer("scale", torch.tensor(float(scale)))
        # every cell as its tuple of per-variable values, and the mixed-radix weights that index it
        self.register_buffer("cells", torch.tensor(list(itertools.product(*[range(m) for m in self.sizes]))))
        self.register_buffer("radix", torch.tensor([int(np.prod(self.sizes[i + 1:])) for i in range(self.k)]))

    @property
    def T(self):
        """(p, D) with orthonormal columns; columns are grouped into the blocks T_1 ... T_k in order."""
        return self.proj.weight.T

    def blocks(self, h):
        """z_i = T_i^T h for each variable, as a list of (B, d_i)."""
        return torch.split(self.proj(h), self.dims, dim=-1)

    def set_kappa(self, kappa):
        self.kappa.fill_(float(kappa))

    def set_scale(self, scale):
        """Typical spread of a projection, so that kappa is dimensionless and one value transfers across layers."""
        self.scale.fill_(float(scale))

    def readout(self, i):
        """Gauge-fixed (A, a) for variable i: centred over values, scaled to unit mean squared pairwise distance."""
        A = self.A_raw[i] - self.A_raw[i].mean(dim=0, keepdim=True)
        A = A / A.norm().clamp_min(1e-8) * np.sqrt((self.sizes[i] - 1) / 2)
        return A, self.a_raw[i] - self.a_raw[i].mean()

    def scores(self, h):
        """The m_i affine scores of each variable, as a list of (B, m_i), already scaled by kappa."""
        out = []
        for i, z in enumerate(self.blocks(h)):
            A, a = self.readout(i)
            out.append(self.sharpness * (z @ A.T + a))
        return out

    def values(self, h):
        """tau(h), the hard joint assignment, as (B, k)."""
        return torch.stack([s.argmax(-1) for s in self.scores(h)], dim=-1)

    @property
    def sharpness(self):
        """kappa made dimensionless by the spread of the projections."""
        return self.kappa / self.scale

    def margin(self, h):
        """Gap between the best and second-best score per variable, in raw projection units (the scores carry a
        factor of kappa/scale, which is divided out here). The discrete reading is faithful where kappa times the
        margin, in spread units, is >> 1."""
        gaps = []
        for s in self.scores(h):
            top2 = s.topk(2, dim=-1).values
            gaps.append((top2[:, 0] - top2[:, 1]) / self.sharpness)
        return torch.stack(gaps, dim=-1)

    def cell_index(self, h):
        """Index in [0, prod_i m_i) of the joint assignment at h."""
        return (self.values(h) * self.radix).sum(-1)

    def log_cell_weights(self, h):
        """log p(c_1, ..., c_k) for every cell, (B, n_cells). Built from log_softmax so a saturated variable gives a
        finite log-probability, and summed rather than multiplied for stability."""
        logq = [F.log_softmax(s, dim=-1) for s in self.scores(h)]
        return sum(logq[i][:, self.cells[:, i]] for i in range(self.k))

    def forward(self, h):
        """The soft high-level output, sum_c p(c) M_c. Differentiable in T, the readouts and M."""
        return self.log_cell_weights(h).exp() @ self.M

    def hard_forward(self, h):
        """The high-level output under the hard argmax: a lookup of the realised cell."""
        return self.M[self.cell_index(h)]


def block_mask(model, var_mask):
    """(B, k) choice of which variables to intervene on -> (B, D) coordinate weights. A variable's whole block moves
    together, which is what makes it one variable rather than d_i of them."""
    return torch.cat([var_mask[:, i:i + 1].expand(-1, d) for i, d in enumerate(model.dims)], dim=-1)


def interchange(model, h_base, values, var_mask):
    """h_patched = (I - P_Theta) h_base + P_Theta h_source, in coordinates. `values` holds the target coordinates of
    the whole subspace, (B, D); pass h_source @ T for an interchange intervention."""
    T = model.T
    return h_base + ((values - h_base @ T) * block_mask(model, var_mask)) @ T.T


def sample_var_mask(family, n, k, generator=None):
    """(n, k) 0/1 mask of which high-level variables an intervention targets."""
    if family == "obs":
        return torch.zeros(n, k)
    if family == "single":
        j = torch.randint(k, (n,), generator=generator)
        return F.one_hot(j, k).to(torch.get_default_dtype())
    if family in ("all", "dense"):
        return (torch.rand(n, k, generator=generator) < 0.5).to(torch.get_default_dtype())
    raise ValueError(f"unknown family {family!r}")


def intervention_set(model, H, family, n, generator, dense_scale=2.0):
    """The triples (base, source, Theta), drawn once. Only the coordinate VALUES move with T, so a loss built from a
    fixed triple set is a deterministic function of the parameters -- resampling every step instead leaves a
    gradient-noise floor that no number of steps removes.

    Base and source rows are drawn with replacement from the activations of real inputs, so n only buys distinct
    triples up to what the input set supports.
    """
    base = H[torch.randint(H.shape[0], (n,), generator=generator)]
    mask = sample_var_mask(family, n, model.k, generator)
    if family == "dense":
        # values from nowhere in the model: a probe, not an interchange intervention
        return base, None, dense_scale * torch.randn(n, model.D, generator=generator), mask
    return base, H[torch.randint(H.shape[0], (n,), generator=generator)], None, mask


def apply_interventions(model, base, source, values, var_mask):
    """Rebuild the intervened activations for a fixed triple set. Built under no_grad: the intervention is data
    generation, and letting the gradient move the sampled POINT competes with the gradient that aligns the
    subspaces."""
    with torch.no_grad():
        v = values if source is None else source @ model.T
        return interchange(model, base, v, var_mask)


def infer_decision(from_layer, H):
    """(decision, n_classes) for a network tail: argmax over logits, or a threshold at 0.5 for a single output."""
    with torch.no_grad():
        out = from_layer(H[:1])
    if out.ndim == 1 or out.shape[-1] == 1:
        return (lambda y: (y.reshape(-1) > 0.5).long()), 2
    return (lambda y: y.argmax(-1)), out.shape[-1]


@torch.no_grad()
def refit_table(model, h_int, target):
    """Majority vote per realised cell, which is the exact optimum of the hard objective given the cells. Leaving M
    to gradient descent alone lets it absorb gradient by growing its logits, which starves T and the readouts."""
    idx = model.cell_index(h_int)
    M = torch.zeros_like(model.M)
    for cell in idx.unique():
        M[cell, target[idx == cell].bincount(minlength=model.n_classes).argmax()] = 1.0
    model.M.data = M
    return model


def decision_subspace(to_layer, from_layer, X, d, method="weights", n_jacobian=512):
    """An orthonormal basis (d, p) for the directions of the layer that the decision depends on, without fitting.

    The decision depends on h only through differences of logits, so the object is the Jacobian of the tail with the
    common shift removed. `method="jacobian"` stacks that over inputs and takes the SVD; `method="weights"` is its
    closed form when the tail is linear -- multiply the tail's weight matrices, ignoring activations, centre the
    rows, and take the row space. For a linear tail the rank is exactly n_classes - 1.

    This is not part of the model above. It is useful as a starting point for `fit(init_T=...)` and as a control:
    wherever the tail is near-linear the two methods agree and learning T adds nothing over linear algebra.
    """
    H = to_layer(X).detach()
    if method == "weights":
        W = None
        for mod in _tail_modules(from_layer):
            if isinstance(mod, nn.Linear):
                W = mod.weight.detach() if W is None else mod.weight.detach() @ W
        if W is None:
            raise ValueError("no Linear layers found in the tail; use method='jacobian'")
        M = (W - W.mean(0, keepdim=True)).numpy()
    elif method == "jacobian":
        rows = []
        for h in H[:n_jacobian]:
            J = torch.autograd.functional.jacobian(lambda z: from_layer(z.unsqueeze(0))[0], h)
            rows.append((J - J.mean(0, keepdim=True)).numpy())
        M = np.concatenate(rows, 0)
    else:
        raise ValueError(f"unknown method {method!r}")
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    return vt[:d], s


def _tail_modules(from_layer):
    """The modules behind the closure that `split_sequential` returns."""
    if isinstance(from_layer, nn.Module):
        return list(from_layer.modules())
    for cell in (from_layer.__closure__ or []):
        obj = cell.cell_contents
        if isinstance(obj, nn.Module):
            return list(obj)
    raise ValueError("could not recover the tail modules; pass an nn.Module as from_layer")


def fit(to_layer, from_layer, X, specs, family="single", steps=3000, n_interventions=4096, lr=1e-2, kappa=8.0,
        margin_weight=0.0, margin=0.5, dense_scale=2.0, seed=0, decision=None, n_classes=None,
        freeze_directions=False, init_T=None, verbose=True, log_every=500):
    """Learn T, the readouts and the mechanism so that the high-level model reproduces the network's decision under
    interchange interventions.

    `specs` is a list of (m_i, d_i), or an int k for k binary variables. The prefix up to the chosen layer is shared
    by both models and is never trained, so h = to_layer(X) is computed once. A fixed set of interventions is drawn
    once; each step rebuilds those intervened activations with the current T, takes the network's own decision there
    as the target, and pushes the soft high-level prediction towards it. kappa is a constant, made dimensionless by
    the spread of the projections: it is a property of the model, not a schedule, and annealing it only makes the
    objective non-stationary.

    `init_T` supplies an orthonormal basis (D, p) to start from, and `freeze_directions` keeps it fixed so only the
    readouts and the mechanism are fitted. Freezing also makes the targets constant for the whole run, since
    h_patched then depends on nothing that moves.

    Returns (model, history) with history a list of (step, loss, IIA).
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        H = to_layer(X).detach()
    if decision is None:
        decision, inferred = infer_decision(from_layer, H)
        n_classes = n_classes or inferred
    if n_classes is None:
        raise ValueError("n_classes is required when a decision function is supplied")

    scale = float(H.std(dim=0).pow(2).mean().sqrt())  # spread of t . h for a generic unit direction
    model = MLPAbstraction(H.shape[1], specs, n_classes, kappa=kappa, scale=scale, seed=seed)
    if init_T is not None:
        with torch.no_grad():
            model.proj.weight = torch.as_tensor(init_T, dtype=model.proj.weight.dtype)
    params = ([q for name, q in model.named_parameters() if not name.startswith("proj")] if freeze_directions
              else list(model.parameters()))
    optimiser = torch.optim.Adam(params, lr=lr)
    triples = intervention_set(model, H, family, n_interventions, generator, dense_scale)
    history = []

    for step in range(steps):
        h_int = apply_interventions(model, *triples)
        with torch.no_grad():
            target = decision(from_layer(h_int))
        loss = F.cross_entropy(model(h_int), target)
        if margin_weight:
            loss = loss + margin_weight * F.relu(margin - model.margin(h_int) / model.scale).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                iia = (model.hard_forward(h_int).argmax(-1) == target).double().mean().item()
            history.append((step, loss.item(), iia))
            if verbose:
                print(f"    step {step:5d}  loss {loss.item():.4f}  IIA {iia:.3f}")

    h_int = apply_interventions(model, *triples)
    with torch.no_grad():
        target = decision(from_layer(h_int))
        model.train_cells = set(model.cell_index(h_int).tolist())
    refit_table(model, h_int, target)
    return model, history


@torch.no_grad()
def evaluate(model, to_layer, from_layer, X, families=INTERCHANGE_FAMILIES, n=4096, seed=1, decision=None,
             dense_scale=2.0):
    """IIA of the high-level model per intervention family, plus the diagnostics that say whether to believe it.

    iia        agreement between the hard high-level prediction and the network's decision at the same intervened h'
    base_rate  what a constant high-level model scores on the same sample; IIA alone is not comparable across
               networks, since one whose decision is nearly always the same class is abstracted at high IIA by a
               single cell
    skill      (iia - base_rate) / (1 - base_rate): 0 is no better than constant, 1 is perfect
    purity     mean share of the majority decision within a realised cell. Purity is the well-definedness condition
               measured at the output row: below 1, two activations with the same high-level state give different
               outcomes, so no mechanism over these cells can be valid, whatever the IIA
    margin     distance to the nearest decision boundary in projection-spread units, and `sharp` = kappa * margin;
               the relaxation is faithful only where that product is >> 1
    cells      realised cells out of prod_i m_i, and the share of examples whose cell the fit never visited -- on
               those, IIA is measuring extrapolation rather than abstraction
    """
    generator = torch.Generator().manual_seed(seed)
    H = to_layer(X).detach()
    if decision is None:
        decision, _ = infer_decision(from_layer, H)
    out = {}
    for family in families:
        triples = intervention_set(model, H, family, n, generator, dense_scale)
        h_int = apply_interventions(model, *triples)
        target = decision(from_layer(h_int))
        pred = model.hard_forward(h_int).argmax(-1)
        idx = model.cell_index(h_int)
        margin = model.margin(h_int).min(dim=-1).values / model.scale
        purity = [target[idx == c].bincount(minlength=model.n_classes).max().item() / (idx == c).sum().item()
                  for c in idx.unique()]
        seen = getattr(model, "train_cells", None)
        base_rate = (target.bincount(minlength=model.n_classes).max() / len(target)).item()
        iia = (pred == target).double().mean().item()
        out[family] = {
            "iia": iia,
            "base_rate": base_rate,
            "skill": (iia - base_rate) / max(1.0 - base_rate, 1e-12),
            "purity": float(np.mean(purity)),
            "margin_min": margin.min().item(),
            "margin_mean": margin.mean().item(),
            "sharp": float(model.kappa) * margin.mean().item(),
            "cells": len(idx.unique()),
            "cells_total": model.n_cells,
            "unseen": 0.0 if seen is None else float(np.mean([i not in seen for i in idx.tolist()])),
        }
    return out


def graded_baseline(model, to_layer, from_layer, X, family="single", steps=600, n=4096, lr=1e-2, seed=2,
                    decision=None, dense_scale=2.0):
    """Accuracy of a *linear* readout of the same projections, as a check on the discrete reading.

    If this beats the high-level model by a wide margin, the layers after the cut use the magnitude along these
    directions and not just which region they fall in, so no number of values on this T will do. Scored on a
    held-out sample, since the IIA it is compared against is out of sample too.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        H = to_layer(X).detach()
        if decision is None:
            decision, _ = infer_decision(from_layer, H)
        batches = []
        for _ in range(2):
            triples = intervention_set(model, H, family, n, generator, dense_scale)
            h_int = apply_interventions(model, *triples)
            batches.append((model.proj(h_int), decision(from_layer(h_int))))
    (z, target), (z_eval, target_eval) = batches
    head = nn.Linear(model.D, model.n_classes)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(head(z), target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        return (head(z_eval).argmax(-1) == target_eval).double().mean().item()


def truth_table(model):
    """The learned mechanism as a (prod_i m_i, k + 1) array: each joint assignment and the class it names."""
    with torch.no_grad():
        pred = model.M.argmax(-1)
    return np.column_stack([model.cells.numpy().astype(int), pred.numpy()])


def degeneracy(H, k):
    """(distinct activations, free parameters, shatterable) for a fit of k directions on the activations H.

    k halfspaces in R^p carry k(p+1) free parameters before orthonormality, so unless the number of distinct
    activations the condition is evaluated on greatly exceeds that, IIA = 1 says nothing about the network: the
    hypothesis class can fit any labelling. `shatterable` is True when the activations number no more than p + 1, in
    which case a single halfspace realises *every* labelling of them and even k = 1 is vacuous.
    """
    n_distinct = len(torch.unique(H, dim=0))
    p = H.shape[1]
    return n_distinct, k * (p + 1), n_distinct <= p + 1


def subspace_error(model, U):
    """||U U^T - T T^T||_F, the distance between the learned span and a known one. Invariant to rotations within the
    subspace, so it is the right score when the planted directions are only identified up to sign and permutation."""
    with torch.no_grad():
        T = model.T.numpy()
    U = np.asarray(U)
    if U.shape[0] != T.shape[0]:
        U = U.T
    return float(np.linalg.norm(U @ U.T - T @ T.T))
