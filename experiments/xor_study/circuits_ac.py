"""Do Méloux et al.'s where-then-what interpretations (circuit + per-neuron gates) satisfy (AC')?

For their pretrained XOR MLP we run their own circuit search (`find_circuits`) and grounding (`Circuit.ground`), keeping the
per-neuron threshold ("separator") and local truth table that their `Grounding` object discards. Each interpretation is a
Boolean high-level model on the circuit's neurons: neuron i is the bit 1[x_i > separator_i], and its mechanism is its local
gate over its circuit parents (the inputs, for the first layer). The low-level model is the circuit as they run it, with
non-circuit neurons and edges ablated.

(AC') with joint interchange interventions: for every subset of the circuit's hidden neurons, each patched with its
activation on any of the 4 inputs, and every base input, every non-intervened neuron's bit (and the rounded output) must
equal its gate applied to its parents' bits. Their interpretation step only checks the 4 un-intervened inputs.

Usage: python -I circuits_ac.py <MI-identifiability clone>
"""
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

REPO = sys.argv[1]
sys.path.insert(0, REPO)
import mi_identifiability.circuit as circuit_mod
from mi_identifiability.circuit import find_circuits
from mi_identifiability.neural_model import MLP

HERE = os.path.dirname(os.path.abspath(__file__))


class KeepingGrounding(dict):
    """Their Grounding, but also keeping each neuron's separator, local truth table and parent order."""
    def __init__(self, assignments, circuit):
        super().__init__({str(a["neuron_info"]).replace(" ", ""): a["gate_name"] for a in assignments})
        self.circuit = circuit
        self.assignments = assignments


circuit_mod.Grounding = KeepingGrounding

d = torch.load(os.path.join(REPO, "xor_data", "xor_mlp.pt"), weights_only=True, map_location="cpu")
model = MLP(d["hidden_sizes"], input_size=d["input_size"], output_size=d["output_size"], activation=d["activation"], device="cpu")
model.load_state_dict(d["state_dict"])
W = [model.layers[l][0].weight.detach().numpy().astype(float) for l in range(len(model.layers))]
b = [model.layers[l][0].bias.detach().numpy().astype(float) for l in range(len(model.layers))]
L = len(W)
INPUTS = np.array(list(itertools.product([0.0, 1.0], repeat=2)))
X4, Y4 = torch.tensor(INPUTS, dtype=torch.float32), torch.tensor([[0.0], [1.0], [1.0], [0.0]])


def phi(z):
    return np.where(z >= 0, z, 0.01 * z)


def circuit_forward(c, x, iv):
    """Batched forward pass of the circuit as their MLP.forward runs it. iv: {(layer, idx): array of values or NaN}."""
    acts = [x]
    for l in range(L):
        h = acts[-1] * np.asarray(c.node_masks[l], float)
        h = phi(h @ (W[l] * np.asarray(c.edge_masks[l], float)).T + b[l])
        for (layer, idx), v in iv.items():
            if layer == l + 1:
                h[:, idx] = np.where(np.isnan(v), h[:, idx], v)
        acts.append(h)
    return acts


def configs(c):
    """All joint interchange interventions on the circuit's hidden neurons. Returns activations, intervened masks."""
    hidden = [(l, i) for l in range(1, L) for i, m in enumerate(c.node_masks[l]) if m]
    clean = circuit_forward(c, INPUTS, {})
    bases, ivs = [], {h: [] for h in hidden}
    for k in range(len(hidden) + 1):
        for theta in itertools.combinations(hidden, k):
            for srcs in itertools.product(range(4), repeat=k):
                for base in range(4):
                    bases.append(base)
                    src = dict(zip(theta, srcs))
                    for h in hidden:
                        ivs[h].append(clean[h[0]][src[h], h[1]] if h in src else np.nan)
    ivs = {h: np.array(v) for h, v in ivs.items()}
    acts = circuit_forward(c, INPUTS[bases], ivs)
    intervened = {h: ~np.isnan(v) for h, v in ivs.items()}
    return acts, intervened, np.array(bases)


def ac_valid(grounding, acts, intervened, bases, cache):
    """(AC') check of one interpretation, row by row (rows are cached across interpretations)."""
    sep = {a["neuron_info"]: a["separator"] for a in grounding.assignments}
    order = [a["neuron_info"] for a in grounding.assignments]
    for a in grounding.assignments:
        l, i = a["neuron_info"]
        if a["par_n"] == "inputs":
            parents = None
        else:  # their local truth tables are keyed by parents in grounding order
            parents = [p for p in order if p in a["par_n"]]
        key = (a["neuron_info"], sep[(l, i)], tuple(sorted(a["local_tt"].items())),
               None if parents is None else tuple((p, sep[p]) for p in parents))
        if key not in cache:
            active = ~intervened.get((l, i), np.zeros(len(bases), bool))
            act = np.round(acts[l][:, i]) if l == L else acts[l][:, i]  # their grounding rounds the output
            bit = (act > sep[(l, i)]).astype(int)
            if parents is None:
                keys = [tuple(int(v) for v in INPUTS[bb]) for bb in bases]
            else:
                pbits = np.stack([(acts[p[0]][:, p[1]] > sep[p]).astype(int) for p in parents], axis=1)
                keys = [tuple(r) for r in pbits]
            want = np.array([a["local_tt"][k] for k in keys])
            cache[key] = bool(np.all((want == bit)[active]))
        if not cache[key]:
            return False
    return True


if __name__ == "__main__":
    circuits, *_ = find_circuits(model, X4, Y4, accuracy_threshold=0.99)
    print(f"{len(circuits)} circuits with perfect accuracy on the 4 inputs")
    total, total_ok, out = 0, 0, []
    for ci, c in enumerate(circuits):
        groundings = c.ground(model(X4, circuit=c, return_activations=True))
        acts, intervened, bases = configs(c)
        cache = {}
        ok = [g for g in groundings if ac_valid(g, acts, intervened, bases, cache)]
        total += len(groundings); total_ok += len(ok)
        n_hidden = sum(int(m) for l in range(1, L) for m in c.node_masks[l])
        survivors = Counter(tuple(sorted(dict(g).items())) for g in ok)
        print(f"circuit {ci:2d}: {n_hidden} hidden neurons, {len(bases):6d} interventions, "
              f"{len(groundings):6d} interpretations -> {len(ok):4d} satisfy (AC')")
        out.append({"circuit": ci, "node_masks": [list(map(int, m)) for m in c.node_masks],
                    "interpretations": len(groundings), "ac_valid": len(ok),
                    "ac_valid_gates": [dict(k) for k in survivors]})
    print(f"\nTotal: {total} interpretations -> {total_ok} satisfy (AC')")
    json.dump(out, open(os.path.join(HERE, "data", "circuits_ac.json"), "w"), indent=1)
