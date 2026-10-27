"""Across many trained XOR MLPs: how many of the paper's IIA-passing explanations satisfy (AC')?

For each network (trained exactly as in reproduce_counts.py / reproduce_counts_noisy.py) we count minimal mappings that pass:
  iia_their_code   IIA in MI-identifiability's released code (one ordering of each input pair)
  iia_all_pairs    the same search with both orderings: the paper's own Definition 6
  ac               (AC') with Boolean high-level variables over every joint interchange intervention (ac_boolean.py, `all`),
                   checked on every mapping found by either search (any (AC')-valid mapping passes IIA on all pairs)

Usage: python -I reproduce_ac.py <MI-identifiability clone> <out.jsonl> <n_workers>
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, OUT, N_WORKERS = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

SETTINGS = [  # (noise, learning rate, loss target, data repeats): the two earlier sweeps
    *[(0.0, 1e-3, lt, 1000) for lt in (0.01, 0.001)],
    *[(0.1, lr, 0.01, 50) for lr in (0.01, 0.001)],
]


def records(model, formula_dataset, n_layers, both=False):
    from mi_identifiability.mappings import find_minimal_mappings
    out = []
    for formula, pairs, trees in formula_dataset:
        if both:
            pairs = pairs + [(b, a) for a, b in pairs]
        for mapping in find_minimal_mappings(model, formula, pairs, trees):
            neuron = {int(k.split(" ")[-1]): [(l, i) for l, ii in v.items() for i in ii] for k, v in mapping.items()}
            for leaf in formula.get_leaf_nodes():
                neuron[leaf.id] = [(0, ord(leaf.op[-1]) - ord("A"))]
            neuron[formula.id] = [(n_layers, 0)]

            def mapped_parents(node):
                ps = []
                for c in node.children:
                    ps += [c.id] if c.id in neuron else mapped_parents(c)
                return ps

            out.append({"formula": str(formula), "root": formula.id,
                        "nodes": [{"id": n.id, "label": str(n), "neurons": neuron[n.id], "parents": mapped_parents(n)}
                                  for n in formula.get_nodes() if n.id in neuron],
                        "tree": {n.id: {"op": n.op, "children": [c.id for c in n.children]} for n in formula.get_nodes()}})
    return out


def job(args):
    noise, lr, lt, reps, seed = args
    import torch
    torch.set_num_threads(1)
    from mi_identifiability.logic_gates import ALL_LOGIC_GATES, get_formula_dataset
    from mi_identifiability.neural_model import MLP
    from mi_identifiability.utils import set_seeds
    import ac_check
    from ac_boolean import count_solutions

    t0 = time.time()
    gate = ALL_LOGIC_GATES["XOR"]
    set_seeds(seed)
    x, y = gate.generate_noisy_data(n_repeats=reps, noise_std=noise, device="cpu")
    xv, yv = gate.generate_noisy_data(n_repeats=50, noise_std=noise, device="cpu")
    set_seeds(seed)
    m = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    loss = m.do_train(x=x, y=y, x_val=xv, y_val=yv, batch_size=100, learning_rate=lr, epochs=1000, loss_target=lt)
    acc = m.do_eval(xv, yv)
    rec = {"noise": noise, "lr": lr, "loss_target": lt, "seed": seed, "converged": bool(acc >= 0.99 and loss <= lt)}
    if rec["converged"]:
        # point the (AC') checker at this network's weights
        ac_check.Ws = [m.layers[l][0].weight.detach().numpy().astype(float) for l in range(len(m.layers))]
        ac_check.bs = [m.layers[l][0].bias.detach().numpy().astype(float) for l in range(len(m.layers))]
        fd = get_formula_dataset([gate], max_depth=3, device="cpu")["XOR"]
        one, both = records(m, fd, len(m.layers)), records(m, fd, len(m.layers), both=True)
        key = lambda r: (r["formula"], json.dumps(sorted((n["id"], n["neurons"]) for n in r["nodes"])))
        union = list({key(r): r for r in one + both}.values())
        ac = [r for r in union if count_solutions(r, "all")[0] > 0]
        rec.update(iia_their_code=len(one), iia_all_pairs=len(both), ac=len(ac),
                   formulas_their_code=len({r["formula"] for r in one}), formulas_iia_all_pairs=len({r["formula"] for r in both}),
                   formulas_ac=len({r["formula"] for r in ac}), ac_survivors=[(r["formula"], {n["label"]: n["neurons"] for n in r["nodes"] if n["parents"]}) for r in ac])
    rec["secs"] = round(time.time() - t0)
    return rec


if __name__ == "__main__":
    jobs = [(*s, seed) for s in SETTINGS for seed in range(50 if s[0] > 0 else 20)]
    with Pool(N_WORKERS) as pool, open(OUT, "w") as fh:
        for rec in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
