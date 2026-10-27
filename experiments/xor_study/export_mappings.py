"""Export the pretrained XOR MLP and its IIA=1 mappings from MI-identifiability (Méloux et al., arXiv 2502.20914).

Usage: python -I export_mappings.py <path to a clone of github.com/MelouxM/MI-identifiability>

Runs their own `find_minimal_mappings` on their pretrained `xor_mlp.pt` (2-3-3-1, LeakyReLU) for every XOR formula of
depth <= 3, and writes, for each mapping, the high-level graph with each node's low-level neurons (its cluster) and its high-level
parents. Formula nodes that their search never maps (the child of a negation) are merged into their parent, so the
parents of a mapped node are its nearest mapped or leaf descendants in the formula tree. Leaves map to input neurons and
the formula root to the output neuron.
"""
import json
import os
import sys

import torch

repo = sys.argv[1]
sys.path.insert(0, repo)
from mi_identifiability.logic_gates import ALL_LOGIC_GATES, get_formula_dataset
from mi_identifiability.mappings import find_minimal_mappings
from mi_identifiability.neural_model import MLP

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

d = torch.load(os.path.join(repo, "xor_data", "xor_mlp.pt"), weights_only=True, map_location="cpu")
model = MLP(d["hidden_sizes"], input_size=d["input_size"], output_size=d["output_size"], activation=d["activation"],
            device="cpu")
model.load_state_dict(d["state_dict"])
n_layers = len(model.layer_sizes) - 1
weights = {"activation": d["activation"], "negative_slope": 0.01, "layer_sizes": model.layer_sizes,
           "W": [model.layers[l][0].weight.detach().tolist() for l in range(n_layers)],
           "b": [model.layers[l][0].bias.detach().tolist() for l in range(n_layers)]}
json.dump(weights, open(os.path.join(OUT, "xor_mlp_weights.json"), "w"), indent=1)

records = []
for formula, pairs, trees in get_formula_dataset([ALL_LOGIC_GATES["XOR"]], max_depth=3, device="cpu")["XOR"]:
    for mapping in find_minimal_mappings(model, formula, pairs, trees):
        neuron = {}
        for key, layers in mapping.items():
            neuron[int(key.split(" ")[-1])] = [(l, i) for l, ii in layers.items() for i in ii]
        for leaf in formula.get_leaf_nodes():
            neuron[leaf.id] = [(0, ord(leaf.op[-1]) - ord("A"))]
        neuron[formula.id] = [(n_layers, 0)]

        def mapped_parents(node):
            out = []
            for c in node.children:
                out += [c.id] if c.id in neuron else mapped_parents(c)
            return out

        nodes = [{"id": n.id, "label": str(n), "neurons": neuron[n.id], "parents": mapped_parents(n)}
                 for n in formula.get_nodes() if n.id in neuron]
        tree = {n.id: {"op": n.op, "children": [c.id for c in n.children]} for n in formula.get_nodes()}
        assert all(t["op"] in ("*", "+", "¬", "A", "B") for t in tree.values()), "only AND/OR/NOT and plain leaves"
        records.append({"formula": str(formula), "root": formula.id, "nodes": nodes, "tree": tree})

json.dump(records, open(os.path.join(OUT, "xor_mappings.json"), "w"), indent=1)
print(f"{len(records)} mappings over {len({r['formula'] for r in records})} formulas -> {OUT}")
