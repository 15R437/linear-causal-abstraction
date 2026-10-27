"""Train XOR MLPs as in MI-identifiability's main.py basic setup and count minimal mappings (their pairs, and both directions)."""
import json, sys, time
from multiprocessing import Pool
R = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, R)

def job(args):
    lt, off = args
    import torch; torch.set_num_threads(1)
    from mi_identifiability.logic_gates import ALL_LOGIC_GATES, get_formula_dataset
    from mi_identifiability.mappings import find_minimal_mappings
    from mi_identifiability.neural_model import MLP
    from mi_identifiability.utils import set_seeds
    t0 = time.time()
    gate = ALL_LOGIC_GATES['XOR']
    set_seeds(off)
    x, y = gate.generate_noisy_data(n_repeats=1000, noise_std=0.0, device='cpu')
    xv, yv = gate.generate_noisy_data(n_repeats=50, noise_std=0.0, device='cpu')
    set_seeds(off)
    m = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device='cpu')
    loss = m.do_train(x=x, y=y, x_val=xv, y_val=yv, batch_size=100, learning_rate=1e-3, epochs=1000, loss_target=lt)
    acc = m.do_eval(xv, yv)
    rec = {'loss_target': lt, 'seed': off, 'loss': float(loss), 'acc': acc, 'converged': bool(acc >= 0.99 and loss <= lt)}
    if rec['converged']:
        fd = get_formula_dataset([gate], max_depth=3, device='cpu')['XOR']
        one, both, f1, f2 = 0, 0, set(), set()
        for f, pairs, trees in fd:
            a = find_minimal_mappings(m, f, pairs, trees)
            b = find_minimal_mappings(m, f, pairs + [(q, p) for p, q in pairs], trees)
            one += len(a); both += len(b)
            if a: f1.add(str(f))
            if b: f2.add(str(f))
        rec.update(mappings_one_dir=one, formulas_one_dir=len(f1), mappings_both_dirs=both, formulas_both_dirs=len(f2))
    rec['secs'] = round(time.time() - t0)
    return rec

if __name__ == '__main__':
    jobs = [(lt, off) for off in range(20) for lt in (0.01, 0.001)]
    with Pool(int(sys.argv[3])) as pool, open(OUT, 'w') as fh:
        for rec in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
