"""Add seeds to frontier2.csv, identical configuration so old and new rows merge cleanly.

Config is NOT touched (same lambdas, epochs, k=|pi_0|, Sigma_u=I, process_output at
n_restarts=5/maxiter=500): changing epochs or the fit budget to go faster would make the new
rows incomparable with the existing ones.
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
import pandas as pd, multiprocessing as mp
import frontier_fig as F

PATH = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/frontier2.csv'
SEEDS, DENS = 12, (0.1, 0.3, 0.5, 0.7, 0.9)

if __name__ == '__main__':
    have = pd.read_csv(PATH)
    done = {(round(float(a), 3), int(b)) for a, b in have[['density', 'seed']].drop_duplicates().values}
    jobs = [(d, s) for d in DENS for s in range(SEEDS) if (round(d, 3), s) not in done]
    print('have %d instances; adding %d -> target %d seeds x %d densities'
          % (len(done), len(jobs), SEEDS, len(DENS)), flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(F.job, jobs), 1):
            rows.extend(rs)
            pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(PATH, index=False)
            print('%d/%d  %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    print('done %.1f min' % ((time.time()-t0)/60))
