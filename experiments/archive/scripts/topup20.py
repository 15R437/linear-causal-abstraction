"""Fill frontier2.csv to exactly 20 instances per density.

d=0.1 lagged because sample_low_level_graph(connected=True) FAILS on some low-density
seeds -- frontier_fig.job() returns [] and the seed is silently lost, so seeds 0..19 do not
yield 20 instances. This first scans which seeds actually generate, then takes the first 20
generable seeds per density and runs whichever are missing. Configuration is untouched so
new rows merge with existing ones.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
import frontier_fig as F

PATH = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/frontier2.csv'
TARGET, DENS, SCAN_TO = 20, (0.1, 0.3, 0.5, 0.7, 0.9), 70


def generable(a):
    d, s = a
    try:
        sample_low_level_graph(n_high=10, n_low=100, density=d, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type='random', seed=s,
            connected=True, max_attempts=300)
        return (d, s, 1)
    except Exception:
        return (d, s, 0)


if __name__ == '__main__':
    print('scanning seeds 0-%d for generability...' % (SCAN_TO - 1), flush=True)
    with mp.Pool(6) as pool:
        scan = pd.DataFrame(pool.map(generable, [(d, s) for d in DENS for s in range(SCAN_TO)]),
                            columns=['density', 'seed', 'ok'])
    print(scan.groupby('density').ok.agg(['sum', 'size']).to_string(), flush=True)

    have = pd.read_csv(PATH)
    done = {(round(float(a), 3), int(b)) for a, b in have[['density', 'seed']].drop_duplicates().values}
    jobs, keep = [], []
    for d in DENS:
        good = scan[(scan.density == d) & (scan.ok == 1)].seed.tolist()[:TARGET]
        keep += [(d, s) for s in good]
        jobs += [(d, s) for s in good if (round(d, 3), s) not in done]
    print('\ntarget %d per density = %d instances; have %d of them; running %d\n'
          % (TARGET, len(keep), len(keep) - len(jobs), len(jobs)), flush=True)

    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(F.job, jobs), 1):
            rows.extend(rs)
            pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(PATH, index=False)
            print('%d/%d  %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    # drop any instance outside the chosen 20-per-density set, so panels are balanced
    keepset = {(round(d, 3), s) for d, s in keep}
    out = out[[(round(float(r.density), 3), int(r.seed)) in keepset for r in out.itertuples()]]
    out.to_csv(PATH, index=False)
    print('\ndone %.1f min' % ((time.time()-t0)/60))
    print(out.drop_duplicates(['density', 'seed']).density.value_counts().sort_index().to_string())
