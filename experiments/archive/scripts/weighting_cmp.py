"""Sigma_u = I vs the generator's own noise variances, on IDENTICAL partitions.

frontier.csv scored RAE with su estimated from sampled data; frontier2.csv scores it with
Sigma_u = I. Training is seeded identically for the shared lambdas, so the partitions
coincide and the weighting is the only difference -- a properly paired comparison, unlike
relerr_sensitivity.csv where only 3 of 19 instances had non-dust error.
"""
import pandas as pd, numpy as np
B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data'
KEY = ['density', 'seed', 'method', 'lam_oc']
a = pd.read_csv(f'{B}/frontier.csv')          # empirical Sigma_u
b = pd.read_csv(f'{B}/frontier2.csv')         # Sigma_u = I
for x in (a, b):
    x['lam_oc'] = x.lam_oc.fillna(-1)
j = a.merge(b, on=KEY, suffixes=('_gen', '_iso'))
# the partitions must be identical for the comparison to be paired
same = np.isclose(j.iem_gen, j.iem_iso) & (j.k_used_gen == j.k_used_iso)
print('paired rows: %d   partitions identical on %d of them' % (len(j), int(same.sum())))
j = j[same & (j.rae_gen.notna()) & (j.rae_iso.notna())]
print('usable: %d rows, %d instances\n' % (len(j), j[['density', 'seed']].drop_duplicates().shape[0]))

print('RAE medians by density:')
print(j.groupby('density').agg(n=('rae_gen', 'size'),
      gen=('rae_gen', 'median'), iso=('rae_iso', 'median')).round(4).to_string())

nz = j[j.rae_gen > 1e-6]
print('\nratio Sigma_u=I / generator, on rows with non-dust generator error (n=%d):' % len(nz))
if len(nz):
    nz = nz.assign(ratio=nz.rae_iso / nz.rae_gen)
    print(nz.groupby('density').ratio.agg(['size', 'median', 'min', 'max']).round(2).to_string())
    print('\n   overall median %.2fx   IQR %.2f-%.2f   (n=%d)' % (
        nz.ratio.median(), nz.ratio.quantile(.25), nz.ratio.quantile(.75), len(nz)))
    print('   fraction where Sigma_u=I is LARGER: %.2f' % (nz.ratio > 1).mean())

print('\nhow many rows clear alpha=0.05 under each weighting?')
print(j.groupby('density').agg(gen=('rae_gen', lambda s: float((s <= 0.05).mean())),
                               iso=('rae_iso', lambda s: float((s <= 0.05).mean())),
                               n=('rae_gen', 'size')).round(3).to_string())
