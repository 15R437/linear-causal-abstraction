"""Is "drop one_child, restart until a run finds an exact abstraction" viable?

THE PROPOSAL. Run HailMary n times independently on the same A_low with one_child OFF; with
some probability one run lands a partition that satisfies All-In and admits an exact
abstraction. Keep that one.

WHY THE HEADLINE NUMBER DOES NOT SETTLE IT. P(exact | All-In holds) = 0.73 for graph_only is
CONDITIONAL on All-In, which itself held on only 37% of draws; unconditionally the arm lands
All-In AND exact on ~27% of single draws. Whether n restarts turn 27% into ~92% depends
entirely on a question the single-draw data cannot answer:

    are the failures a property of the SEED (restarts help) or of the INSTANCE (they do not)?

If every instance succeeds ~27% of the time, 8 draws give 1 - 0.73^8 ~ 92%. If 27% of
instances always succeed and 73% never do, 8 draws give 27% -- and the report's "5 restarts
== 1 restart in 7 of 8 density cells" is evidence for exactly that. Section 2 decomposes the
per-draw rate into never / mixed / always instances, which separates the two cases.

Three success criteria are tracked, from strictest to loosest:
  allin_exact  All-In holds AND an exact abstraction was found   <- the proposal
  exact        an exact abstraction was found (All-In or not)
  accept       rel_err <= 0.05, the report's alpha rule (approximate abstractions allowed)

Exactness is scored as "EITHER fitter found it", since a fitter failing to find an
abstraction is not proof that none exists -- and fit_abstraction_alternative runs SLSQP to a
budget, so it misses the 1e-8 bar on partitions where fit_abstraction reaches it analytically.
"""
import sys
import numpy as np, pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction'
pd.set_option('display.width', 250)
KEY = ['p', 'density', 'seed']
DRAW = KEY + ['arm', 'restart']
ARMS = ['oc_info', 'no_oc_info', 'no_oc_graph', 'no_oc_g_info', 'no_oc_g_recon']
TOL, ALPHA = 1e-8, 0.05
CRIT = ['allin_exact', 'exact', 'accept']

raw = pd.read_csv(f'{B}/experiments/abstraction_study/data/nochild_restarts.csv')
raw = raw[raw.get('fit_failed', 0) == 0]
raw['exact_f'] = (raw.obs < TOL) & (raw.intv < TOL)

# One row per DRAW, with exactness taken as the union over fitters and the alpha test
# reported per fitter (the alpha rule is applied to whichever fitter you actually ship).
piv = raw.pivot_table(index=DRAW, columns='fitter',
                      values=['exact_f', 'rel_err', 'rel_err_worst', 'info_eff_t2',
                              'survivors', 'survivor_frac', 'k_alive'])
piv.columns = ['%s_%s' % (a, b) for a, b in piv.columns]
flat = raw.groupby(DRAW).agg(allin=('allin', 'first'), one_child=('one_child', 'first'),
                             valid=('valid', 'first'), k_used=('k_used', 'first'),
                             largest=('largest', 'first'), intra=('info', 'first'))
d = flat.join(piv)
# pivot_table averages the bool, so cast back; a missing fitter row counts as "not found"
for c in ('exact_f_reg', 'exact_f_alt'):
    d[c] = d[c].fillna(0) > 0.5
d['exact'] = d['exact_f_reg'] | d['exact_f_alt']
d['allin_exact'] = d.exact & d.allin.astype(bool)
d['accept'] = d['rel_err_alt'] <= ALPHA
d['accept_reg'] = d['rel_err_reg'] <= ALPHA
d['accept_worst'] = d['rel_err_worst_alt'] <= ALPHA
d = d.reset_index()
R = int(d.restart.max()) + 1
print(f'{len(raw)} rows -> {len(d)} draws   {d[KEY].drop_duplicates().shape[0]} instances   '
      f'R={R} restarts/arm   TOL={TOL:g}  ALPHA={ALPHA}')

print('\n' + '='*104 + '\n1. PER-DRAW RATES (each restart counted separately)\n' + '='*104)
print(d.groupby('arm').agg(
    n=('exact', 'size'), allin=('allin', 'mean'), one_child=('one_child', 'mean'),
    k_used=('k_used', 'mean'), largest=('largest', 'mean'), intra=('intra', 'mean'),
    exact=('exact', 'mean'), allin_exact=('allin_exact', 'mean'),
    accept=('accept', 'mean'), accept_worst=('accept_worst', 'mean')
).reindex(ARMS).round(3).to_string())
print('\n  P(exact | All-In holds) and P(exact | All-In violated), per arm:')
for arm in ARMS:
    s = d[d.arm == arm]
    a, v = s[s.allin == 1], s[s.allin == 0]
    print(f'    {arm:15s} All-In holds n={len(a):3d} -> exact {a.exact.mean() if len(a) else float("nan"):.2f}   '
          f'violated n={len(v):3d} -> exact {v.exact.mean() if len(v) else float("nan"):.2f}')

print('\n' + '='*104 +
      '\n2. THE CRUX: per-SEED or per-INSTANCE failures?  (does restarting buy anything?)\n'
      + '='*104)
for crit in CRIT:
    print(f'\n  criterion = {crit}')
    rows = []
    for arm in ARMS:
        s = d[d.arm == arm]
        hit = s.groupby(KEY)[crit].mean()
        n = len(hit)
        never, always = int((hit == 0).sum()), int((hit == 1).sum())
        per_draw, any_of_R = float(s[crit].mean()), float((hit > 0).mean())
        # what an independence assumption would predict for R draws
        indep = 1 - (1 - per_draw) ** R
        rows.append(dict(arm=arm, per_draw=round(per_draw, 3),
                         never=never, mixed=n - never - always, always=always,
                         any_of_R=round(any_of_R, 3),
                         if_independent=round(indep, 3),
                         restart_gain=round(any_of_R - per_draw, 3)))
    t = pd.DataFrame(rows).set_index('arm')
    print(t.to_string())
    print('    never/mixed/always = instances where NO draw, SOME draws, ALL draws succeeded.')
    print('    mixed=0 means success is fixed by the instance and restarts are wasted;')
    print('    any_of_R close to if_independent means restarts behave like independent tries.')

print('\n' + '='*104 + '\n3. WHAT YOU GET WHEN IT WORKS (accepted/exact draws only)\n' + '='*104)
for crit in ('allin_exact', 'accept'):
    print(f'\n  criterion = {crit}')
    rows = []
    for arm in ARMS:
        s = d[(d.arm == arm) & d[crit]]
        if not len(s):
            rows.append(dict(arm=arm, hits=0)); continue
        rows.append(dict(
            arm=arm, hits=len(s),
            k_used=round(float(s.k_used.mean()), 1),
            compression=round(float((s.k_used / s.p).mean()), 2),
            largest=round(float(s.largest.mean()), 1),
            intra=round(float(s.intra.mean()), 3),
            survivors=round(float(s['survivors_alt'].mean()), 1),
            surv_frac=round(float(s['survivor_frac_alt'].mean()), 3),
            info_eff_t2=round(float(s['info_eff_t2_alt'].mean()), 3)))
    print(pd.DataFrame(rows).set_index('arm').to_string())

print('\n' + '='*104 + '\n4. BEST-OF-R vs OCC  (pick the best qualifying draw per instance)\n' + '='*104)
occ = raw[(raw.arm == 'occ') & (raw.fitter == 'alt')].set_index(KEY)
for crit in ('allin_exact', 'accept'):
    print(f'\n  criterion = {crit}   (retention = info_eff_t2, lower is better)')
    for arm in ARMS:
        s = d[(d.arm == arm) & d[crit]].copy()
        s['ie'] = s['info_eff_t2_alt']
        base = occ.info_eff_t2
        if not len(s):
            print(f'    {arm:15s} no qualifying draws   (OCC {base.mean():.3f})'); continue
        best = s.loc[s.groupby(KEY).ie.idxmin()].set_index(KEY).ie
        chosen = best.reindex(base.index)
        hybrid = np.where(chosen.notna(), chosen, base)
        print(f'    {arm:15s} instances with a hit {chosen.notna().sum():2d}/{len(base):2d}   '
              f'best-of-R retention {best.mean():.3f}   hybrid {np.nanmean(hybrid):.3f}   '
              f'OCC {base.mean():.3f}   ({100*(np.nanmean(hybrid)-base.mean())/base.mean():+.0f}%)')

print('\n' + '='*104 + '\n5. BY DENSITY and INIT NOISE\n' + '='*104)
nz = raw.groupby(DRAW).noise.first().reset_index()
d2 = d.merge(nz, on=DRAW, how='left')
for crit in ('allin_exact', 'accept'):
    print(f'\n  {crit} per-draw rate by density:')
    print(d2.pivot_table(index='density', columns='arm', values=crit).reindex(columns=ARMS).round(2).to_string())
    print(f'  {crit} per-draw rate by init noise:')
    print(d2.pivot_table(index='noise', columns='arm', values=crit).reindex(columns=ARMS).round(2).to_string())
