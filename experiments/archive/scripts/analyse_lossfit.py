"""Does switching to fit_abstraction_alternative change any loss conclusion?

Reads lossfit.csv (identical partitions scored under both fitters) and re-asks each loss
claim in REPORT.md of each fitter in turn. Prints one block per claim.
"""
import sys
import numpy as np, pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction'
pd.set_option('display.width', 250)
ORDER = ['occ', 'graph_only', 'info_only', 'oc_info', 'oc_info_g1', 'oc_info_g10',
         'oc_info_g100', 'oc_info_u1', 'oc_info_u100', 'oc_recon', 'oc_recon_g1']
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.25]
KEY = ['p', 'density', 'seed']


def hybrid(d, arm, alpha, fitter, ie='info_eff', err='rel_err'):
    """Accept the arm when its measured error clears alpha AND it retains more than OCC;
    else fall back to OCC. Returns (acceptance rate, hybrid retention, OCC retention)."""
    occ = d[(d.arm == 'occ') & (d.fitter == fitter)].set_index(KEY)[ie]
    g = d[(d.arm == arm) & (d.fitter == fitter)].set_index(KEY)
    base = occ.reindex(g.index)
    ok = (g[err] <= alpha) & (g[ie] < base)
    return float(ok.mean()), float(np.nanmean(np.where(ok, g[ie], base))), float(np.nanmean(base))


def header(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)


d = pd.read_csv(f'{B}/experiments/abstraction_study/data/lossfit.csv')
d = d[d.fit_failed == 0] if 'fit_failed' in d else d
# optional: restrict to one p, e.g. `analyse_lossfit.py 50`, to read a cell on its own
if len(sys.argv) > 1:
    d = d[d.p == int(sys.argv[1])]
    # keep only instances whose every arm x fitter row is present, so the pairing is exact
    full = d.groupby(KEY).size()
    d = d[d.set_index(KEY).index.isin(full[full == full.max()].index)]
n_inst = d[KEY].drop_duplicates().shape[0]
print(f'{len(d)} rows   {n_inst} instances   arms {d.arm.nunique()}   '
      f'p {sorted(d.p.unique())}   density {sorted(d.density.unique())}')

# Partitions are shared, so any structural disagreement between fitters is a bug.
piv = d.pivot_table(index=KEY + ['arm'], columns='fitter', values=['valid', 'k_used', 'info'])
for c in ('valid', 'k_used', 'info'):
    assert np.allclose(piv[(c, 'alt')], piv[(c, 'reg')]), c
print('sanity: structural columns identical across fitters (partitions are shared)  OK')

header('0. WHAT THE FITTER ITSELF DOES  (same partitions, so every gap is the fitter)')
print(d.groupby('fitter').agg(
    n=('arm', 'size'), obs_med=('obs', 'median'), intv_med=('intv', 'median'),
    exact=('obs', lambda s: float((s < 1e-8).mean())),
    rel_err_med=('rel_err', 'median'), rel_err_p90=('rel_err', lambda s: s.quantile(.9)),
    A_high_edges=('edges', 'mean'), dag=('dag', 'mean'),
    marg_exact=('marginalised', 'mean'), marg_1e3=('marginalised_t3', 'mean'),
    marg_1e2=('marginalised_t2', 'mean'), rowmax_min=('rowmax_min', 'median'),
    secs=('fit_secs', 'mean')).round(4).to_string())
print('\nexact abstraction rate (obs & intv < 1e-8) by fitter x arm:')
ex = d.assign(exact=(d.obs < 1e-8) & (d.intv < 1e-8))
print(ex.pivot_table(index='arm', columns='fitter', values='exact').reindex(ORDER).round(3).to_string())

header('1. THE MARGINALISATION METRIC  (report scores info_eff on EXACT zeros)')
print('marginalised nodes at a ladder of tolerances on max_j |T[i,j]| (columns unit-norm):')
print(d.pivot_table(index='fitter', values=['marginalised', 'marginalised_t6',
                                            'marginalised_t3', 'marginalised_t2']).round(2).to_string())
print('\nmean info_eff under each definition, by fitter x arm:')
for ie in ('info_eff', 'info_eff_t3', 'info_eff_t2'):
    print(f'\n  {ie}:')
    print(d.pivot_table(index='arm', columns='fitter', values=ie).reindex(ORDER).round(3).to_string())

header('2. lambda_graph: does adding graph_loss to (one_child + info) help?')
print('rerun_fixed.csv (fit_abstraction) put g1 marginally ahead of the recommended g0.\n')
for f in ('reg', 'alt'):
    print(f'--- fitter = {f} ---')
    s = d[d.fitter == f]
    print(s.groupby('arm').agg(
        n=('valid', 'size'), allin=('allin', 'mean'), one_child=('one_child', 'mean'),
        valid=('valid', 'mean'), k_used=('k_used', 'mean'), info=('info', 'mean'),
        info_eff=('info_eff', 'mean'), info_eff_t2=('info_eff_t2', 'mean'),
        rel_err_med=('rel_err', 'median'), marg=('marginalised_t2', 'mean'),
        secs=('secs', 'mean')).reindex(ORDER).round(4).to_string())
    print()

header('3. THE ALPHA-HYBRID: which loss arm wins, per fitter?')
for ie in ('info_eff', 'info_eff_t2'):
    print(f'\n### retention metric = {ie}   (lower is better; OCC is the fallback)')
    for f in ('reg', 'alt'):
        rows = []
        for arm in [a for a in ORDER if a != 'occ']:
            r = {'arm': arm}
            for al in ALPHAS:
                acc, hy, base = hybrid(d, arm, al, f, ie=ie)
                r[f'acc@{al}'] = round(acc, 2)
                r[f'hyb@{al}'] = round(hy, 3)
            rows.append(r)
        t = pd.DataFrame(rows).set_index('arm')
        _, _, base = hybrid(d, 'oc_info', 0.05, f, ie=ie)
        print(f'\n  fitter = {f}   (OCC alone = {base:.3f})')
        print(t.to_string())
        for al in ALPHAS:
            c = f'hyb@{al}'
            best = t[c].idxmin()
            print(f'    alpha={al:<6} best = {best:<13} {t.loc[best, c]:.3f}  '
                  f'({100*(t.loc[best,c]-base)/base:+.0f}% vs OCC)   '
                  f'[oc_info = {t.loc["oc_info", c]:.3f}]')

header('4. UTILITY LOSS: info (intra_mass) vs recon, paired per instance')
print('slack_nograph_robust (fit_abstraction): info beats recon at EVERY alpha.')
print('marginalisation/init_k_sweep (also fit_abstraction): recon was the better objective.\n')
for ie in ('info_eff', 'info_eff_t2'):
    for f in ('reg', 'alt'):
        a = d[(d.arm == 'oc_info') & (d.fitter == f)].set_index(KEY)
        b = d[(d.arm == 'oc_recon') & (d.fitter == f)].set_index(KEY)
        j = a.join(b, rsuffix='_r', how='inner')
        wins = int((j[ie] < j[f'{ie}_r']).sum()); ties = int((j[ie] == j[f'{ie}_r']).sum())
        print(f'  {ie:12s} {f}:  info {j[ie].mean():.3f} vs recon {j[f"{ie}_r"].mean():.3f}   '
              f'info better on {wins}/{len(j)} (ties {ties})   '
              f'rel_err med {j.rel_err.median():.4f} vs {j.rel_err_r.median():.4f}')
    print()
print('hybrid at alpha=0.05, both metrics:')
for ie in ('info_eff', 'info_eff_t2'):
    for f in ('reg', 'alt'):
        ai, hi, bs = hybrid(d, 'oc_info', 0.05, f, ie=ie)
        ar, hr, _ = hybrid(d, 'oc_recon', 0.05, f, ie=ie)
        print(f'  {ie:12s} {f}: info hyb {hi:.3f} (acc {ai:.2f})   recon hyb {hr:.3f} '
              f'(acc {ar:.2f})   OCC {bs:.3f}   winner = {"info" if hi < hr else "recon"}')

header('5. lambda_info (utility magnitude): is 10 still optimal?')
for f in ('reg', 'alt'):
    print(f'\n  fitter = {f}')
    for arm in ('oc_info_u1', 'oc_info', 'oc_info_u100'):
        s = d[(d.arm == arm) & (d.fitter == f)]
        _, h, b = hybrid(d, arm, 0.05, f)
        _, h2, b2 = hybrid(d, arm, 0.05, f, ie='info_eff_t2')
        print(f'    {arm:14s} lambda_info={"1" if arm.endswith("u1") else "100" if arm.endswith("u100") else "10":>3}  '
              f'allin {s.allin.mean():.2f} one_child {s.one_child.mean():.2f} '
              f'rel_err_med {s.rel_err.median():.4f}  hyb@.05 {h:.3f}  hyb_t2@.05 {h2:.3f}')

header('6. IS lambda_one_child STILL LOAD-BEARING?')
for f in ('reg', 'alt'):
    s = d[d.fitter == f]
    print(f'\n  fitter = {f}')
    print(s[s.arm.isin(['graph_only', 'info_only', 'oc_info', 'oc_info_g10'])].groupby('arm').agg(
        allin=('allin', 'mean'), one_child=('one_child', 'mean'), valid=('valid', 'mean'),
        exact=('obs', lambda x: float((x < 1e-8).mean())),
        info=('info', 'mean'), info_eff=('info_eff', 'mean'),
        info_eff_t2=('info_eff_t2', 'mean'), marg_t2=('marginalised_t2', 'mean'),
        rel_err_med=('rel_err', 'median')).round(4).to_string())

header('7. ROBUSTNESS: the conservative (worst-weighting) acceptance test')
for ie in ('info_eff', 'info_eff_t2'):
    print(f'\n  {ie}, alpha=0.05, rel_err vs rel_err_worst:')
    for f in ('reg', 'alt'):
        for arm in ('graph_only', 'oc_info', 'oc_info_g1', 'oc_recon'):
            a1, h1, b = hybrid(d, arm, 0.05, f, ie=ie)
            a2, h2, _ = hybrid(d, arm, 0.05, f, ie=ie, err='rel_err_worst')
            print(f'    {f} {arm:13s} generator: acc {a1:.2f} hyb {h1:.3f} | '
                  f'worst-of-3: acc {a2:.2f} hyb {h2:.3f}   (OCC {b:.3f})')

header('8. BY DENSITY  (alpha=0.05, info_eff_t2)')
for f in ('reg', 'alt'):
    print(f'\n  fitter = {f}')
    rows = []
    for dens in sorted(d.density.unique()):
        sub = d[d.density == dens]
        r = {'density': dens}
        for arm in ('graph_only', 'oc_info', 'oc_info_g1', 'oc_recon'):
            acc, hy, base = hybrid(sub, arm, 0.05, f, ie='info_eff_t2')
            r['OCC'] = round(base, 3); r[arm] = round(hy, 3); r[f'acc_{arm}'] = round(acc, 2)
        rows.append(r)
    print(pd.DataFrame(rows).set_index('density').to_string())
