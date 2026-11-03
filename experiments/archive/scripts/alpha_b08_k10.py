"""Frontier at blockiness 0.8, density 0.5, p=100, ground-truth k=10, budget k=10.

Reconstruction error against the tolerance alpha, with AOC Closure as the fallback for EVERY
method under one acceptance rule:

    accept the method's abstraction at tolerance alpha  iff  RAE <= alpha
                                                        AND  ReconErr < ReconErr(AOC Closure)
    otherwise report AOC Closure.

So each curve is what you would actually ship if you insisted on RAE <= alpha, and the AOC
Closure line is the floor every curve returns to when its method is rejected.

Differences from scripts/alpha_recon_K.py, which this is otherwise modelled on:

  * BOTH metrics are the ratio-of-norms form now used in the paper -- ReconErr is
    ||A - T A_high T^+||_F / ||A||_F (not squared) and RAE comes from src.utils.losses.RAE
    (not the stale squared copy in scripts/frontier_fig.py).
  * ONE fitter: `fit_abstraction_auto`, the recommended production path (propagation on an
    acyclic quotient, K_faithful on a cyclic one), rather than scoring every partition twice.
  * budget k = 10 = the generating k for EVERY arm, OC Closure included (coarsened there by
    budget_coarsening, like AOC Closure). It stays a distinct method: the draw is chosen on
    information loss and the All-In repair is skipped.
  * UCA added as a baseline (argmax of its continuous T).

blockiness 0.8 is chosen so reconstruction still has range: A is close to, but not exactly,
T A_high T^+, so the ground-truth partition does not trivially score 0.

CYCLIC: pass --n-back N. The cycles go into A_high and the pair is concretised from there
(scripts/cyclic_gen.py), so (A, T_gt, S_gt, A_high_gt) remains a valid abstraction. The
companion cyclic run uses --n-back 2.

GradCAP is OFF by default (--gradcap restores it): it is scrapped, EXCLUDE_METHODS already
keeps it out of the figure, and it was 63% of the runtime.

FITTER, via --fitter:
    K       fit_abstraction_K(faithful=False) -- the fitter tk_sweep reports. USE THIS for
            anything meant to sit beside that study.
    auto    fit_abstraction_auto at its default faithful=True. What the pre-2026-09-25 csvs
            used, and both faithful paths were rewritten that day, so those need rerunning.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import fit_abstraction_auto, fit_abstraction_K, get_S, is_dag
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import RAE, reconstruction_loss
from src.models.hail_mary import HailMary
from src.models.wrapper import uca
from occ import occ, oc_closure, budget_coarsening
from cyclic_gen import sample_cyclic_pair
from partition_baselines import allin_agglom

P, K_GT, BLK, HET = 100, 10, 0.8, 0.0
RHO = 0.8       # spectral radius the cyclic draw is rescaled to, so (I-A)^-1 exists
DENSITIES = [0.1, 0.3, 0.5]
EPOCHS, LAMBDAS = 3000, [0., 100., 1000.]
NR, MAXITER = 5, 100
B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
ALPHAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])


def job(arg):
    dens, seed, n_back, skip_gradcap, fitter, blk = arg
    torch.set_num_threads(1)
    try:
        if n_back:
            # CYCLIC: the cycles go in A_high and the pair is CONCRETISED from there, so
            # (A, T_gt, S_gt, A_high_gt) stays a valid abstraction (see scripts/cyclic_gen.py).
            # Reversing edges in A_low instead -- what scripts/cyclic.py does -- leaves the
            # image of T_concretisation_sampling and destroys the ground truth.
            A, T_gt, S_gt, A_high_gt = sample_cyclic_pair(
                n_high=K_GT, n_low=P, density=dens, n_back=n_back, blockiness=blk,
                heterogeneity=HET, internal_density=0., seed=seed, max_attempts=400)
        else:
            A, T_gt, S_gt, A_high_gt = sample_low_level_graph(
                n_high=K_GT, n_low=P, density=dens, shuffle=True, blockiness=blk,
                heterogeneity=HET, internal_density=0., graph_type='random', seed=seed,
                connected=True, max_attempts=400)
    except Exception as e:
        return [dict(seed=seed, density=dens, n_back=n_back, method='GEN_FAILED',
                     blk=blk, err=type(e).__name__)]

    E = np.linalg.inv(np.eye(P) - A)
    com = dict(seed=seed, density=dens, blk=blk, het=HET, p=P, k_gt=K_GT, n_back=n_back)
    out = []

    def score(nm, Th, secs, extra=None):
        kq = Th.shape[1]
        r = dict(method=nm, k=kq, train_secs=secs, **com, **(extra or {}))
        try:
            Aq = (Th.T @ np.abs(A) @ Th) * (1 - np.eye(kq))
            r['qcyc'] = int(not is_dag(Aq))
        except Exception:
            r['qcyc'] = -1
        t0 = time.time()
        try:
            if fitter == 'eigen':
                Ah, Tp = fit_abstraction_eigen(A_low=A, T_hard=Th, seed=seed,
                                               n_restarts=NR)[:2]
            elif fitter == 'K':
                Ah, Tp = fit_abstraction_K(A_low=A, E_low=E, T_hard=Th, n_restarts=NR,
                                           maxiter=MAXITER, seed=seed, faithful=False)
            else:
                Ah, Tp = fit_abstraction_auto(A_low=A, T_hard=Th, E_low=E, n_restarts=NR,
                                              maxiter=MAXITER, seed=seed)[:2]
            S = get_S(T=Tp, T_hard=Th, A_low=A)
            r['recon'] = float(reconstruction_loss(A_low=A, T=Tp, A_high=Ah))
            r['rae'], r['marg_skip'] = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E,
                                           T_hard=Th, return_skipped=True)
            r['norm_Ahigh'] = float(np.linalg.norm(Ah))
            r['marg'] = int((np.abs(Tp).max(1) <= 1e-2).sum())
        except Exception as e:
            r['recon'] = r['rae'] = np.nan
            r['err'] = type(e).__name__
        r['fit_secs'] = time.time() - t0
        return [r]

    # TWO ground-truth references, and they are not the same thing:
    #
    #   'ground truth'  the generating (T, A_high) pair scored analytically. No fit, so it is
    #                   free, and it is the floor -- what the metrics read when T is exactly
    #                   right.
    #   'GT partition'  the generating PARTITION handed to the fitter like any other arm.
    #                   This is the one that isolates the FITTER: the partition is perfect,
    #                   so whatever it loses is the fitter's doing. Re-enabled 2026-09-25 on
    #                   request. Caveat it was dropped for: the generating partition has
    #                   multi-child clusters, so on an acyclic quotient its RAE reports the
    #                   propagation fitter's feasibility bound (ptol=1e-9, i.e. ~3e-5) rather
    #                   than anything about the partition -- read its ReconErr, not its RAE.
    Th_gt = (T_gt != 0).astype(float)
    out += score('GT partition', Th_gt, 0.)
    if True:
        out.append(dict(method='ground truth', k=K_GT, train_secs=0., fit_secs=0., qcyc=np.nan,
                        recon=float(reconstruction_loss(A_low=A, T=T_gt, A_high=A_high_gt)),
                        rae=float(RAE(A_low=A, T=T_gt, A_high=A_high_gt, S=S_gt, E_low=E,
                                      T_hard=Th_gt)),
                        norm_Ahigh=float(np.linalg.norm(A_high_gt)), marg=0, **com))
    # AOC Closure at the budget -- the fallback every other arm is measured against
    t0 = time.time()
    try:
        g, Th, info = occ(A, seed=seed, k=K_GT)
    except Exception:
        g, Th, info = occ(A, seed=seed)          # cyclic: budget may admit no safe merge
        info['budget_met'] = 0
    out += score('AOC Closure', Th, time.time() - t0,
                 dict(budget_met=int(info.get('budget_met', 0)), k_closure=info.get('k_closure')))

    # OC Closure AT THE SAME BUDGET: the one-child fixed point defines no width of its own,
    # so it is coarsened to k with budget_coarsening exactly as AOC Closure is. It remains a
    # distinct method -- the draw is selected on information loss and the All-In repair is
    # skipped -- so it can be (and often is) All-In-invalid or cyclic at the quotient, which
    # budget_coarsening preserves rather than fixes.
    t0 = time.time()
    gO = oc_closure(A, seed=seed, return_T=False)[0]
    try:
        gO, _ = budget_coarsening(A, gO, K_GT, rng=np.random.default_rng(seed))
    except Exception:
        pass                                   # no admissible merge: report at its own width
    Th = np.zeros((P, max(len(gO), K_GT)))
    for j, mem in enumerate(gO):
        Th[np.asarray(mem), j] = 1.0
    out += score('OC Closure', Th, time.time() - t0)

    t0 = time.time(); Th = allin_agglom(A, K_GT)
    out += score('Greedy All-In', Th, time.time() - t0)

    for lam in ([] if skip_gradcap else LAMBDAS):
        t0 = time.time()
        torch.manual_seed(31337 + seed)
        m = HailMary(n_low=P, n_high=K_GT, A_low=torch.from_numpy(A).float(),
                     init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
                lambda_info=0., lambda_surj=0, verbose=False)
        Th = m.get_partition()[0].detach().numpy()
        out += score('GradCAP', Th, time.time() - t0, dict(lam_oc=lam))

    t0 = time.time()
    try:
        Tu = uca(A_low=A, k=K_GT)
        Th = np.zeros((P, K_GT)); Th[np.arange(P), np.argmax(np.abs(Tu), axis=1)] = 1.
        out += score('UCA', Th, time.time() - t0)
    except Exception as e:
        out.append(dict(method='UCA', recon=np.nan, rae=np.nan, err=type(e).__name__, **com))
    return out


# Methods kept OUT of the figure but still written to the csv -- demoted, not deleted, so
# removing a name here restores its curves.
#   GT partition  it reaches ~0.39-0.49 where every learned arm sits at ~0.72-1.00, so
#                 drawing it stretches the y-axis over a band nothing else occupies and the
#                 methods the figure is about become indistinguishable. Read it from the
#                 per-density table this script prints instead.
#
# GradCAP is NOT listed: it is gated by --gradcap at compute time instead, so a csv that has
# it wanted it. It only separates at low density -- measured at blockiness 0.8, lam_oc=100
# reaches 0.940 against 0.990 for AOC Closure at d=0.1, but 0.997 against 0.995 at d=0.3 --
# so it is worth drawing on a low-density grid and worth skipping on 0.1/0.3/0.5.
EXCLUDE_METHODS = ('GT partition',)


def arm_label(r):
    return r'GradCAP ($\lambda_{oc}=%g$)' % r.lam_oc if r.method == 'GradCAP' else r.method


ALPHA_MIN = 1e-6   # a decade below the smallest RAE any inexact arm achieves (~1e-5)


def alpha_grid(d, n=12):
    """A log grid from ALPHA_MIN to the largest RAE achieved, padded.

    The exact arms sit at ~1e-16, so anchoring the low end on them would spend ten decades on
    a region where every curve is flat by construction: below ~1e-5 the accepted set is just
    "the arms that are exact", and ALPHA_MIN keeps one decade of that visible without letting
    it dominate the axis. The floor is FIXED rather than data-driven so panels and reruns
    share an x-axis. The top end follows the data so no arm is cut off.
    """
    r = d.rae.replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    hi = min(1.0, float(r.max()) * 3) if len(r) else 1.0
    return np.unique(np.concatenate([np.geomspace(ALPHA_MIN, max(hi, 1e-3), n), [1.0]]))


def curves(d, alphas):
    """Per-method ReconErr after the fallback rule, median over seeds, at ONE density."""
    d = d.copy()
    d['arm'] = d.apply(arm_label, axis=1)
    base = d[d.method == 'AOC Closure'].set_index('seed').recon.to_dict()
    # Drop any arm not present on EVERY seed: a csv extended with --seed-start can carry a
    # partial arm from an earlier revision of this script, and a median over a different seed
    # set is not comparable with the rest of the panel.
    n_seed = d.groupby('arm').seed.nunique()
    full = int(n_seed.max()) if len(n_seed) else 0
    dropped = set(d[d.method.isin(EXCLUDE_METHODS)].arm)
    arms = [a for a in d.arm.unique()
            if a not in ('AOC Closure', 'ground truth') and a not in dropped
            and int(n_seed.get(a, 0)) == full]
    out = {}
    for a in arms:
        sub, ys = d[d.arm == a], []
        for al in alphas:
            per = []
            for sd, g in sub.groupby('seed'):
                b = base.get(sd, np.nan)
                ok = g[(g.rae <= al) & (g.recon < b)]
                per.append(float(ok.recon.min()) if len(ok) else b)
            ys.append(np.nanmedian(per))
        out[a] = np.array(ys)
    out['AOC Closure'] = np.full(len(alphas), np.nanmedian(list(base.values())))
    return out


STYLE = {'AOC Closure': ('#eb6834', '-', 'o', 2.0),
         'OC Closure': ('#7a7a7a', '--', 's', 1.4),
         'Greedy All-In': ('#2ca02c', '--', 'D', 1.4),
         r'GradCAP ($\lambda_{oc}=0$)': ('#86b6ef', '-', 'o', 1.4),
         r'GradCAP ($\lambda_{oc}=100$)': ('#2a78d6', '-', 's', 1.4),
         r'GradCAP ($\lambda_{oc}=1000$)': ('#0d366b', '-', '^', 1.4),
         'UCA': ('#9467bd', ':', 'v', 1.4),
         'GT partition': ('#111111', '-.', '*', 1.2)}
ORDER = ['AOC Closure', 'Greedy All-In', 'OC Closure', 'UCA',
         r'GradCAP ($\lambda_{oc}=0$)', r'GradCAP ($\lambda_{oc}=100$)',
         r'GradCAP ($\lambda_{oc}=1000$)']


def plot(path_csv, path_png):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        # TrueType, not Type 3: most venues reject Type 3 in submitted PDFs
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    d = pd.read_csv(path_csv)
    d = d[d.method != 'GEN_FAILED']
    alphas = alpha_grid(d)
    dens = sorted(d.density.unique())
    # Per-panel y, NOT shared: the AOC baseline rises from ~1.0 at d=0.1 to ~1.7 at d=0.5, so
    # a shared axis is set by the worst panel and squashes the 0.5-1.1 band where the arms
    # actually separate. Each panel keeps its own scale; the generating abstraction's line is
    # drawn in every panel so the floor stays readable across them.
    fig, axes = plt.subplots(1, len(dens), figsize=(2.4 * len(dens) + 0.6, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, dn in zip(axes, dens):
        sub = d[d.density == dn]
        cv = curves(sub, alphas)
        for a in [x for x in ORDER if x in cv] + [x for x in cv if x not in ORDER]:
            c, ls, mk, lw = STYLE.get(a, ('#999999', '-', 'o', 1.2))
            ax.plot(alphas, cv[a], ls, color=c, marker=mk, ms=3, lw=lw, label=a)
        # The generating abstraction's ReconErr is NOT drawn: at 0.39-0.44 it sits well below
        # every curve and its line forced ~30% of each panel to be empty. It is printed below
        # instead, for the caption.
        ax.set_xscale('log')
        ax.set_title('density %.1f' % dn, fontsize=9)
        ax.set_xlabel(r'tolerance $\alpha$', fontsize=9)
        ax.grid(alpha=0.22, lw=0.5)
    axes[0].set_ylabel('ReconErr', fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=4, loc='lower center',
               bbox_to_anchor=(0.5, -0.17))
    fig.savefig(path_png, dpi=220)
    pdf = path_png.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf)                      # vector copy for the paper
    print('wrote', path_png, 'and', pdf)
    ref = d[d.method == 'ground truth'].groupby('density').recon.agg(['median', 'min', 'max'])
    print('\ngenerating abstraction ReconErr (for the caption):')
    print(ref.round(3).to_string())
    d['arm'] = d.apply(arm_label, axis=1)
    for dn in dens:
        sub = d[d.density == dn]
        print('\n=== density %.1f ===' % dn)
        print(sub.groupby('arm')[['recon', 'rae', 'k', 'qcyc', 'marg']].median().round(4).to_string())
    print('\nalpha grid: %s' % np.array2string(alphas, precision=2, max_line_width=200))
    print('\nRAE quantiles by arm (all densities): ')
    print(d.groupby('arm').rae.describe(percentiles=[.25, .5, .75])
          [['min', '25%', '50%', '75%', 'max']].to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed-start', type=int, default=0,
                    help='first seed; use it to EXTEND an existing csv instead of recomputing')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--out', default='alpha_b08_k10')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--n-back', type=int, default=0,
                    help='reverse this many edges to make the LOW-LEVEL graph cyclic')
    ap.add_argument('--dens', type=float, nargs='+', default=None,
                    help='override the density grid (default %s). NOTE connected=True is '
                         'checked on A_low, which needs the 10-cluster quotient weakly '
                         'connected, so very low densities mostly fail to generate.'
                         % DENSITIES)
    ap.add_argument('--blk', type=float, default=BLK,
                    help='blockiness (default %g). At 1.0 A is EXACTLY T A_high T^+, so the '
                         'generating partition reconstructs at ~0; %g was chosen so it does '
                         'not, but that also compresses every learned arm into the top 1%%%%.'
                         % (BLK, BLK))
    ap.add_argument('--fitter', default='auto', choices=['auto', 'K', 'eigen'],
                    help="'K' is fit_abstraction_K(faithful=False), the fitter tk_sweep "
                         "reports. 'auto' is fit_abstraction_auto at its default "
                         'faithful=True, which is what the pre-2026-09-25 csvs used.')
    ap.add_argument('--gradcap', action='store_true',
                    help='also train the GradCAP arms. OFF by default: GradCAP is scrapped '
                         'and already in EXCLUDE_METHODS, so it is 63%% of the runtime '
                         '(3852 of 6900 CPU-s measured) spent on curves nothing draws.')
    args = ap.parse_args()
    csv = '%s/data/%s.csv' % (B, args.out)
    png = '%s/figures/%s.png' % (B, args.out)
    if not args.plot_only:
        t0 = time.time(); rows = []
        if args.seed_start and os.path.exists(csv):
            rows = pd.read_csv(csv).to_dict('records')
            print('extending %s (%d existing rows)' % (csv, len(rows)), flush=True)
        grid = args.dens if args.dens else DENSITIES
        jobs = [(d, i, args.n_back, not args.gradcap, args.fitter, args.blk)
                for i in range(args.seed_start, args.seed_start + args.n) for d in grid]
        with mp.Pool(args.procs) as pool:
            for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
                rows.extend(rs); pd.DataFrame(rows).to_csv(csv, index=False)
                el = (time.time() - t0) / 60
                print('%d/%d jobs  %.1fmin (eta %.0fmin)'
                      % (i, len(jobs), el, el / i * (len(jobs) - i)), flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False)
        print('done %.1f min -> %s' % ((time.time() - t0) / 60, csv), flush=True)
    plot(csv, png)
