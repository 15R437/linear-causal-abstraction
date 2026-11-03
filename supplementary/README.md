# Linear causal abstraction — supplementary code

Reference implementations of the partition learners and abstraction fitters, the data
generator, and the Leontief input–output data used in the paper.

## Layout

    abstraction/
      closure.py              AOC Closure and OC Closure
      partition_baselines.py  Greedy All-In, plus structural-equivalence baselines
      eigen_solver.py         closed-form fitter (sinks optimised, cyclic SCCs pinned)
      fitters.py              propagation and rank-one fitters, and the router
      generate_data.py        synthetic models with a known ground-truth abstraction
      losses.py               RAE, reconstruction error, structural diagnostics
    data/
      leontief_A_low.npy      172 x 172 input coefficient matrix (EXIOBASE3, GB)
      leontief_labels.pkl     {'sectors': [...163], 'factors': [...9]}
    figures/                  the figures and tables from the paper
    example.py                runnable end-to-end demonstration

## Quick start

    pip install -r requirements.txt
    python example.py            # synthetic, acyclic and cyclic quotients
    python example.py leontief   # the Leontief economy

## The two ingredients

**Partition learning.** A partition is sufficient for a linear abstraction to exist when it
satisfies *All-In* (for every high-level edge X→Y, every member of X has an edge into Y) and
*one-child* (every cluster has at most one child cluster). `closure.occ` reaches both by
taking the one-child merge closure and repairing All-In violations;
`partition_baselines.allin_agglom` (Greedy All-In) instead merges greedily on the All-In
violation count, breaking ties on added intra-cluster edge mass. Acyclicity of the quotient
is *not* required by either condition and is not enforced.

**Fitting T.** Each quotient edge i→j imposes the collinearity constraint

    M_ij t_j = m_ij t_i,    M_ij := A[pi_i, pi_j] (I - A[pi_j, pi_j])^{-1}

On an acyclic quotient these determine every cluster from its children, leaving only the
sinks free. Around a cycle they close on themselves, so composing them gives

    Pi t = (prod m) t,      Pi := M_{i1 i2} M_{i2 i3} ... M_{ik i1}

and a cluster on a cycle is an eigenvector of the cycle product, with eigenvalue the product
of the high-level weights traversed. `eigen_solver.fit_abstraction_eigen` uses this:
it condenses the quotient into strongly connected components, solves cyclic components in
closed form, propagates the rest, and optimises only the sink directions.

## Metrics

`RAE` is the relative abstraction error, the worst case over the observational and the k
interventional regimes of a noise-weighted relative error; it is zero exactly when the
abstraction is exact. `reconstruction_loss` is ||A - T A_high T^+||_F / ||A||_F. The two are
not redundant: on a correct partition RAE reaches machine precision for many different T,
because the abstraction condition does not pin the sink directions, while reconstruction
error does distinguish them.

## Notes

* `losses.kl_loss` depended on a third-party package and is stubbed out; no result in the
  paper uses it.
* The Leontief matrix is non-negative across all 163 sector-to-sector columns. The only
  negative entries (0.09% of the matrix) lie in the two value-added factors *Other net taxes
  on production* and *Operating surplus*, which are net of subsidies and therefore signed by
  construction.
