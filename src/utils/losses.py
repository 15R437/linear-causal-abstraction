import numpy as np
import scipy.linalg as la
"""
loss_fn should take as input:
    A_high, A_low, A_hat, T, X
"""
def reconstruction_loss(A_low, T, A_high, A_hat=None, **kwds):
    baseline = np.linalg.norm(A_low, ord='fro')
    if A_hat is None:
        R = np.linalg.pinv(T.T)
        A_hat = T @ A_high @ R.T

    return np.linalg.norm((A_low - A_hat), ord='fro') / baseline

def reconstruction_loss_with_data(A_low, T, A_high, X, A_hat=None, **kwds):
    """
    Params
        A_hat (np.ndarray): (n_low x n_low) adjacency matrix assumed to be upper-triangular under topological ordering
        X (np.ndarray): (n_samples x n_low) data matrix
        A_low (np.ndarray): (n_low x n_low) true adjacency matrix
    """
    if A_hat is None:
            R = np.linalg.pinv(T.T)
            A_hat = T @ A_high @ R.T
    
    I = np.eye(A_hat.shape[0])
    residual = (I-A_hat.T)@X.T #pxN matrix
    residual_oracle = (I-A_low.T)@X.T
    rmse = np.sqrt(np.mean(residual**2,axis=1))
    rmse_oracle = np.sqrt(np.mean(residual_oracle**2,axis=1))
    rmse_ratio = np.mean(rmse / (rmse_oracle + 1e-8))
    return rmse_ratio

def kl_loss(E_high, A_low, T, **kwds):
    """Consistency loss of Unsupervised Causal Abstraction, used only as a baseline.

    Requires `unsupervised_causal_reduction`, which was provided privately by that
    paper's authors and is not redistributed here. No result in the paper depends
    on this function.
    """
    raise NotImplementedError(
        "kl_loss requires 'unsupervised_causal_reduction', which was provided "
        "privately by the authors of Unsupervised Causal Abstraction and is not "
        "redistributed here. No result in the paper depends on it.")

def observation_loss(E_low,E_high,S,T,**kwds):
    lhs = S @ E_high #(I_low - A_low) @ S
    rhs = E_low @ T #T @ (I_high - A_high)
    error = lhs - rhs
    obs_loss = la.norm(error, 2) #np.mean((lhs - rhs)**2)
    return obs_loss

def intervention_loss(E_low, E_high, S, T, T_hard=None, **kwds):
    if T_hard is None:
        T_hard = (T != 0).astype(float)
    n_low, n_high = T.shape
    I_low = np.eye(n_low)
    I_high = np.eye(n_high)

    F_ii = np.zeros((n_high, n_high, n_high))
    idx = np.arange(n_high)
    F_ii[idx, idx, idx] = 1.0
    F_pi = (T_hard.T)[..., None] * I_low

    # E_high_do needs the same (I + F(E-I))^{-1} correction as E_low_do below -- without it,
    # this only computes the correct post-intervention total-effect matrix when the intervened
    # high-level variable has no cycle back into itself (the correction reduces to the identity
    # there, which is why this was invisible on every acyclic A_high). With a genuine cycle
    # through variable i, omitting it gives the wrong E_high_do and a spurious nonzero loss even
    # for an exact abstraction.
    inv_term_high = np.linalg.inv(I_high + F_ii @ (E_high - I_high))
    E_high_do = E_high[None, ...] - (E_high - I_high)[None, ...] @ inv_term_high @ F_ii @ E_high[None, ...]

    inv_term_low = np.linalg.inv(I_low + F_pi @ (E_low - I_low))
    E_low_do = E_low - (E_low - I_low) @ inv_term_low @ F_pi @ E_low

    loss = 0.
    for i in range(n_high):
        LHS = E_low_do[i] @ T          # (n_low, n_high)
        RHS = (S + (T - S) @ F_ii[i]) @ E_high_do[i]  # (n_low, n_high)
        loss += np.mean((LHS - RHS) ** 2)

    return loss / n_high

def RAE(A_low, T, A_high, S, E_low=None, T_hard=None, sigma_u=None, tau=1e-2,
        return_skipped=False, **kwds):
    """Relative Abstraction Error: the single number the frontier is scored on.

    max over the observational regime and the k interventional ones of the Sigma_u-weighted
    relative error

        ||Sigma^1/2 D||_F / ||Sigma^1/2 R||_F  =  sqrt( tr(D' Sigma D) / tr(R' Sigma R) ),

    where D is the abstraction defect and R the low-level signal it is measured against.
    Unifies `observation_loss` and `intervention_loss` on a scale that is comparable across
    instances.

    The norm is NOT squared (changed 2026-09-24), so the metric is homogeneous of degree one
    in D: doubling the defect doubles the score, and the value sits on the same scale as
    `reconstruction_loss`, which is likewise a ratio of norms rather than of squares. Values
    are therefore the SQUARE ROOT of those produced before that date -- an exact fit that
    logged 1e-12 now logs 1e-6 -- so thresholds calibrated on the old scale (rae < 1e-6 for
    "exact", the alpha sweeps) and any cached CSVs need squaring to compare. The max is over
    the same regime either way, since sqrt is monotone.

    Note this is a ratio of ROOT-MEAN-SQUARE norms, sqrt(E||u'D||^2) / sqrt(E||u'R||^2), not
    E||u'D|| / E||u'R||: the latter has no closed form, since u'D is a correlated Gaussian
    vector and the expected norm of one is not elementary.

    Clusters whose T-weight has been driven below `tau` are SKIPPED: `fit_abstraction`
    marginalises variables rather than fitting them, and do(Y_i) is not well defined for a
    column carrying no weight -- the denominator vanishes and the ratio is meaningless.
    Charge those variables through `info_effective`, not here. Pass `return_skipped=True`
    to get (rae, n_skipped) and see how many regimes were dropped.

    `sigma_u` is a vector of exogenous VARIANCES and defaults to ones (standard normal noise),
    which is what makes the expectation closed form. Note the weighting is not innocuous on
    INEXACT partitions: it moves the value by up to 30x on the squared scale (~5.5x here) and
    can flip an accept/reject decision. Exact ones score 0 under every weighting.
    """
    p, k = T.shape
    if E_low is None:
        E_low = np.linalg.inv(np.eye(p) - A_low)
    if T_hard is None:
        T_hard = (T != 0).astype(float)
    if sigma_u is None:
        sigma_u = np.ones(p)
    I_low, I_high = np.eye(p), np.eye(k)
    E_high = np.linalg.inv(I_high - A_high)

    def ratio(D, R):
        num = float(np.sum(sigma_u[:, None] * D ** 2))
        den = float(np.sum(sigma_u[:, None] * R ** 2))
        return np.inf if den <= 1e-30 else float(np.sqrt(num / den))

    per = [ratio(S @ E_high - E_low @ T, E_low @ T)]

    F_ii = np.zeros((k, k, k))
    idx = np.arange(k)
    F_ii[idx, idx, idx] = 1.0
    F_pi = (T_hard.T)[..., None] * I_low
    inv_hi = np.linalg.inv(I_high + F_ii @ (E_high - I_high))
    E_hi_do = E_high[None] - (E_high - I_high)[None] @ inv_hi @ F_ii @ E_high[None]
    inv_lo = np.linalg.inv(I_low + F_pi @ (E_low - I_low))
    E_lo_do = E_low - (E_low - I_low) @ inv_lo @ F_pi @ E_low

    colnorm = np.linalg.norm(T, axis=0)
    n_skip = 0
    for i in range(k):
        if colnorm[i] <= tau:
            n_skip += 1
            continue
        R = E_lo_do[i] @ T
        D = R - (S + (T - S) @ F_ii[i]) @ E_hi_do[i]
        per.append(ratio(D, R))

    rae = float(np.nanmax(per))
    return (rae, n_skip) if return_skipped else rae


def info_loss(A_low, T_hard, **kwds):
    """Fraction of |A_low| edge mass falling INSIDE clusters, so represented by no high-level
    edge. Read it only beside validity -- an invalid partition scores well precisely because
    it never pays the coarsening price validity demands.
    """
    absA = np.abs(A_low)
    co = T_hard @ T_hard.T
    np.fill_diagonal(co, 0)
    return float((absA * co).sum() / max(absA.sum(), 1e-12))


def info_effective(A_low, T_hard, T, tau=1e-2, **kwds):
    """`info_loss` plus the mass incident to variables the fit MARGINALISED (|T| <= tau).

    Charges both kinds of destruction, so a partition cannot win by having `fit_abstraction`
    silently drop the nodes it could not fit. The tolerance is 1e-2 rather than exact zero
    because `fit_abstraction_alternative` never returns a hard zero, and testing at zero
    understates the loss by 0.07-0.29 -- enough to reverse which method looks better.
    """
    absA = np.abs(A_low)
    co = T_hard @ T_hard.T
    np.fill_diagonal(co, 0)
    lost = co.astype(bool)
    marg = np.abs(T).max(axis=1) <= tau
    lost[marg, :] = True
    lost[:, marg] = True
    np.fill_diagonal(lost, False)
    return float((absA * lost).sum() / max(absA.sum(), 1e-12))


def abstraction_residual(A_low, T, A_high, **kwds):
        n_low,n_high = T.shape
        I = np.eye(n_low)

        S_list = []
        for j in range(n_high):
            t_j = T[:, j]
            A_jj = t_j[:, None] * A_low * t_j[None, :]
            s_j = np.linalg.solve(I - A_jj, t_j)
            S_list.append(s_j)
        S = np.stack(S_list, axis=1)  # (n_low, n_high)

        influence_matrix = A_low @ S # (n_low, n_high)

        T_sq = T ** 2                                          # (n_low, n_high)
        t_i_norms = np.clip(T_sq.sum(axis=0), a_min=1e-8, a_max=None)                # (n_high,) effective block sizes

        # Weighted mean: a_star[i,j] = sum_r w[r,i] * influence[r,j] / sum_r w[r,i]
        #a_star = (T_sq.T @ influence_matrix) / t_i_norms[:,None]  # (n_high, n_high)
        diff = influence_matrix[None, :, :] - A_high[:, None, :]* T.T[:, :, None]

        weights = T_sq.T[:, :, None]                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(axis=1) / t_i_norms[:,None]  # (n_high, n_high)
        diag_mask = 1.0 - np.eye(n_high)
        return (residuals * diag_mask).sum()

def graph_loss(A_low, T_hard, **kwds):
    n_low = A_low.shape[0]
    A_low_binary = (A_low != 0).astype(float)
    #T_hard = (np.abs(T_hard) > 1e-6).astype(float)
    
    # H[i,j] = number of edges between low-level node i and high-level cluster j excluding intra-cluster edges
    H = A_low_binary @ T_hard * (1 - T_hard)
    
    # 2. Replicate the forward pass of: 1 - relu(1 - H)
    # This clamps the upper bound of H to 1.0 without affecting values below it
    C = 1.0 - np.maximum(0, 1.0 - H) 
    
    # 3. Compute number of shared cluster connections between low-level nodes
    CC = C @ C.T 
    
    # 4. Pairwise squared Euclidean distances: ||C[i] - C[j]||^2 using broadcasting
    CC_norm = (C ** 2).sum(axis=1)
    C_diff_sq = CC_norm[:, None] + CC_norm[None, :] - 2 * CC  
    
    co_assign = T_hard @ T_hard.T                                   
    diag_mask = 1.0 - np.eye(n_low)
    
    pairwise_violations = co_assign * diag_mask * C_diff_sq
    
    # 7. Compute total violations per node across all other co-assigned nodes
    node_violation_scores = pairwise_violations.sum(axis=1)
    
    # 8. Count nodes with non-zero violations
    violating_nodes_count = np.count_nonzero(node_violation_scores > 0)
    
    return violating_nodes_count


def topo_loss(A_low, T, A_high, **kwargs):
    n_high = T.shape[1]
    
    # 1. Binarize low-level graph connections
    A_low_binary = (A_low != 0).astype(float)
    dag_mask = (A_high != 0).astype(float)
    
    # M[i,j] represents the aggregate structural connection from cluster i to cluster j
    M = T.T @ A_low_binary @ T 
        
    violation_mask = 1.0 - dag_mask - np.eye(n_high)
    
    topo_loss = np.mean(violation_mask * (M ** 2))
    
    return topo_loss

def abstraction_loss(E_low,A_low,A_high,S,T,**kwds):
    obs_loss = observation_loss(A_low=A_low, A_high=A_high, S=S, T=T)
    intv_loss = intervention_loss(E_low=E_low, S=S, T=T)
    return obs_loss + intv_loss

def rand_index(T_gt,T_pred,**kwds):
    p,k = T_gt.shape
    a=b=0
    for i in range(p-1):
        for j in range(i+1,p):
            SAME_CLST_gt = set(np.where(T_gt[i,:]!=0)[0]).intersection(set(np.where(T_gt[j,:]!= 0)[0]))
            SAME_CLST_pred = set(np.where(T_pred[i,:]!=0)[0]).intersection(set(np.where(T_pred[j,:]!= 0)[0]))
            if SAME_CLST_gt and SAME_CLST_pred:
                a += 1
            elif not SAME_CLST_gt and not SAME_CLST_pred:
                b += 1

    return 2*(a+b)/(p*(p-1))

