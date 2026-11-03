import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, pandas as pd, multiprocessing as mp, torch

def job(a):
    torch.set_num_threads(1)
    p,k,trial=a
    A_low,T_gt,_=gen_instance('acyclic',p,k,0.6,trial)
    E=np.linalg.inv(np.eye(p)-A_low)
    Th=random_partition(p,k,np.random.default_rng(10_000+trial))
    blk=block_structure(A_low,Th); idx=cluster_indices(Th); lab=np.argmax(Th,axis=1)
    Ah,T=fit_abstraction_K(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial)
    cm=np.abs(T).max(axis=0); cm[cm==0]=1.
    rel=np.abs(T)/cm[None,:]
    alive=np.array([rel[r,lab[r]]>=1e-6 for r in range(p)])
    absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.)
    mx=max(absA.max(),1e-300); pred=absA>1e-6*mx
    dead=blk&~pred                                   # real block, no high-level edge

    # for each dead (i,j): did cluster j lose ALL nodes that emit specifically into pi_i?
    struct=0; cancel=0
    for i in range(k):
        for j in range(k):
            if not dead[i,j]: continue
            # nodes of cluster j with a child in cluster i
            emit_ij=[r for r in idx[j] if np.any(A_low[np.ix_(idx[i],[r])])]
            if all(not alive[r] for r in emit_ij): struct+=1   # support avoids them entirely
            else: cancel+=1                                    # live emitters, yet edge vanished
    # column-level: is the loss all-or-nothing per parent cluster?
    livecol=pred.any(axis=0); blkcol=blk.any(axis=0)
    dead_in_livecol=int((dead&livecol[None,:]).sum())
    return dict(p=p,trial=trial,n_dead=int(dead.sum()),struct=struct,cancel=cancel,
                dead_cols=int((blkcol&~livecol).sum()),true_cols=int(blkcol.sum()),
                dead_edges_in_live_cols=dead_in_livecol,
                edges_in_dead_cols=int((blk&(~livecol)[None,:]).sum()))

if __name__=='__main__':
    jobs=[(p,k,t) for (p,k) in ((24,4),(40,6)) for t in range(15)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',220)
    g=d.groupby('p')[['n_dead','struct','cancel','dead_cols','true_cols',
                      'edges_in_dead_cols','dead_edges_in_live_cols']].sum()
    g['frac_structural']=(g.struct/g.n_dead).round(3)
    g['frac_dead_edges_explained_by_dead_cols']=(g.edges_in_dead_cols/g.n_dead).round(3)
    print(g.to_string())
