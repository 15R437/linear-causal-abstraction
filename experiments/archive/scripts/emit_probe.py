import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, pandas as pd, multiprocessing as mp, torch

def job(a):
    torch.set_num_threads(1)
    kind,p,k,part,trial,fit=a
    A_low,T_gt,_=gen_instance(kind,p,k,0.6,trial)
    E=np.linalg.inv(np.eye(p)-A_low)
    Th=(T_gt!=0).astype(float) if part=='gt' else random_partition(p,k,np.random.default_rng(10_000+trial))
    fn={'K':fit_abstraction_K,'alt':fit_abstraction_alternative}[fit]
    Ah,T=fn(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial)
    blk=block_structure(A_low,Th); idx=cluster_indices(Th); lab=np.argmax(Th,axis=1)
    colmax=np.abs(T).max(axis=0); colmax[colmax==0]=1.
    rel=np.abs(T)/colmax[None,:]
    emit=np.array([any(lab[c]!=lab[r] for c in np.where(A_low[:,r]!=0)[0]) for r in range(p)])
    marg=np.array([rel[r,lab[r]]<1e-6 for r in range(p)])
    absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.)
    mx=absA.max() if absA.max()>0 else 1.
    pred=absA>1e-6*mx
    # a cluster is a "live parent" if its COLUMN of A_high has any edge
    live=pred.any(axis=0)
    blkcol=blk.any(axis=0)                      # clusters that really do emit at low level
    return dict(kind=kind,p=p,k=k,part=part,trial=trial,fit=fit,
        n_emit=int(emit.sum()), n_nonemit=int((~emit).sum()),
        marg_given_emit=float(marg[emit].mean()) if emit.any() else np.nan,
        marg_given_nonemit=float(marg[~emit].mean()) if (~emit).any() else np.nan,
        live_parents=int(live.sum()), true_parents=int(blkcol.sum()), k_=k,
        edges=int(pred.sum()), block_edges=int(blk.sum()))

if __name__=='__main__':
    jobs=[(kind,p,k,part,t,f) for kind in ('acyclic','cyclic') for (p,k) in ((24,4),(40,6))
          for part in ('gt','random') for t in range(15) for f in ('K','alt')]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',250)
    d['frac_parents_alive']=d.live_parents/d.true_parents
    d['frac_edges']=d.edges/d.block_edges
    print('P(node marginalised | emits cross-cluster)  vs  P(marginalised | does not emit)')
    print(d.groupby(['fit','kind','p','part'])[['marg_given_emit','marg_given_nonemit',
          'live_parents','true_parents','frac_parents_alive','edges','block_edges','frac_edges']].mean().round(3).to_string())
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/emit_probe.csv',index=False)
