"""Instrument fit_abstraction_K's restart pool: what does the tie-break actually choose from?"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch, pandas as pd, multiprocessing as mp
from scipy.optimize import minimize
from src.utils.helper import cluster_indices

def pool_K(A_low,E_low,T_hard,n_restarts=10,seed=None,maxiter=100):
    """Verbatim fit_abstraction_K, but returns every candidate."""
    p,k=T_hard.shape
    idx=cluster_indices(T_hard)
    active=[c for c in range(k) if len(idx[c])]
    E_cols=[torch.from_numpy(E_low[:,idx[j]].copy()) if len(idx[j]) else None for j in range(k)]
    seg=[torch.from_numpy(ix.copy()).long() for ix in idx]
    def _ET(tp):
        return torch.stack([E_cols[j]@tp[j] if E_cols[j] is not None else torch.zeros(p,dtype=torch.float64) for j in range(k)],dim=1)
    def _split(t):
        out,off=[],0
        for j in range(k): n=len(idx[j]); out.append(t[off:off+n]); off+=n
        return out
    def _A(t_np):
        Ah=np.zeros((k,k)); ET=_ET(_split(torch.from_numpy(t_np))).numpy(); M=np.eye(len(active))
        for a,i in enumerate(active):
            Ki=ET[idx[i],:]; ki=Ki[:,i]; d=ki@ki
            if d>1e-30: M[a,:]=(ki@Ki)[active]/d
        Mi=np.linalg.inv(M); dd=np.diag(Mi).copy()
        Ah[np.ix_(active,active)]=np.eye(len(active))-Mi/dd[None,:]
        return Ah
    def obj(t_np):
        t=torch.tensor(t_np,dtype=torch.float64,requires_grad=True)
        ET=_ET(_split(t)); f=torch.zeros((),dtype=torch.float64)
        for i in active:
            Ki=ET[seg[i],:]; ki=Ki[:,i]
            den=(ki*ki).sum().clamp(min=1e-12); coef=(ki@Ki)/den
            f=f+((Ki-ki.unsqueeze(1)*coef.unsqueeze(0))**2).sum()
        f.backward(); return f.item(), t.grad.numpy().astype(np.float64)
    sizes=[len(ix) for ix in idx]; off=np.cumsum([0]+sizes); cons=[]
    for j in range(k):
        lo,hi=off[j],off[j+1]
        if hi==lo: continue
        cons.append({'type':'eq','fun':(lambda lo,hi: lambda x: np.sum(x[lo:hi]**2)-1.)(lo,hi),
                     'jac':(lambda lo,hi: lambda x: np.concatenate([np.zeros(lo),2*x[lo:hi],np.zeros(p-hi)]))(lo,hi)})
    rng=np.random.RandomState(seed); out=[]
    for _ in range(n_restarts):
        t0=rng.randn(p)
        for j in range(k):
            lo,hi=off[j],off[j+1]
            if hi>lo:
                n=np.linalg.norm(t0[lo:hi])
                if n>1e-12: t0[lo:hi]/=n
        r=minimize(obj,t0,jac=True,method='SLSQP',constraints=cons,tol=1e-9,options={'maxiter':maxiter})
        try: nrm=float(np.linalg.norm(_A(r.x)))
        except Exception: nrm=-1.
        out.append((r.fun,nrm))
    return out

def job(a):
    torch.set_num_threads(1)
    kind,p,k,part,trial=a
    A,Tg,Ag=gen_instance(kind,p,k,0.6,trial)
    E=np.linalg.inv(np.eye(p)-A)
    Th=(Tg!=0).astype(float) if part=='gt' else random_partition(p,k,np.random.default_rng(10_000+trial))
    c=pool_K(A,E,Th,n_restarts=10,seed=trial)
    res=np.array([x[0] for x in c]); nrm=np.array([x[1] for x in c])
    best=res.min(); tie_tol=best*1e-3+1e-8
    tied=nrm[res<=best+tie_tol]
    return dict(kind=kind,p=p,part=part,trial=trial,best_res=best,tie_tol=tie_tol,
        n_tied=len(tied),n_restarts=len(res),
        chosen=float(tied.max()) if len(tied) else np.nan,
        best_overall=float(nrm.max()), worst_tied=float(tied.min()) if len(tied) else np.nan,
        gt_norm=float(np.linalg.norm(Ag)) if part=='gt' else np.nan,
        all_feasible_max=float(nrm[res<=1e-8].max()) if (res<=1e-8).any() else np.nan)

if __name__=='__main__':
    jobs=[(kind,p,k,part,t) for kind in ('acyclic','cyclic') for (p,k) in ((24,4),(40,6))
          for part in ('gt','random') for t in range(15)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',250)
    d['edgeless']=(d.chosen<1e-3).astype(int)
    d['left_on_table']=d.best_overall-d.chosen
    print(d.groupby(['kind','p','part'])[['best_res','tie_tol','n_tied','chosen','best_overall',
          'left_on_table','gt_norm','edgeless']].mean().round(4).to_string())
    print('\ninstances where the CHOSEN A_high is numerically edgeless (||A_high||<1e-3):')
    dgt = d[d.part == 'gt']
    print('  %d/%d overall;  GT partitions only: %d/%d' % (d.edgeless.sum(), len(d), dgt.edgeless.sum(), len(dgt)))
    print('\ninstances where a BETTER-norm restart existed but was excluded by the tie window:')
    print(f'  {int((d.left_on_table>1e-6).sum())}/{len(d)}')
    d.to_csv('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/probeK.csv',index=False)
