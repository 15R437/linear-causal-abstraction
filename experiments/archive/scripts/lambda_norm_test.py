"""Soft lambda*||offdiag(M)||^2 reward added to fit_abstraction_K's residual.
M[i,j] = e_ij/e_ii, so this is ||E_high|| in per-cluster-normalised form."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch, pandas as pd, multiprocessing as mp
from scipy.optimize import minimize
from src.utils.losses import reconstruction_loss

def fit_K_lambda(A_low,E_low,T_hard,lam=0.0,n_restarts=10,seed=None,maxiter=100):
    p,k=T_hard.shape
    idx=cluster_indices(T_hard)
    row_i=np.concatenate(idx) if k else np.array([],int)
    col_i=np.repeat(np.arange(k),[len(ix) for ix in idx])
    active=[c for c in range(k) if len(idx[c])]
    E_cols=[torch.from_numpy(E_low[:,idx[j]].copy()) if len(idx[j]) else None for j in range(k)]
    seg=[torch.from_numpy(ix.copy()).long() for ix in idx]
    def _split(t):
        out,off=[],0
        for j in range(k):
            n=len(idx[j]); out.append(t[off:off+n]); off+=n
        return out
    def _ET(tp):
        return torch.stack([E_cols[j]@tp[j] if E_cols[j] is not None
                            else torch.zeros(p,dtype=torch.float64) for j in range(k)],dim=1)
    def obj(t_np):
        t=torch.tensor(t_np,dtype=torch.float64,requires_grad=True)
        ET=_ET(_split(t)); f=torch.zeros((),dtype=torch.float64); rew=torch.zeros((),dtype=torch.float64)
        for i in active:
            Ki=ET[seg[i],:]; ki=Ki[:,i]
            den=(ki*ki).sum().clamp(min=1e-12); coef=(ki@Ki)/den
            f=f+((Ki-ki.unsqueeze(1)*coef.unsqueeze(0))**2).sum()
            m=coef.clone(); m=torch.cat([m[:i],torch.zeros(1,dtype=torch.float64),m[i+1:]])
            rew=rew+(m**2).sum()                      # ||offdiag(M) row i||^2
        L=f-lam*rew
        L.backward()
        return L.item(), t.grad.numpy().astype(np.float64)
    sizes=[len(ix) for ix in idx]; off=np.cumsum([0]+sizes); cons=[]
    for j in range(k):
        lo,hi=off[j],off[j+1]
        if hi==lo: continue
        cons.append({'type':'eq','fun':(lambda lo,hi: lambda x: np.sum(x[lo:hi]**2)-1.)(lo,hi)})
    rng=np.random.RandomState(seed); best=(np.inf,None)
    for _ in range(n_restarts):
        t0=rng.randn(p)
        for j in range(k):
            lo,hi=off[j],off[j+1]
            if hi>lo:
                n=np.linalg.norm(t0[lo:hi]);  t0[lo:hi]/= n if n>1e-12 else 1.
        r=minimize(obj,t0,jac=True,method='SLSQP',constraints=cons,tol=1e-9,options={'maxiter':maxiter})
        if r.fun<best[0]: best=(r.fun,r.x)
    T=np.zeros((p,k)); T[row_i,col_i]=best[1]
    # A_high from the same M route
    ET=_ET(_split(torch.from_numpy(best[1]))).numpy(); M=np.eye(len(active))
    for a,i in enumerate(active):
        Ki=ET[idx[i],:]; ki=Ki[:,i]; d=ki@ki
        if d>1e-30: M[a,:]=(ki@Ki)[active]/d
    Mi=np.linalg.inv(M); dd=np.diag(Mi).copy()
    Ah=np.zeros((k,k)); Ah[np.ix_(active,active)]=np.eye(len(active))-Mi/dd[None,:]
    return Ah,T

def job(a):
    torch.set_num_threads(1)
    p,k,part,trial,lam=a
    A_low,T_gt,_=gen_instance('acyclic',p,k,0.6,trial)
    E=np.linalg.inv(np.eye(p)-A_low)
    Th=(T_gt!=0).astype(float) if part=='gt' else random_partition(p,k,np.random.default_rng(10_000+trial))
    blk=block_structure(A_low,Th)
    Ah,T=fit_K_lambda(A_low,E,Th,lam=lam,n_restarts=10,seed=trial)
    S=get_S_local(T,Th,A_low)
    rae=float(RAE(A_low=A_low,T=T,A_high=Ah,S=S,E_low=E,T_hard=Th))
    cm=np.abs(T).max(axis=0); cm[cm==0]=1.
    lab=np.argmax(Th,axis=1)
    marg=np.mean([np.abs(T[r,lab[r]])/cm[lab[r]]<1e-6 for r in range(p)])
    absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.); mx=max(absA.max(),1e-300)
    pred=absA>1e-6*mx
    return dict(p=p,part=part,lam=lam,trial=trial,rae=rae,exact=int(rae<1e-6),marg=marg,
        recall=float((pred&blk).sum()/max(int(blk.sum()),1)),
        recon=float(reconstruction_loss(A_low=A_low,T=T,A_high=Ah)))

if __name__=='__main__':
    jobs=[(p,k,part,t,lam) for (p,k) in ((24,4),(40,6)) for part in ('gt','random')
          for t in range(10) for lam in (0.0,0.01,0.1,1.0,10.0)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',250)
    print(d.groupby(['p','part','lam'])[['exact','rae','marg','recall','recon']].agg(
        {'exact':'mean','rae':'median','marg':'mean','recall':'mean','recon':'mean'}).round(4).to_string())
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/lambda_norm_test.csv',index=False)
