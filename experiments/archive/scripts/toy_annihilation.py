"""Small toy: what distinguishes a good restart from a bad one?
Is a poor-recon A_high SPARSER than truth, or MIS-SCALED, or both?"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch
from scipy.optimize import minimize
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import cluster_indices
from src.utils.losses import reconstruction_loss
np.set_printoptions(precision=4,suppress=True,linewidth=170)

def restart_pool(A_low,E_low,T_hard,n_restarts,seed):
    """fit_abstraction_K's restart loop, returning every candidate."""
    p,k=T_hard.shape; idx=cluster_indices(T_hard)
    active=[c for c in range(k) if len(idx[c])]
    E_cols=[torch.from_numpy(E_low[:,idx[j]].copy()) if len(idx[j]) else None for j in range(k)]
    seg=[torch.from_numpy(ix.copy()).long() for ix in idx]
    def _split(t):
        o=[];off=0
        for j in range(k): n=len(idx[j]); o.append(t[off:off+n]); off+=n
        return o
    def _ET(tp): return torch.stack([E_cols[j]@tp[j] if E_cols[j] is not None else torch.zeros(p,dtype=torch.float64) for j in range(k)],dim=1)
    def _A(t_np):
        ET=_ET(_split(torch.from_numpy(t_np))).numpy(); M=np.eye(len(active)); Ah=np.zeros((k,k))
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
        r=minimize(obj,t0,jac=True,method='SLSQP',constraints=cons,tol=1e-9,options={'maxiter':100})
        T=np.zeros((p,k)); T[np.concatenate(idx),np.repeat(np.arange(k),sizes)]=r.x
        out.append((r.fun,r.x,_A(r.x),T))
    return out,idx

p,k,seed=18,3,4
A_low,Tg,_,Ag=sample_low_level_graph(n_high=k,n_low=p,density=1.0,shuffle=False,
    blockiness=1.,heterogeneity=0.,internal_density=0.,graph_type='random',seed=seed)
E=np.linalg.inv(np.eye(p)-A_low); Th=(Tg!=0).astype(float)
blk=block_structure(A_low,Th)
print(f'toy: p={p} k={k}, blockiness=1 het=0, GT partition. oracle recon='
      f'{reconstruction_loss(A_low=A_low,T=Tg,A_high=Ag):.2e}')
print('true A_high:\n',Ag)
print('low-level block structure (1 = real cross-cluster edge):\n',blk.astype(int))

cands,idx=restart_pool(A_low,E,Th,40,seed)
rows=[]
for (res,t,Ah,T) in cands:
    absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.); mx=max(absA.max(),1e-300)
    pred=absA>1e-6*mx
    # scale-corrected comparison: t_fit[:,i] = c_i * t_gt[:,i]  =>  A_gt ~ A_fit * c_i/c_j
    c=np.array([(T[idx[i],i]@Tg[idx[i],i])/(Tg[idx[i],i]@Tg[idx[i],i]) if len(idx[i]) else 1. for i in range(k)])
    with np.errstate(divide='ignore',invalid='ignore'):
        Acorr=Ah*(c[:,None]/c[None,:])
    rel=np.abs(Acorr-Ag)[blk]/np.maximum(np.abs(Ag)[blk],1e-12)
    rows.append(dict(res=res,recon=reconstruction_loss(A_low=A_low,T=T,A_high=Ah),
        recall=float((pred&blk).sum()/blk.sum()), nrm=float(np.linalg.norm(Ah)),
        n_missing=int((blk&~pred).sum()), scale_err=float(np.median(rel)),
        cos_t=float(np.min([abs(T[idx[i],i]@Tg[idx[i],i])/(np.linalg.norm(T[idx[i],i])*np.linalg.norm(Tg[idx[i],i])) for i in range(k)]))))
import pandas as pd
d=pd.DataFrame(rows).sort_values('recon').reset_index(drop=True)
pd.set_option('display.width',200)
print(f'\n40 restarts sorted by reconstruction  (||A_high_true|| = {np.linalg.norm(Ag):.3f}, {int(blk.sum())} real edges)')
print(d.to_string(float_format=lambda v:f'{v:.4g}'))
