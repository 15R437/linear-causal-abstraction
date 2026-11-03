"""fit_abstraction_K with faithful=True: maximise ||A_high||^2 s.t. unit column norms and residual <= eps."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch, pandas as pd, multiprocessing as mp, time
from scipy.optimize import minimize
from src.utils.helper import cluster_indices, fit_abstraction_K
from src.utils.losses import reconstruction_loss

def fit_K_faithful(A_low, E_low, T_hard, n_restarts=10, seed=None, maxiter=100,
                   faithful=True, rtol=1e-9):
    p,k = T_hard.shape
    if E_low is None: E_low = np.linalg.inv(np.eye(p)-A_low)
    idx = cluster_indices(T_hard)
    row_i = np.concatenate(idx) if k else np.array([],int)
    col_i = np.repeat(np.arange(k),[len(ix) for ix in idx])
    active = [c for c in range(k) if len(idx[c])]
    E_cols=[torch.from_numpy(E_low[:,idx[j]].copy()) if len(idx[j]) else None for j in range(k)]
    seg=[torch.from_numpy(ix.copy()).long() for ix in idx]

    def _split(t):
        out,off=[],0
        for j in range(k): n=len(idx[j]); out.append(t[off:off+n]); off+=n
        return out
    def _ET(tp):
        return torch.stack([E_cols[j]@tp[j] if E_cols[j] is not None
                            else torch.zeros(p,dtype=torch.float64) for j in range(k)],dim=1)

    def _M_rows(t):
        """M[a,:] over active, the same coef vector the residual already forms."""
        ET=_ET(_split(t)); rows=[]; Ks=[]
        for i in active:
            Ki=ET[seg[i],:]; ki=Ki[:,i]
            den=(ki*ki).sum().clamp(min=1e-12)
            rows.append(((ki@Ki)/den)[active]); Ks.append((Ki,ki,den))
        return torch.stack(rows), Ks

    def residual_t(Ks):
        f=torch.zeros((),dtype=torch.float64)
        for (Ki,ki,den) in Ks:
            coef=(ki@Ki)/den
            f=f+((Ki-ki.unsqueeze(1)*coef.unsqueeze(0))**2).sum()
        return f

    def res_and_grad(t_np):
        t=torch.tensor(t_np,dtype=torch.float64,requires_grad=True)
        _,Ks=_M_rows(t); f=residual_t(Ks); f.backward()
        return f.item(), t.grad.numpy().astype(np.float64)

    def negnorm_and_grad(t_np):
        t=torch.tensor(t_np,dtype=torch.float64,requires_grad=True)
        try:
            M,_=_M_rows(t)
            Mi=torch.linalg.inv(M); d=torch.diagonal(Mi)
            A=torch.eye(len(active),dtype=torch.float64)-Mi/d.unsqueeze(0)
            f=-(A**2).sum()
        except Exception:
            return 0.0, np.zeros_like(t_np)
        if not torch.isfinite(f): return 0.0, np.zeros_like(t_np)
        f.backward()
        return f.item(), np.nan_to_num(t.grad.numpy().astype(np.float64))

    def A_from_t(t_np):
        Ah=np.zeros((k,k))
        if not active: return Ah
        with torch.no_grad(): M,_=_M_rows(torch.from_numpy(t_np))
        Mi=np.linalg.inv(M.numpy()); dd=np.diag(Mi).copy()
        Ah[np.ix_(active,active)]=np.eye(len(active))-Mi/dd[None,:]
        return Ah

    sizes=[len(ix) for ix in idx]; off=np.cumsum([0]+sizes)
    norm_cons=[]
    for j in range(k):
        lo,hi=off[j],off[j+1]
        if hi==lo: continue
        norm_cons.append({'type':'eq','fun':(lambda lo,hi: lambda x: np.sum(x[lo:hi]**2)-1.)(lo,hi),
            'jac':(lambda lo,hi: lambda x: np.concatenate([np.zeros(lo),2*x[lo:hi],np.zeros(p-hi)]))(lo,hi)})

    def draw(rng):
        t0=rng.randn(p)
        for j in range(k):
            lo,hi=off[j],off[j+1]
            if hi>lo:
                n=np.linalg.norm(t0[lo:hi])
                if n>1e-12: t0[lo:hi]/=n
        return t0

    if faithful:
        cons=[{'type':'ineq','fun':lambda x: rtol-res_and_grad(x)[0],
               'jac':lambda x: -res_and_grad(x)[1]}]+norm_cons
        rng=np.random.RandomState(seed); best_x,best_f=None,np.inf
        for _ in range(n_restarts):
            t0=draw(rng)
            try:
                r=minimize(negnorm_and_grad,t0,jac=True,method='SLSQP',constraints=cons,
                           tol=1e-9,options={'maxiter':maxiter})
            except Exception:
                continue
            nerr=max((abs(np.sum(r.x[off[j]:off[j+1]]**2)-1.) for j in range(k) if off[j+1]>off[j]),default=0.)
            if res_and_grad(r.x)[0]<=max(rtol*1e3,1e-6) and nerr<=1e-6 and r.fun<best_f:
                best_f,best_x=r.fun,r.x
        if best_x is not None:
            A=A_from_t(best_x)
            if np.all(np.isfinite(A)):
                T=np.zeros((p,k)); T[row_i,col_i]=best_x
                return A,T
    return fit_abstraction_K(A_low=A_low,E_low=E_low,T_hard=T_hard,n_restarts=n_restarts,seed=seed,maxiter=maxiter)


def job(a):
    torch.set_num_threads(1)
    kind,p,k,part,trial=a
    A_low,Tg,Ag=gen_instance(kind,p,k,0.6,trial)
    E=np.linalg.inv(np.eye(p)-A_low)
    Th=(Tg!=0).astype(float) if part=='gt' else random_partition(p,k,np.random.default_rng(10_000+trial))
    blk=block_structure(A_low,Th)
    row=dict(kind=kind,p=p,part=part,trial=trial,block_edges=int(blk.sum()))
    for nm,fn in (('K',lambda: fit_abstraction_K(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial)),
                  ('Kf',lambda: fit_K_faithful(A_low,E,Th,n_restarts=10,seed=trial))):
        t0=time.perf_counter()
        Ah,T=fn(); dt=time.perf_counter()-t0
        S=get_S_local(T,Th,A_low)
        rae=float(RAE(A_low=A_low,T=T,A_high=Ah,S=S,E_low=E,T_hard=Th))
        absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.); mx=max(absA.max(),1e-300)
        pred=absA>1e-6*mx
        cm=np.abs(T).max(axis=0); cm[cm==0]=1.; lab=np.argmax(Th,axis=1)
        row.update({f'{nm}_rae':rae,f'{nm}_exact':int(rae<1e-6),f'{nm}_norm':float(np.linalg.norm(Ah)),
            f'{nm}_edgeless':int(np.linalg.norm(Ah)<1e-3),
            f'{nm}_recall':float((pred&blk).sum()/max(int(blk.sum()),1)),
            f'{nm}_recon':float(reconstruction_loss(A_low=A_low,T=T,A_high=Ah)),
            f'{nm}_marg':float(np.mean([np.abs(T[r,lab[r]])/cm[lab[r]]<1e-6 for r in range(p)])),
            f'{nm}_time':dt})
    return row

if __name__=='__main__':
    jobs=[(kind,p,k,part,t) for kind in ('acyclic','cyclic') for (p,k) in ((24,4),(40,6))
          for part in ('gt','random') for t in range(15)]
    t0=time.time()
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',260)
    d.to_csv('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/kfaith.csv',index=False)
    print(f'{len(d)} instances in {(time.time()-t0)/60:.1f} min\n')
    for m in ('K','Kf'):
        print(f'=== {m} ===')
        print(d.groupby(['kind','p','part'])[[f'{m}_exact',f'{m}_recall',f'{m}_norm',f'{m}_edgeless',
              f'{m}_marg',f'{m}_recon',f'{m}_time']].mean().round(3).to_string())
        print()
