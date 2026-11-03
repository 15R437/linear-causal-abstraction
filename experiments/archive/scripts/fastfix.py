"""Candidate fast fixes for the annihilation failure, benchmarked on GT partitions."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch, pandas as pd, multiprocessing as mp, time
from scipy.optimize import minimize
from src.utils.generate_data import sample_low_level_graph, generate_random_T, T_concretisation_sampling
from src.utils.helper import (cluster_indices, fit_abstraction_K, fit_abstraction_alternative,
                              fit_abstraction)
from src.utils.losses import reconstruction_loss


def emission_init(A_low, E_low, T_hard):
    """t_i = argmax ||O_i E_ii t_i|| s.t. ||t_i||=1, where O_i stacks cluster i's OUTGOING
    blocks. The failure mode is the optimiser picking a t_i that emits nothing; this starts
    at the direction of maximal emission instead. One small SVD per cluster, O(p^3/k) total."""
    p, k = T_hard.shape
    idx = cluster_indices(T_hard)
    t0 = np.zeros(p)
    for i in range(k):
        ni = len(idx[i])
        if ni == 0:
            continue
        out_rows = np.concatenate([idx[j] for j in range(k) if j != i and len(idx[j])]) \
                   if any(j != i and len(idx[j]) for j in range(k)) else np.array([], int)
        E_ii = E_low[np.ix_(idx[i], idx[i])]
        if len(out_rows) == 0:
            v = np.ones(ni)
        else:
            Oi = A_low[np.ix_(out_rows, idx[i])] @ E_ii     # emission operator on t_i
            try:
                v = np.linalg.svd(Oi, full_matrices=False)[2][0]
            except np.linalg.LinAlgError:
                v = np.ones(ni)
        n = np.linalg.norm(v)
        t0[idx[i]] = v / n if n > 1e-12 else np.ones(ni) / np.sqrt(ni)
    return t0


def fit_K_init(A_low, E_low, T_hard, t_init, maxiter=100):
    """fit_abstraction_K's residual minimisation from ONE given starting point."""
    p, k = T_hard.shape
    idx = cluster_indices(T_hard); active = [c for c in range(k) if len(idx[c])]
    E_cols = [torch.from_numpy(E_low[:, idx[j]].copy()) if len(idx[j]) else None for j in range(k)]
    seg = [torch.from_numpy(ix.copy()).long() for ix in idx]
    sizes = [len(ix) for ix in idx]; off = np.cumsum([0] + sizes)
    def _split(t):
        o=[];c=0
        for j in range(k): o.append(t[c:c+len(idx[j])]); c+=len(idx[j])
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
    cons=[]
    for j in range(k):
        lo,hi=off[j],off[j+1]
        if hi==lo: continue
        cons.append({'type':'eq','fun':(lambda lo,hi: lambda x: np.sum(x[lo:hi]**2)-1.)(lo,hi),
                     'jac':(lambda lo,hi: lambda x: np.concatenate([np.zeros(lo),2*x[lo:hi],np.zeros(p-hi)]))(lo,hi)})
    r=minimize(obj,t_init,jac=True,method='SLSQP',constraints=cons,tol=1e-9,options={'maxiter':maxiter})
    T=np.zeros((p,k)); T[np.concatenate(idx),np.repeat(np.arange(k),sizes)]=r.x
    return _A(r.x),T


def gen(kind,p,k,seed):
    if kind=='acyclic':
        A,T,_,Ah=sample_low_level_graph(n_high=k,n_low=p,density=0.6,shuffle=True,blockiness=1.,
            heterogeneity=0.,internal_density=0.,graph_type='random',seed=seed)
        return A,T,Ah
    rng=np.random.default_rng(seed); Ah=cyclic_A_high(k,0.6,0.7,rng)
    T,_=generate_random_T(p,k,(1.,-2.),(2.,-1.),seed=seed)
    A,_=T_concretisation_sampling(A_high=Ah,T=T,internal_density=0.,blockiness=1.,heterogeneity=0.,seed=seed)
    perm=rng.permutation(p)
    return A[perm][:,perm],T[perm],Ah

def job(a):
    torch.set_num_threads(1)
    kind,p,k,trial=a
    A,Tg,Ag=gen(kind,p,k,trial)
    E=np.linalg.inv(np.eye(p)-A); Th=(Tg!=0).astype(float)
    r={'kind':kind,'p':p,'trial':trial,'oracle':float(reconstruction_loss(A_low=A,T=Tg,A_high=Ag))}
    def timed(nm,fn):
        t0=time.perf_counter()
        try: Ah,T=fn()
        except Exception: return
        r[f'{nm}_t']=time.perf_counter()-t0
        r[f'{nm}_rec']=float(reconstruction_loss(A_low=A,T=T,A_high=Ah))
    timed('base',   lambda: fit_abstraction_K(A_low=A,E_low=E,T_hard=Th,n_restarts=10,seed=trial,faithful=False))
    timed('Kfaith', lambda: fit_abstraction_K(A_low=A,E_low=E,T_hard=Th,n_restarts=10,seed=trial,faithful=True))
    timed('emit',   lambda: fit_K_init(A,E,Th,emission_init(A,E,Th)))
    timed('altf',   lambda: fit_abstraction_alternative(A_low=A,E_low=E,T_hard=Th,n_restarts=10,seed=trial,faithful=True))
    timed('prop',   lambda: fit_abstraction(A_low=A,T_hard=Th,E_low=E,n_restarts=10,seed=trial,faithful=True))
    return r

if __name__=='__main__':
    jobs=[(kind,p,k,t) for kind in ('acyclic','cyclic') for (p,k) in ((24,4),(40,6),(60,8),(100,10)) for t in range(8)]
    t0=time.time()
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',250)
    d.to_csv('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fastfix.csv',index=False)
    print(f'wall {(time.time()-t0)/60:.1f} min   (6 workers, so absolute times are contended)\n')
    ms=['base','emit','altf','prop','Kfaith']
    print('MEDIAN RECONSTRUCTION  (oracle ~1e-16)')
    print(d.groupby(['kind','p'])[[f'{m}_rec' for m in ms]].median().applymap(lambda v:'-' if pd.isna(v) else f'{v:.1e}').to_string())
    print('\nMEDIAN TIME (s)')
    print(d.groupby(['kind','p'])[[f'{m}_t' for m in ms]].median().round(2).to_string())
