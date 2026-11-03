import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph, generate_random_T, T_concretisation_sampling
from src.utils.losses import reconstruction_loss
from src.utils.helper import fit_abstraction, fit_abstraction_alternative

def gen(kind,p,k,dens,seed,blockiness,heterogeneity):
    if kind=='acyclic':
        A,T,_,Ah=sample_low_level_graph(n_high=k,n_low=p,density=dens,shuffle=True,
            blockiness=blockiness,heterogeneity=heterogeneity,internal_density=0.,
            graph_type='random',seed=seed)
        return A,T,Ah
    rng=np.random.default_rng(seed); Ah=cyclic_A_high(k,dens,0.7,rng)
    T,_=generate_random_T(p,k,(1.,-2.),(2.,-1.),seed=seed)
    A,_=T_concretisation_sampling(A_high=Ah,T=T,internal_density=0.,blockiness=blockiness,
                                  heterogeneity=heterogeneity,seed=seed)
    perm=rng.permutation(p)
    return A[perm][:,perm],T[perm],Ah

def job(a):
    torch.set_num_threads(1)
    kind,p,k,b,h,trial=a
    A_low,Tg,Ag=gen(kind,p,k,0.6,trial,b,h)
    E=np.linalg.inv(np.eye(p)-A_low); Th=(Tg!=0).astype(float)
    blk=block_structure(A_low,Th)
    r={'kind':kind,'p':p,'blockiness':b,'heterogeneity':h,'trial':trial,
       'recon_oracle':float(reconstruction_loss(A_low=A_low,T=Tg,A_high=Ag))}
    for nm,fn in (('K',lambda: fit_abstraction_K(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial)),
                  ('alt',lambda: fit_abstraction_alternative(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial)),
                  ('reg',lambda: fit_abstraction(A_low=A_low,T_hard=Th,E_low=E,n_restarts=10,seed=trial))):
        try:
            Ah,T=fn()
        except Exception:
            continue
        absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.); mx=max(absA.max(),1e-300)
        r[f'{nm}_recon']=float(reconstruction_loss(A_low=A_low,T=T,A_high=Ah))
        r[f'{nm}_recall']=float(((absA>1e-6*mx)&blk).sum()/max(int(blk.sum()),1))
        r[f'{nm}_rae']=float(RAE(A_low=A_low,T=T,A_high=Ah,S=get_S_local(T,Th,A_low),E_low=E,T_hard=Th))
    return r

if __name__=='__main__':
    jobs=[(kind,p,k,b,h,t) for kind in ('acyclic','cyclic') for (p,k) in ((24,4),(40,6))
          for (b,h) in ((1.0,0.0),(1.0,1.0),(0.5,1.0)) for t in range(10)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',250)
    c=['recon_oracle','K_recon','alt_recon','reg_recon','K_recall','K_rae']
    g=d.groupby(['kind','p','blockiness','heterogeneity'])[c].median()
    print(g.applymap(lambda v:'-' if pd.isna(v) else (f'{v:.2e}' if abs(v)<0.01 else f'{v:.3f}')).to_string())
