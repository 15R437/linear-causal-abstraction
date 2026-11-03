import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/tmp/kfaith.py').read().split("def job(")[0])
exec(open('/tmp/recon_b1.py').read().split("def job(")[0].split("import numpy as np, torch, pandas as pd, multiprocessing as mp")[1])
import numpy as np, torch, pandas as pd, multiprocessing as mp, time
from src.utils.losses import reconstruction_loss
from src.utils.helper import fit_abstraction_alternative

def job(a):
    torch.set_num_threads(1)
    kind,p,k,trial=a
    A_low,Tg,Ag=gen(kind,p,k,0.6,trial,1.0,0.0)     # blockiness=1, heterogeneity=0
    E=np.linalg.inv(np.eye(p)-A_low); Th=(Tg!=0).astype(float)
    blk=block_structure(A_low,Th)
    r={'kind':kind,'p':p,'trial':trial,
       'oracle':float(reconstruction_loss(A_low=A_low,T=Tg,A_high=Ag))}
    for nm,fn in (('K',   lambda: fit_abstraction_K(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial)),
                  ('Kf',  lambda: fit_K_faithful(A_low,E,Th,n_restarts=10,seed=trial)),
                  ('alt', lambda: fit_abstraction_alternative(A_low=A_low,E_low=E,T_hard=Th,n_restarts=10,seed=trial))):
        t0=time.perf_counter()
        try: Ah,T=fn()
        except Exception: continue
        dt=time.perf_counter()-t0
        absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.); mx=max(absA.max(),1e-300)
        r[f'{nm}_recon']=float(reconstruction_loss(A_low=A_low,T=T,A_high=Ah))
        r[f'{nm}_recall']=float(((absA>1e-6*mx)&blk).sum()/max(int(blk.sum()),1))
        r[f'{nm}_rae']=float(RAE(A_low=A_low,T=T,A_high=Ah,S=get_S_local(T,Th,A_low),E_low=E,T_hard=Th))
        r[f'{nm}_time']=dt
    return r

if __name__=='__main__':
    jobs=[(kind,p,k,t) for kind in ('acyclic','cyclic') for (p,k) in ((24,4),(40,6)) for t in range(10)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',250)
    d.to_csv('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/kf_b1.csv',index=False)
    print('blockiness=1, heterogeneity=0, GROUND-TRUTH partitions.  medians:\n')
    c=['oracle','K_recon','Kf_recon','alt_recon','K_recall','Kf_recall','alt_recall','K_rae','Kf_rae']
    g=d.groupby(['kind','p'])[c].median()
    print(g.applymap(lambda v:'-' if pd.isna(v) else (f'{v:.2e}' if abs(v)<0.01 else f'{v:.3f}')).to_string())
    print('\nmean runtime (s):')
    print(d.groupby(['kind','p'])[['K_time','Kf_time','alt_time']].mean().round(2).to_string())
