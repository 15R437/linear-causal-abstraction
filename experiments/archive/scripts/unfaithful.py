"""With faithful=False on GROUND-TRUTH partitions at blockiness=1 (where recon SHOULD be 0):
how often do we actually get it, and do more restarts rescue it?"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/5d843d73-a9e7-4270-82fa-13ca5dc30040/scratchpad/fit3cmp.py').read().split('if __name__')[0])
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import fit_abstraction_K
from src.utils.losses import reconstruction_loss

def job(a):
    torch.set_num_threads(1)
    p,k,nr,trial=a
    A,Tg,_,Ag=sample_low_level_graph(n_high=k,n_low=p,density=0.6,shuffle=True,
        blockiness=1.,heterogeneity=0.,internal_density=0.,graph_type='random',seed=trial)
    E=np.linalg.inv(np.eye(p)-A); Th=(Tg!=0).astype(float)
    blk=block_structure(A,Th)
    Ah,T=fit_abstraction_K(A_low=A,E_low=E,T_hard=Th,n_restarts=nr,seed=trial,faithful=False)
    absA=np.abs(Ah).copy(); np.fill_diagonal(absA,0.); mx=max(absA.max(),1e-300)
    return dict(p=p,k=k,n_restarts=nr,trial=trial,
        recon=float(reconstruction_loss(A_low=A,T=T,A_high=Ah)),
        recall=float(((absA>1e-6*mx)&blk).sum()/max(int(blk.sum()),1)),
        rae=float(RAE(A_low=A,T=T,A_high=Ah,S=get_S_local(T,Th,A),E_low=E,T_hard=Th)),
        oracle=float(reconstruction_loss(A_low=A,T=Tg,A_high=Ag)))

if __name__=='__main__':
    jobs=[(p,k,nr,t) for (p,k) in ((24,4),(40,6)) for nr in (10,30,100) for t in range(20)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',240)
    d['recovered']=(d.recon<1e-3).astype(float)
    d['vacuous']=(d.recon>0.9).astype(float)
    g=d.groupby(['p','n_restarts']).agg(
        oracle=('oracle','median'), recon_med=('recon','median'),
        recon_min=('recon','min'), recon_max=('recon','max'),
        frac_recovered=('recovered','mean'), frac_vacuous=('vacuous','mean'),
        recall=('recall','mean'), rae_med=('rae','median'))
    print('faithful=False, GROUND-TRUTH partition, blockiness=1 het=0 (oracle recon = 0)\n')
    print(g.round(4).to_string())
