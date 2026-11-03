"""Does selecting the restart with the LOWEST reconstruction_loss rescue faithful=False?
reconstruction_loss needs no ground truth, so this is a legal selection rule."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('/tmp/toy.py').read().split("p,k,seed=18,3,4")[0])
import numpy as np, pandas as pd, multiprocessing as mp, torch, time
from src.utils.helper import fit_abstraction_K

def job(a):
    torch.set_num_threads(1)
    p,k,nr,trial=a
    A,Tg,_,Ag=sample_low_level_graph(n_high=k,n_low=p,density=0.6,shuffle=True,
        blockiness=1.,heterogeneity=0.,internal_density=0.,graph_type='random',seed=trial)
    E=np.linalg.inv(np.eye(p)-A); Th=(Tg!=0).astype(float)
    t0=time.perf_counter()
    cands,idx=restart_pool(A,E,Th,nr,trial)
    recons=[reconstruction_loss(A_low=A,T=T,A_high=Ah) for (_,_,Ah,T) in cands]
    t_pool=time.perf_counter()-t0
    # what the shipped tie-break picks
    res=np.array([c[0] for c in cands]); nrm=np.array([np.linalg.norm(c[2]) for c in cands])
    tie=res.min()*1e-3+1e-8; mask=res<=res.min()+tie
    pick=int(np.arange(len(cands))[mask][np.argmax(nrm[mask])])
    t0=time.perf_counter()
    A1,T1=fit_abstraction_K(A_low=A,E_low=E,T_hard=Th,n_restarts=nr,seed=trial,faithful=True)
    t_f=time.perf_counter()-t0
    return dict(p=p,n_restarts=nr,trial=trial,
        tiebreak=recons[pick], minrecon=float(np.min(recons)),
        faithful=float(reconstruction_loss(A_low=A,T=T1,A_high=A1)),
        t_pool=t_pool,t_faithful=t_f)

if __name__=='__main__':
    jobs=[(p,k,nr,t) for (p,k) in ((24,4),(40,6)) for nr in (10,100) for t in range(20)]
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    d=pd.DataFrame(rows); pd.set_option('display.width',240)
    for c in ('tiebreak','minrecon','faithful'): d[c+'_ok']=(d[c]<1e-3).astype(float)
    g=d.groupby(['p','n_restarts']).agg(
        tiebreak=('tiebreak','median'), minrecon=('minrecon','median'), faithful=('faithful','median'),
        tb_ok=('tiebreak_ok','mean'), mr_ok=('minrecon_ok','mean'), f_ok=('faithful_ok','mean'),
        t_pool=('t_pool','median'), t_faithful=('t_faithful','median'))
    print('GT partition, blockiness=1 het=0.  median reconstruction, and frac < 1e-3\n')
    print(g.round(4).to_string())
