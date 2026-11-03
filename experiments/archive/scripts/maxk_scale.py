import sys, importlib.util, time
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, pandas as pd, multiprocessing as mp
_s=importlib.util.spec_from_file_location('mk','/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk=importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
from src.utils.generate_data import sample_low_level_graph

def job(a):
    gt,d,p,nh,seed=a
    A,Tg,_,Ah=sample_low_level_graph(n_high=nh,n_low=p,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=seed,connected=True,max_attempts=400)
    t0=time.time(); est,g=mk.estimate_max_k(A,n_orders=12,seed=seed); el=time.time()-t0
    info=np.nan
    if g is not None:
        T=np.zeros((p,len(g)))
        for j,m in enumerate(g): T[m,j]=1
        co=T@T.T; np.fill_diagonal(co,0)
        info=(np.abs(A)*co).sum()/np.abs(A).sum()
    return dict(graph_type=gt,p=p,true_nhigh=nh,seed=seed,max_k=est,
                max_k_frac=est/p, info_at_max_k=info,
                low_edges=int((A!=0).sum()), max_outdeg=int((A!=0).sum(1).max()),
                secs=el)

if __name__=='__main__':
    jobs=[]
    for gt,d in [('random',0.5),('one_child',1.0)]:
        for p in (10,20,50,100,200):
            nh=max(3,p//10)
            for s in range(8): jobs.append((gt,d,p,nh,s))
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    df=pd.DataFrame(rows); df.to_csv('/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/maxk.csv',index=False)
    pd.set_option('display.width',260)
    print('MAX k satisfying one-child AND All-In (estimator, exact on all 9 p=10 brute-force checks)')
    print()
    print(df.groupby(['graph_type','p'])[['true_nhigh','low_edges','max_outdeg','max_k','max_k_frac','info_at_max_k','secs']].mean().round(3).to_string())
