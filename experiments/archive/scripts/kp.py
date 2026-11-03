import sys, importlib.util, time
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
OUTCSV='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/kp.csv'
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss
_s=importlib.util.spec_from_file_location('mk','/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk=importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)

def job(a):
    torch.set_num_threads(1)
    gt,d,p,nh,k,seed=a
    A,_,_,_=sample_low_level_graph(n_high=nh,n_low=p,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=seed,connected=True,max_attempts=400)
    # the usual epoch rule degenerates to 1000 when k == p, so use a fixed generous budget
    epochs=6000
    gen=torch.Generator().manual_seed(4242+seed)
    li=closure_logits(A,k,rng=np.random.default_rng(4242+seed),generator=gen)
    m=HailMary(n_low=p,n_high=k,A_low=torch.from_numpy(A).float(),logits_init=li)
    t0=time.time()
    m.train(num_epochs=epochs,lambda_recon=0.,lambda_graph=10.,lambda_one_child=100.,
            lambda_surj=0.,lambda_info=10.,verbose=False)
    rt=time.time()-t0
    T=m.get_partition()[0].detach().numpy()
    M=(T.T@(A!=0).astype(float)@T)>0; np.fill_diagonal(M,False)
    co=T@T.T; np.fill_diagonal(co,0)
    ceil,_=mk.estimate_max_k(A,n_orders=8,seed=seed)
    return dict(graph_type=gt,p=p,k_offered=k,seed=seed,
                used=int((T.sum(0)>0).sum()), ceiling=ceil,
                graph_ok=int(graph_loss(A_low=A,T_hard=T)<1e-8),
                one_child=int(M.sum(1).max()<=1),
                info=(np.abs(A)*co).sum()/np.abs(A).sum(),
                max_clust=int(T.sum(0).max()), secs=rt)

if __name__=='__main__':
    jobs=[]
    for gt,d in [('one_child',1.0),('random',0.5)]:
        for p in (50,100):
            nh=max(3,p//10)
            for k in (p//2, p):
                for s in range(5): jobs.append((gt,d,p,nh,k,s))
    with mp.Pool(6) as pool: rows=pool.map(job,jobs)
    df=pd.DataFrame(rows); df.to_csv(OUTCSV,index=False)
    pd.set_option('display.width',260)
    df['ratio']=df.used/df.ceiling
    print('HailMary offered k = p/2 and k = p, vs the estimated ceiling')
    print()
    print(df.groupby(['graph_type','p','k_offered'])[
        ['used','ceiling','ratio','graph_ok','one_child','info','max_clust','secs']].mean().round(3).to_string())
