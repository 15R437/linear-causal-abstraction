"""Test the participation-ratio-minus-one form of one_child_loss.

    pr = (sum H)^2 / sum H^2   == c  (the effective number of child clusters)
    term = mean over ACTIVE nodes of (pr - 1)

Unbounded in c like the current form, so severe violations stay expensive; invariant to
the mass magnitude a, so no k-drift. That combination is what neither previous variant had.
"""
import sys, time
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss

OUT='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/cminus1.csv'

class Variant(HailMary):
    def __init__(self,*a,oc_norm='current',**kw):
        self.oc_norm=oc_norm
        super().__init__(*a,**kw)
    def forward(self,temp=1.,compute_recon=True):
        r,g,oc,s,e,inf=super().forward(temp,compute_recon)
        if self.oc_norm=='current':
            return r,g,oc,s,e,inf
        _,T_soft=self.get_partition(temp)
        Ab=self.A_low_binary
        H=torch.max(Ab[:,:,None,:]*T_soft.T[None,None,:,:],dim=-1)[0]*(1-T_soft)
        s1=torch.sum(H,dim=-1); s2=torch.sum(H**2,dim=-1)
        active=(s1>1e-3).float()
        pr=(s1**2)/s2.clamp(min=1e-8)            # == c
        oc2=((active*(pr-1.0).clamp(min=0.0)).sum()/active.sum().clamp(min=1.0))
        return r,g,oc2,s,e,inf

def job(a):
    torch.set_num_threads(1)
    norm,lam,gt,d,p,k,seed=a
    A,_,_,_=sample_low_level_graph(n_high=max(3,p//10),n_low=p,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=seed,connected=True,max_attempts=400)
    gen=torch.Generator().manual_seed(31337+seed)
    li=closure_logits(A,k,rng=np.random.default_rng(31337+seed),generator=gen)
    m=Variant(n_low=p,n_high=k,A_low=torch.from_numpy(A).float(),logits_init=li,oc_norm=norm)
    m.train(num_epochs=6000,lambda_recon=0.,lambda_graph=10.,lambda_one_child=lam,
            lambda_surj=0.,lambda_info=10.,verbose=False)
    T=m.get_partition()[0].detach().numpy()
    M=(T.T@(A!=0).astype(float)@T)>0; np.fill_diagonal(M,False)
    co=T@T.T; np.fill_diagonal(co,0)
    gl=graph_loss(A_low=A,T_hard=T); oc=int(M.sum(1).max()<=1)
    return dict(norm=norm,lam=lam,graph_type=gt,p=p,k=k,seed=seed,
                used=int((T.sum(0)>0).sum()), graph_ok=int(gl<1e-8), one_child=oc,
                success=int(gl<1e-8 and oc), info=(np.abs(A)*co).sum()/np.abs(A).sum(),
                maxkids=int(M.sum(1).max()), max_clust=int(T.sum(0).max()))

if __name__=='__main__':
    jobs=[]
    for gt,d in [('random',0.5),('one_child',1.0)]:
        for k in (25,50,100):
            for norm,lam in [('current',100.),('cminus1',100.),('cminus1',300.)]:
                for s in range(8): jobs.append((norm,lam,gt,d,100,k,s))
    t0=time.time()
    with mp.Pool(6) as pool:
        rows=[]
        for i,r in enumerate(pool.imap_unordered(job,jobs),1):
            rows.append(r)
            if i%36==0: print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    pd.set_option('display.width',260)
    print(f'\ndone {(time.time()-t0)/60:.1f} min\n')
    df['arm']=np.where(df.norm=='current','current L=100','c-1 L='+df.lam.astype(int).astype(str))
    for gt,g in df.groupby('graph_type'):
        print(f'===== {gt} p=100 (8 seeds) =====')
        for m,lab in [('used','CLUSTERS USED (should not fall with k)'),
                      ('success','SUCCESS: All-In AND one-child'),
                      ('one_child','one-child satisfied'),
                      ('info','information loss'),
                      ('maxkids','max children per cluster')]:
            print(lab)
            print(g.pivot_table(index='arm',columns='k',values=m).round(3).to_string()); print()
