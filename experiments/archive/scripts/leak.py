"""Does softmax leakage into empty columns cause the k-drift in one_child_loss?

The drift is driven by sum H^2 collapsing as tiny leaked values spread over k-1 empty
columns, inflating the effective child count. Two variants that target the leakage rather
than the functional form:

  hard   : build H from T_hard (straight-through), so empty columns contribute exactly 0
  thresh : zero H entries below 0.05 before summing

The user reports having tried this direction before and settled on T_soft; this records
the comparison rather than assuming it.
"""
import sys, time
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss

OUT='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/leak.csv'

class Variant(HailMary):
    def __init__(self,*a,mode='current',**kw):
        self.mode=mode
        super().__init__(*a,**kw)
    def forward(self,temp=1.,compute_recon=True):
        r,g,oc,s,e,inf=super().forward(temp,compute_recon)
        if self.mode=='current':
            return r,g,oc,s,e,inf
        T_hard,T_soft=self.get_partition(temp)
        Ab=self.A_low_binary
        Tm = T_hard if self.mode=='hard' else T_soft
        H=torch.max(Ab[:,:,None,:]*Tm.T[None,None,:,:],dim=-1)[0]*(1-Tm)
        if self.mode=='thresh':
            H=H*(H>0.05).float()          # gradient flows through the surviving entries
        oc2=torch.mean(torch.sum(H,dim=-1)**2-torch.sum(H**2,dim=-1))
        return r,g,oc2,s,e,inf

def job(a):
    torch.set_num_threads(1)
    mode,gt,d,p,k,seed=a
    A,_,_,_=sample_low_level_graph(n_high=max(3,p//10),n_low=p,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=seed,connected=True,max_attempts=400)
    gen=torch.Generator().manual_seed(31337+seed)
    li=closure_logits(A,k,rng=np.random.default_rng(31337+seed),generator=gen)
    m=Variant(n_low=p,n_high=k,A_low=torch.from_numpy(A).float(),logits_init=li,mode=mode)
    m.train(num_epochs=6000,lambda_recon=0.,lambda_graph=10.,lambda_one_child=100.,
            lambda_surj=0.,lambda_info=10.,verbose=False)
    T=m.get_partition()[0].detach().numpy()
    M=(T.T@(A!=0).astype(float)@T)>0; np.fill_diagonal(M,False)
    co=T@T.T; np.fill_diagonal(co,0)
    gl=graph_loss(A_low=A,T_hard=T); oc=int(M.sum(1).max()<=1)
    return dict(mode=mode,graph_type=gt,p=p,k=k,seed=seed,
                used=int((T.sum(0)>0).sum()), graph_ok=int(gl<1e-8), one_child=oc,
                success=int(gl<1e-8 and oc), info=(np.abs(A)*co).sum()/np.abs(A).sum(),
                maxkids=int(M.sum(1).max()))

if __name__=='__main__':
    jobs=[(mode,gt,d,100,k,s)
          for gt,d in [('random',0.5),('one_child',1.0)]
          for k in (25,50,100)
          for mode in ('current','hard','thresh')
          for s in range(8)]
    t0=time.time()
    with mp.Pool(6) as pool:
        rows=[]
        for i,r in enumerate(pool.imap_unordered(job,jobs),1):
            rows.append(r)
            if i%36==0: print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    pd.set_option('display.width',260)
    print(f'\ndone {(time.time()-t0)/60:.1f} min\n')
    for gt,g in df.groupby('graph_type'):
        print(f'===== {gt} p=100 (8 seeds) =====')
        for m,lab in [('used','CLUSTERS USED'),('success','SUCCESS: All-In AND one-child'),
                      ('one_child','one-child satisfied'),('info','information loss')]:
            print(lab); print(g.pivot_table(index='mode',columns='k',values=m).round(3).to_string()); print()
