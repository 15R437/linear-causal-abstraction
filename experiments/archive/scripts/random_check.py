"""How does the shipped hail_mary_graph fare on random DAGs?

Compares the shipped best-of-5 (closure init, intra_mass, no surj, no recon) against the
pre-session configuration at the SAME restart budget, so the comparison is not confounded
by one arm getting more attempts.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits, regularised_surjective_init
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output, is_dag
from src.utils.losses import graph_loss, observation_loss, intervention_loss

OUT='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/random_check.csv'

class Legacy(HailMary):
    """Pre-session behaviour: in-edge All-In term, dead hard-count surjective term."""
    def forward(self, temp=1., compute_recon=True):
        r,g,o,s,e,i = super().forward(temp, compute_recon)
        Th,_ = self.get_partition(temp)
        s = (self.n_high - torch.sum(Th.sum(dim=0) > 0)) / self.n_high
        return r,g,o,s,e,i

def score(A,T,k):
    gl=graph_loss(A_low=A,T_hard=T)
    M=(T.T@(A!=0).astype(float)@T)>0; np.fill_diagonal(M,False)
    oc=int(M.sum(1).max()<=1)
    co=T@T.T; np.fill_diagonal(co,0)
    info=(np.abs(A)*co).sum()/max(np.abs(A).sum(),1e-12)
    return gl,oc,info,int(M.sum()),int(T.sum(0).max()),int((T.sum(0)>0).sum())

def job(args):
    torch.set_num_threads(1)
    arm,gt,d,nl,nh,trial=args
    A,Tg,_,Ah=sample_low_level_graph(n_high=nh,n_low=nl,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=trial,connected=True,max_attempts=300)
    epochs=1000*int(np.log((nl-nh)*nh+1)+1)
    best=None; bestkey=None
    t0=time.time()
    for r in range(5):
        sd=770000+1000*trial+r
        gen=torch.Generator().manual_seed(sd)
        if arm=='SHIPPED':
            li=closure_logits(A,nh,rng=np.random.default_rng(sd),generator=gen)
            m=HailMary(n_low=nl,n_high=nh,A_low=torch.from_numpy(A).float(),logits_init=li)
            m.train(num_epochs=epochs,lambda_recon=0.,lambda_graph=10.,lambda_one_child=100.,
                    lambda_surj=0.,lambda_info=10.,verbose=False)
        else:
            torch.manual_seed(sd)
            li=regularised_surjective_init(nl,nh).float()   # hail_mary's version returns a tensor
            m=Legacy(n_low=nl,n_high=nh,A_low=torch.from_numpy(A).float(),logits_init=li)
            m.train(num_epochs=epochs,lambda_recon=10.,lambda_graph=10.,lambda_one_child=100.,
                    lambda_surj=1.,lambda_info=0.,verbose=False)
        T=m.get_partition()[0].detach().numpy()
        gl,oc,info,E,mx,ku=score(A,T,nh)
        key=(0 if (gl<1e-8 and oc) else 1, info if (gl<1e-8 and oc) else gl)
        if bestkey is None or key<bestkey: best,bestkey=(T,gl,oc,info,E,mx,ku),key
    rt=time.time()-t0
    T,gl,oc,info,E,mx,ku=best
    Elow=np.linalg.inv(np.eye(nl)-A)
    try:
        S,Tp,Ahp=process_output(T_hard=T,A_low=A,E_low=Elow)
        Eh=np.linalg.inv(np.eye(nh)-Ahp)
        ob=observation_loss(E_low=Elow,E_high=Eh,S=S,T=Tp); iv=intervention_loss(E_low=Elow,E_high=Eh,S=S,T=Tp)
        dag=int(is_dag(Ahp))
    except Exception:
        ob=iv=np.nan; dag=0
    return dict(arm=arm,graph_type=gt,n_low=nl,n_high=nh,trial=trial,
                graph_loss=gl,one_child=oc,success=int(gl<1e-8 and oc),
                info=info,edges=E,gt_edges=int((Ah!=0).sum()),
                max_clust=mx,gt_max=int((np.abs(Tg)>0).sum(0).max()),
                clusters=ku,obs=ob,intv=iv,valid=int(ob<1e-8 and iv<1e-8) if ob==ob else 0,
                dag=dag,runtime=rt)

if __name__=='__main__':
    jobs=[(a,'random',0.5,100,10,t) for a in ('SHIPPED','LEGACY') for t in range(10)]
    jobs+=[(a,'random',0.5,200,20,t) for a in ('SHIPPED','LEGACY') for t in range(8)]
    t0=time.time()
    with mp.Pool(6) as p:
        rows=[]
        for i,x in enumerate(p.imap_unordered(job,jobs),1):
            rows.append(x)
            if i%6==0: print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    pd.set_option('display.width',280)
    print(f'\ndone {(time.time()-t0)/60:.1f} min\n')
    for (nl,nh),g in df.groupby(['n_low','n_high']):
        print(f'===== random p={nl}, k={nh}  (best-of-5 both arms, {g.trial.nunique()} problems) =====')
        print(g.groupby('arm')[['success','graph_loss','one_child','valid','info','clusters',
                                'max_clust','gt_max','edges','gt_edges','dag','runtime']].mean().round(3).to_string())
        a=g[g.arm=='SHIPPED'].set_index('trial'); b=g[g.arm=='LEGACY'].set_index('trial')
        i=a.index.intersection(b.index)
        for m in ['success','info','clusters']:
            dd=(a.loc[i,m]-b.loc[i,m]).dropna(); se=dd.std(ddof=1)/np.sqrt(len(dd)) if len(dd)>1 else 0
            print(f'   paired (shipped-legacy) {m:>9}: {dd.mean():+.3f} [{dd.mean()-1.96*se:+.3f},{dd.mean()+1.96*se:+.3f}]')
        print()
