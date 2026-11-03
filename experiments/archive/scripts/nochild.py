"""Random DAGs with the one-child loss OFF, then post-hoc closure to enforce it.

i)  how often does the learned partition satisfy All-In (graph_loss == 0)?
ii) how much edge mass does it destroy?
iii) after merging to enforce one-child, what happens to both?
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss

OUT='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/nochild.csv'

def groups_of(T):
    return [list(np.where(T[:,j])[0]) for j in range(T.shape[1]) if T[:,j].sum()>0]

def child_matrix(A,groups):
    Ab=(A!=0); K=len(groups)
    M=np.zeros((K,K),bool)
    for a in range(K):
        for b in range(K):
            if a!=b and Ab[np.ix_(groups[a],groups[b])].any(): M[a,b]=True
    return M

def enforce_one_child(A,groups):
    """Merge the children of any cluster with more than one, to a fixed point."""
    groups=[list(g) for g in groups]
    while True:
        M=child_matrix(A,groups)
        done=True
        for a in range(len(groups)):
            kids=list(np.where(M[a])[0])
            if len(kids)>1:
                keep=kids[0]
                for o in sorted(kids[1:],reverse=True):
                    groups[keep]=groups[keep]+groups[o]
                    groups.pop(o)
                done=False
                break
        if done: return groups

def to_T(groups,p,k=None):
    K=k or len(groups)
    T=np.zeros((p,max(K,len(groups))))
    for j,mem in enumerate(groups): T[mem,j]=1
    return T

def stats(A,groups):
    p=A.shape[0]; T=to_T(groups,p)
    M=child_matrix(A,groups)
    co=T@T.T; np.fill_diagonal(co,0)
    info=(np.abs(A)*co).sum()/max(np.abs(A).sum(),1e-12)
    return dict(k=len(groups), graph_loss=graph_loss(A_low=A,T_hard=T),
                one_child=int(M.sum(1).max()<=1 if len(groups) else 1),
                info=info, max_clust=int(max(len(g) for g in groups)),
                edges=int(M.sum()))

def job(args):
    torch.set_num_threads(1)
    gt,d,nl,nh,trial,r=args
    A,Tg,_,Ah=sample_low_level_graph(n_high=nh,n_low=nl,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=trial,connected=True,max_attempts=300)
    gen=torch.Generator().manual_seed(800000+1000*trial+r)
    li=closure_logits(A,nh,rng=np.random.default_rng(800000+1000*trial+r),generator=gen)
    m=HailMary(n_low=nl,n_high=nh,A_low=torch.from_numpy(A).float(),logits_init=li)
    # ONE-CHILD LOSS OFF
    m.train(num_epochs=1000*int(np.log((nl-nh)*nh+1)+1),
            lambda_recon=0., lambda_graph=10., lambda_one_child=0.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T=m.get_partition()[0].detach().numpy()
    g0=groups_of(T); s0=stats(A,g0)
    g1=enforce_one_child(A,g0); s1=stats(A,g1)
    row={'graph_type':gt,'n_low':nl,'n_high':nh,'trial':trial,'restart':r,
         'gt_max':int((np.abs(Tg)>0).sum(0).max()),'gt_edges':int((Ah!=0).sum())}
    row.update({f'pre_{k}':v for k,v in s0.items()})
    row.update({f'post_{k}':v for k,v in s1.items()})
    return row

if __name__=='__main__':
    jobs=[('random',0.5,100,10,t,r) for t in range(15) for r in range(3)]
    jobs+=[('random',0.5,200,20,t,r) for t in range(8) for r in range(3)]
    jobs+=[('one_child',1.0,100,10,t,r) for t in range(15) for r in range(3)]
    t0=time.time()
    with mp.Pool(6) as p:
        rows=[]
        for i,x in enumerate(p.imap_unordered(job,jobs),1):
            rows.append(x)
            if i%20==0: print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    pd.set_option('display.width',280)
    print(f'\ndone {(time.time()-t0)/60:.1f} min\n')
    for (gt,nl,nh),g in df.groupby(['graph_type','n_low','n_high']):
        print(f'===== {gt} p={nl},k={nh}  (one-child loss OFF, {len(g)} runs) =====')
        print(f"  BEFORE closure:  All-In ok {(g.pre_graph_loss<1e-8).mean():.3f}   one-child ok {g.pre_one_child.mean():.3f}   "
              f"info {g.pre_info.mean():.3f}   k={g.pre_k.mean():.1f}  max_clust {g.pre_max_clust.mean():.1f} (gt {g.gt_max.mean():.0f})  edges {g.pre_edges.mean():.1f} (gt {g.gt_edges.mean():.0f})")
        print(f"  AFTER  closure:  All-In ok {(g.post_graph_loss<1e-8).mean():.3f}   one-child ok {g.post_one_child.mean():.3f}   "
              f"info {g.post_info.mean():.3f}   k={g.post_k.mean():.1f}  max_clust {g.post_max_clust.mean():.1f}            edges {g.post_edges.mean():.1f}")
        sub=g[g.pre_graph_loss<1e-8]
        if len(sub):
            print(f"  of the {len(sub)} runs that had All-In BEFORE: kept it after closure in {(sub.post_graph_loss<1e-8).mean():.3f}; "
                  f"info {sub.pre_info.mean():.3f} -> {sub.post_info.mean():.3f} (delta {(sub.post_info-sub.pre_info).mean():+.3f})")
        print()
