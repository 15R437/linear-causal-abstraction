"""Are the sweep-D 'winners' blob + singletons? Record cluster-size structure."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output
from src.utils.losses import observation_loss, intervention_loss, graph_loss

OUT='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/blobcheck.csv'

class Old(HailMary):
    def forward(self, temp=1., compute_recon=True):
        r,g,o,_,e = super().forward(temp, compute_recon)
        Th,_=self.get_partition(temp)
        return r,g,o,(self.n_high-torch.sum(Th.sum(0)>0))/self.n_high,e

ARMS={'ORIGINAL (recon10,hard)':(Old,dict(lambda_recon=10.,lambda_surj=1.)),
      'SETTLED (recon0,empty10)':(HailMary,dict(lambda_recon=0.,lambda_surj=10.))}

def job(a):
    torch.set_num_threads(1)
    arm,trial,nl,nh,gt,d=a
    A,Tg,_,Ah=sample_low_level_graph(n_high=nh,n_low=nl,density=d,shuffle=True,blockiness=0.5,
        heterogeneity=1.,internal_density=0.,graph_type=gt,seed=trial,connected=True,max_attempts=200)
    cls,kw=ARMS[arm]; torch.manual_seed(1000*trial+7)
    m=cls(n_low=nl,n_high=nh,A_low=torch.from_numpy(A).float())
    m.train(num_epochs=1000*int(np.log((nl-nh)*nh+1)+1),verbose=False,**kw)
    Th=m.get_partition()[0].detach().numpy()
    sizes=Th.sum(0).astype(int)
    E=np.linalg.inv(np.eye(nl)-A)
    S,T,Ahp=process_output(T_hard=Th,A_low=A,E_low=E)
    Eh=np.linalg.inv(np.eye(nh)-Ahp)
    obs=observation_loss(E_low=E,E_high=Eh,S=S,T=T); iv=intervention_loss(E_low=E,E_high=Eh,S=S,T=T)
    gl=graph_loss(A_low=A,T_hard=Th)
    gt_sizes=(np.abs(Tg)>0).sum(0)
    return {'arm':arm,'graph_type':gt,'trial':trial,'n_low':nl,'n_high':nh,
            'max_cluster':int(sizes.max()),'min_cluster':int(sizes.min()),
            'n_singletons':int((sizes==1).sum()),
            'blob_frac':float(sizes.max()/nl),
            'gt_max_cluster':int(gt_sizes.max()),'gt_min_cluster':int(gt_sizes.min()),
            'size_gini':float(np.abs(np.subtract.outer(sizes,sizes)).mean()/(2*sizes.mean())),
            'surjective':int((sizes>0).all()),
            'valid':int(obs<1e-8 and iv<1e-8),'graph_ok':int(gl<1e-8),
            'edges_high':int((np.abs(Ahp)>1e-9).sum()),'edges_high_gt':int((Ah!=0).sum())}

if __name__=='__main__':
    jobs=[(a,t,nl,nh,gt,d) for a in ARMS for t in range(10)
          for (nl,nh) in [(100,10)] for (gt,d) in [('one_child',1.0),('random',0.5)]]
    t0=time.time()
    with mp.Pool(6) as p: rows=p.map(job,jobs)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    pd.set_option('display.width',250)
    print(f'done {(time.time()-t0)/60:.1f} min\n')
    cols=['max_cluster','blob_frac','n_singletons','min_cluster','gt_max_cluster','gt_min_cluster',
          'valid','surjective','graph_ok','edges_high','edges_high_gt']
    print(df.groupby(['graph_type','arm'])[cols].mean().round(2).to_string())
