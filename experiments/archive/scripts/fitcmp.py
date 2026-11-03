"""fit_abstraction vs fit_abstraction_alternative, on identical partitions."""
import os, sys, time, warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,'/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import (fit_abstraction, fit_abstraction_alternative, get_S,
                              high_level_mask, is_dag)
from src.utils.losses import observation_loss, intervention_loss

OUT='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad/fitcmp.csv'

def score(name, A_high, T, A_low, E_low, k, mask_edges, t):
    E_high = np.linalg.inv(np.eye(k)-A_high)
    S = get_S(T=T, E_low=E_low)
    obs = observation_loss(E_low=E_low,E_high=E_high,S=S,T=T)
    intv = intervention_loss(E_low=E_low,E_high=E_high,S=S,T=T)
    ne = int((np.abs(A_high)>1e-9).sum())
    return {f'{name}_obs':obs, f'{name}_intv':intv, f'{name}_edges':ne,
            f'{name}_lost_all_edges': int(mask_edges>0 and ne==0),
            f'{name}_edges_vs_mask': ne-mask_edges,
            f'{name}_valid': int(obs<1e-8 and intv<1e-8),
            f'{name}_dag': int(is_dag(A_high)), f'{name}_time': t}

def job(args):
    torch.set_num_threads(1)
    trial,n_low,n_high,density,lrec = args
    k=n_high
    A_low,_,_,A_high_gt = sample_low_level_graph(n_high=k,n_low=n_low,density=density,shuffle=True,
        blockiness=0.5,heterogeneity=1.,internal_density=0.,graph_type='one_child',seed=trial)
    torch.manual_seed(1000*trial+7)
    m=HailMary(n_low=n_low,n_high=k,A_low=torch.from_numpy(A_low).float())
    m.train(num_epochs=1000*int(np.log((n_low-k)*k+1)+1), lambda_recon=lrec, verbose=False)
    T_hard=m.get_partition()[0].detach().numpy()
    E_low=np.linalg.inv(np.eye(n_low)-A_low)
    mask_edges=int(high_level_mask(A_low=A_low,T_hard=T_hard).sum())

    row={'trial':trial,'n_low':n_low,'n_high':n_high,'density':density,'lambda_recon':lrec,
         'mask_edges':mask_edges,'gt_edges':int((A_high_gt!=0).sum())}
    t0=time.perf_counter()
    Aa,Ta = fit_abstraction_alternative(A_low=A_low,E_low=E_low,T_hard=T_hard,n_restarts=10)
    row.update(score('alt',Aa,Ta,A_low,E_low,k,mask_edges,time.perf_counter()-t0))
    t0=time.perf_counter()
    try:
        out = fit_abstraction(A_low=A_low,T_hard=T_hard,E_low=E_low,n_restarts=10)
        Ar,Tr = out[0],out[1]
        row.update(score('reg',Ar,Tr,A_low,E_low,k,mask_edges,time.perf_counter()-t0))
        row['reg_failed']=0
    except Exception as e:
        row['reg_failed']=1; row['reg_err']=type(e).__name__
    return row

if __name__=='__main__':
    jobs=[(t,nl,nh,d,lr) for t in range(15) for (nl,nh) in [(50,5),(100,10)]
          for d in [0.4,1.0] for lr in [0.,10.]]
    t0=time.time()
    with mp.Pool(6) as p:
        rows=[]
        for i,r in enumerate(p.imap_unordered(job,jobs),1):
            rows.append(r)
            if i%30==0: print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    pd.set_option('display.width',250)
    print(f'\ndone in {(time.time()-t0)/60:.1f} min   n={len(df)}   reg_failed={df.reg_failed.sum()}\n')
    cols=['mask_edges','alt_edges','reg_edges','alt_lost_all_edges','reg_lost_all_edges',
          'alt_valid','reg_valid','alt_time','reg_time']
    print('=== by regime ==='); print(df.groupby(['density','n_low'])[cols].mean().round(3).to_string())
    print()
    print('=== overall ==='); print(df[cols].mean().round(4).to_string())
    print()
    print('runs where the mask has edges but the fit returns none:')
    print(f'  alternative: {int(df.alt_lost_all_edges.sum())}/{len(df)}')
    print(f'  regular    : {int(df.reg_lost_all_edges.sum())}/{len(df)}')
    print()
    print('median log10 obs/intv:')
    for n in ('alt','reg'):
        print(f'  {n}: obs={np.log10(df[f"{n}_obs"]+1e-18).median():.1f} intv={np.log10(df[f"{n}_intv"]+1e-18).median():.1f}')
