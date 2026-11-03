"""Paired read of the hard-loss smoke test. Arms share an init seed per (instance, lambda)."""
import sys, numpy as np, pandas as pd
B='/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
d=pd.read_csv(f'{B}/data/hardmix.csv')
hm=d[d.method=='HailMary']; occ=d[d.method=='OCC'].set_index(['density','seed'])
pd.set_option('display.width',240)
print('n rows %d | arms %s | lam %s'%(len(hm),sorted(hm.arm.unique()),sorted(hm.lam_oc.unique())))

print('\n=== medians by arm (pooled over 3 densities x 3 instances) ===')
t=hm.groupby(['lam_oc','arm']).agg(n=('rae','size'),rae=('rae','median'),iem=('iem','median'),
    allin=('allin','mean'),onechild=('one_child','mean'),valid=('valid','mean'),
    k_used=('k_used','median'),marg=('marg','median'),secs=('secs','mean')).round(3)
print(t.to_string())

print('\n=== per density ===')
for lam in sorted(hm.lam_oc.unique()):
    print('\n-- lambda_oc = %g'%lam)
    t=hm[hm.lam_oc==lam].groupby(['density','arm']).agg(
        rae=('rae','median'),iem=('iem','median'),allin=('allin','mean'),
        k_used=('k_used','median')).round(3).unstack('arm')
    print(t.to_string())

print('\n=== PAIRED vs baseline arm "none" (same instance, same init) ===')
base=hm[hm.arm=='none'].set_index(['density','seed','lam_oc'])
for metric,better in (('rae','lower'),('iem','lower')):
    print('\n  %s (%s is better)  -- median paired delta [win/loss/tie over 9 instances]'%(metric,better))
    for lam in sorted(hm.lam_oc.unique()):
        row='    lam=%-6g'%lam
        for arm in ('graph','recon','both'):
            s=hm[(hm.arm==arm)&(hm.lam_oc==lam)].set_index(['density','seed','lam_oc'])
            idx=s.index.intersection(base.index)
            if not len(idx): continue
            dl=(s.loc[idx,metric]-base.loc[idx,metric]).values
            w=(dl<-1e-9).sum(); l=(dl>1e-9).sum(); t_=len(dl)-w-l
            row+='  %-6s %+.4f [%d/%d/%d]'%(arm,np.median(dl),w,l,t_)
        print(row)

print('\n=== hybrid IEM at alpha=0.05 (accept if rae<=a, else fall back to closure) ===')
for lam in sorted(hm.lam_oc.unique()):
    print('  lam=%g'%lam)
    for arm in ('none','graph','recon','both'):
        s=hm[(hm.arm==arm)&(hm.lam_oc==lam)]
        vals=[];acc=0
        for (dn,sd),g in s.groupby(['density','seed']):
            b=occ.loc[(dn,sd),'iem']
            ok=g[g.rae<=0.05]
            if len(ok): vals.append(min(ok.iem.min(),b)); acc+=1
            else: vals.append(b)
        print('    %-6s acc %.2f  IEM %.3f'%(arm,acc/max(len(vals),1),np.median(vals)))
