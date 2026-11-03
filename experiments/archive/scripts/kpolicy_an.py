import pandas as pd, numpy as np
SP='/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'
df=pd.read_csv(f'{SP}/kpolicy.csv')
pd.set_option('display.width',250)
print(f'{len(df)} runs\n')
ORD=['ceiling','0.25p','0.4p']

print('===== HEADLINE: success rate / info loss / recon, by policy and family =====')
g=df.groupby(['graph_type','policy']).agg(n=('success','size'),success=('success','mean'),
    allin=('graph_ok','mean'),onechild=('one_child','mean'),info=('info','mean'),
    recon=('recon','mean'),used=('used','mean'),k=('k','mean'),ceiling=('ceiling','mean'))
print(g.round(3).reindex(ORD,level=1).to_string())

print('\n===== by p =====')
g=df.groupby(['graph_type','p','policy']).agg(success=('success','mean'),info=('info','mean'),
    recon=('recon','mean'),k=('k','mean'),ceiling=('ceiling','mean'),used=('used','mean'),
    maxclust=('max_clust','mean'))
print(g.round(3).reindex(ORD,level=2).to_string())

print('\n===== by true n_high (does ground-truth granularity matter?) =====')
g=df.groupby(['graph_type','true_nhigh','policy']).agg(success=('success','mean'),
    info=('info','mean'),ceiling=('ceiling','mean'),k=('k','mean'),used=('used','mean'))
print(g.round(3).reindex(ORD,level=2).to_string())

print('\n===== over-provisioning: success vs k/ceiling =====')
df['bin']=pd.cut(df.k_over_ceiling,[0,0.5,0.9,1.1,2,5,1e9],
                 labels=['<0.5','0.5-0.9','~1','1.1-2','2-5','>5'])
print(df.groupby(['graph_type','bin'],observed=True).agg(n=('success','size'),
    success=('success','mean'),allin=('graph_ok','mean'),onechild=('one_child','mean'),
    info=('info','mean'),used=('used','mean')).round(3).to_string())

print('\n===== ceiling estimate itself =====')
print(df.groupby(['graph_type','p','true_nhigh']).ceiling.agg(['mean','std','min','max']).round(2).to_string())
print('\nestimator wall-time (s):')
print(df.groupby(['graph_type','p']).t_ceil.agg(['mean','max']).round(3).to_string())
