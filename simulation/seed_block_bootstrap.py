"""Seed-block bootstrap for the 50% points of the threshold statistic (Supp. S4, Sec. VI.C).

The 256 population seeds are shared across all 21 dispersions (common random numbers), so the independent
sampling unit is a seed's whole response record across dispersions. Each replicate draws seed indices once,
keeps them at every dispersion, and keeps the same indices for all three detectors, so the detector
difference is estimated as a paired quantity. Point estimator: the binomial logistic fit of aggregate.py.
B = 10000, seed 20260926. Runtime: a few minutes. Output: outputs/seed_block_bootstrap.csv."""
import numpy as np, pandas as pd
from aggregate import fit50
D=np.load('outputs/diag_definedness.npz'); C=np.load('outputs/diag_continuous.npz')
def table(npz,col):
    r=npz['rows']; c=list(npz['cols']); sd=r[:,c.index('sd')]; seed=r[:,c.index('seed')]; v=r[:,c.index(col)]
    S=np.unique(sd); K=np.unique(seed)
    M=np.full((len(K),len(S)),np.nan)
    for j,s in enumerate(S):
        m=sd==s; idx=np.searchsorted(K,seed[m]); M[idx,j]=v[m]
    assert not np.isnan(M).any(); return S,K,M
S,K,dense=table(D,'defined'); S2,K2,coarse=table(D,'defined_1200'); S3,K3,event=table(C,'crossed')
assert np.allclose(S,S2) and np.allclose(S,S3) and np.array_equal(K,K2) and np.array_equal(K,K3)
S=S/1.0; n=len(K)
def p50_many(Kc, N):
    """Binomial logistic MLE, logit p = a + b s, for many count vectors at once (rows of Kc); returns -a/b.
    Newton-Raphson; identical estimator to aggregate.fit50 (statsmodels GLM), checked below."""
    X=np.column_stack([np.ones_like(S),S]); th=np.zeros((Kc.shape[0],2)); th[:,0]=5; th[:,1]=-3.5
    for _ in range(100):
        eta=th@X.T; p=1/(1+np.exp(-eta)); g=(Kc-N*p)@X
        w=N*p*(1-p); H=np.einsum('bi,ij,ik->bjk',w,X,X)
        step=np.linalg.solve(H,g[...,None])[...,0]; th+=step
        if np.max(np.abs(step))<1e-12: break
    return -th[:,0]/th[:,1]
N=float(n); full=np.arange(n)
est={k:p50_many(M.sum(0)[None,:],N)[0] for k,M in (('dense',dense),('1200',coarse),('event',event))}
for k,M in (('dense',dense),('1200',coarse),('event',event)):
    ref=fit50(S,M.sum(0),np.full(len(S),n))[0]; assert abs(ref-est[k])<1e-8,(k,ref,est[k])
print("estimator check against aggregate.fit50: passed")
rng=np.random.default_rng(20260926); B=10000
IDX=rng.integers(0,n,(B,n))
cnt=lambda M: np.stack([M[i].sum(0) for i in IDX])
out=np.column_stack([p50_many(cnt(dense),N),p50_many(cnt(coarse),N),p50_many(cnt(event),N)])
ci=lambda x: np.percentile(x,[2.5,97.5])
rows=[]
for j,k in enumerate(('dense','1200','event')):
    lo,hi=ci(out[:,j]); rows.append((k,est[k],lo,hi,out[:,j].std()))
d=out[:,0]-out[:,2]; lo,hi=ci(d); rows.append(('dense-event',est['dense']-est['event'],lo,hi,d.std()))
o=pd.DataFrame(rows,columns=['quantity','estimate','ci_lo','ci_hi','boot_se']); o.to_csv('outputs/seed_block_bootstrap.csv',index=False)
print(o.to_string(index=False,float_format=lambda x:'%.5f'%x))
