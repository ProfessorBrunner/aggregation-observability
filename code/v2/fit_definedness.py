"""Logistic fit and bootstrap interval for the 50% point of the threshold statistic."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
d=pd.read_csv('../../data/cs_simulation/definedness_summary.csv'); s=d.sd_over_Delta.values; n=256
for lat in ['24000','1200']:
    k=np.round(d[f'frac_defined_{lat}'].values*n)
    def nll(th,kk):
        p=np.clip(1/(1+np.exp(-(th[0]+th[1]*s))),1e-12,1-1e-12); return -np.sum(kk*np.log(p)+(n-kk)*np.log(1-p))
    r=minimize(nll,[10,-7],args=(k,),method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-10,maxiter=20000))
    rng=np.random.default_rng(7); pf=1/(1+np.exp(-(r.x[0]+r.x[1]*s))); S=[]
    for _ in range(2000):
        rr=minimize(nll,r.x,args=(rng.binomial(n,pf),),method='Nelder-Mead',options=dict(xatol=1e-8,fatol=1e-8)); S.append(-rr.x[0]/rr.x[1])
    print(f"lattice {lat}: 50% point {-r.x[0]/r.x[1]:.3f}, 95% interval [{np.percentile(S,2.5):.3f}, {np.percentile(S,97.5):.3f}]")
