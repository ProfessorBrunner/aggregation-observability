"""Sec. IV.C: Monte Carlo check of the detectability formula, Eq. (22).
Population: overlap c ~ Beta(mean 0.70, sd 0.25) and first-answer propensity q ~ Beta(mean 0.50, sd 0.25),
joined by a Gaussian copula whose correlation is tuned (on a 4e5-draw reference sample) to a target C_q.
15 covariances (0 and +/- seven magnitudes log-spaced 0.002-0.055) x 5 sizes N; M = 2000 surveys per cell.
Each survey: N/2 respondents in the A->B order, answers drawn from their projective probabilities, and the
two-sided Bonferroni test of G2 (z_alpha = 2.638) with plug-in variance. Analytic power uses Eq. (19) with the
population X, Z of Theorem 2. Seed 20260925. Runtime: a few minutes."""
import numpy as np, pandas as pd
from scipy.stats import norm, beta
from scipy.optimize import brentq
rng=np.random.default_rng(20260925); za=norm.ppf(1-0.05/12)
def bpar(m,sd): k=m*(1-m)/sd**2-1; return m*k,(1-m)*k
ac,bc=bpar(0.70,0.25); aq,bq=bpar(0.50,0.25)
grid=np.linspace(0,1,200001); icdf_c=beta.ppf(grid,ac,bc); icdf_q=beta.ppf(grid,aq,bq)
def draw(n,rho,g):
    z1=g.standard_normal(n); z2=rho*z1+np.sqrt(1-rho**2)*g.standard_normal(n)
    return np.interp(norm.cdf(z1),grid,icdf_c), np.interp(norm.cdf(z2),grid,icdf_q)
ref=np.random.default_rng(7)
Z1=ref.standard_normal(400000); Z2=ref.standard_normal(400000)
def cov_for(rho):
    c=np.interp(norm.cdf(Z1),grid,icdf_c); q=np.interp(norm.cdf(rho*Z1+np.sqrt(1-rho**2)*Z2),grid,icdf_q)
    return np.cov(c,q)[0,1], c, q
mags=np.geomspace(0.002,0.055,7); targets=np.r_[0.0,mags,-mags]
Ns=[200,500,741,1500,5000]; M=2000; rows=[]
for tgt in targets:
    rho=0.0 if tgt==0 else brentq(lambda r: cov_for(r)[0]-tgt,-0.999,0.999)
    C,c,q=cov_for(rho); cb,qb=c.mean(),q.mean()
    X=cb+C/qb; Z=cb-C/(1-qb); G=X-Z
    for N in Ns:
        se=np.sqrt((2/N)*(X*(1-X)/qb+Z*(1-Z)/(1-qb))); th=abs(G)/se
        pw=norm.cdf(th-za)+norm.cdf(-th-za)
        h=N//2; cc,qq=draw(M*h,rho,rng); cc=cc.reshape(M,h); qq=qq.reshape(M,h)
        ya=rng.random((M,h))<qq; same=rng.random((M,h))<cc
        n11=np.sum(ya&same,1); n10=np.sum(ya&~same,1); n00=np.sum(~ya&same,1); n01=np.sum(~ya&~same,1)
        ny=n11+n10; nn=n00+n01; ok=(ny>0)&(nn>0)
        Xh=np.where(ny>0,n11/np.maximum(ny,1),0); Zh=np.where(nn>0,n00/np.maximum(nn,1),0)
        var=Xh*(1-Xh)/np.maximum(ny,1)+Zh*(1-Zh)/np.maximum(nn,1)
        rej=ok&(var>0)&(np.abs(Xh-Zh)>za*np.sqrt(np.where(var>0,var,1)))
        rows.append((tgt,C,N,pw,rej.mean()))
o=pd.DataFrame(rows,columns=['target','C_q','N','analytic','empirical']); o['diff']=o.empirical-o.analytic
o.to_csv('../data/power_validation.csv',index=False)
nz=o[o.target!=0]; d=nz['diff'].abs()
print("cells with C_q != 0: %d; |empirical - analytic|: median %.4f, mean %.4f, rms %.4f, max %.4f"%(len(nz),d.median(),d.mean(),np.sqrt((d**2).mean()),d.max()))
w=nz.loc[d.idxmax()]; print("largest at N=%d, C_q=%.4f: analytic %.3f, empirical %.3f"%(w.N,w.C_q,w.analytic,w.empirical))
z=o[o.target==0]; print("false-positive rate at C_q=0 by N:", dict(zip(z.N,z.empirical.round(4))), " nominal", round(2*(1-norm.cdf(za)),4))
