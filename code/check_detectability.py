"""Independent check of the Section 4 power boundary (not the ChatGPT script).
Population: c, q bivariate normal clipped to [0.02,0.98]; realized C_q measured on a large draw.
Survey: N respondents, half per order; answers from exact projective probabilities.
Test: Wald on X-hat - Z-hat, Bonferroni over six two-sided tests."""
import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(7)
za=norm.ppf(1-0.05/12)
def pop(n,rho,cbar=0.70,qbar=0.50,s=0.25):
    z=rng.multivariate_normal([0,0],[[1,rho],[rho,1]],size=n)
    return np.clip(cbar+s*z[:,0],.02,.98),np.clip(qbar+s*z[:,1],.02,.98)
def realized(rho):
    c,q=pop(400000,rho); return np.mean((c-c.mean())*(q-q.mean())),c.mean(),q.mean()
def survey(N,rho):
    c,q=pop(N//2,rho); a=rng.random(N//2)<q; agree=rng.random(N//2)<c; b=np.where(a,agree,~agree)
    n0=a.sum(); n1=(~a).sum(); X=(a&b).sum()/n0; Z=(~a&~b).sum()/n1
    return abs((X-Z)/np.sqrt(X*(1-X)/n0+Z*(1-Z)/n1))>za
out=[]
for N,rho in [(741,0.0),(741,0.4976),(200,0.9360),(5000,0.1931)]:
    Cq,cb,qb=realized(rho); G2=Cq/(qb*(1-qb)); X=cb+(1-qb)*G2; Z=cb-qb*G2
    se=np.sqrt((2/N)*(X*(1-X)/qb+Z*(1-Z)/(1-qb))); pred=1-norm.cdf(za-abs(G2)/se) if Cq!=0 else 0.05/6
    emp=np.mean([survey(N,rho) for _ in range(3000)])
    out.append((N,round(Cq,4),round(pred,3),round(emp,3)))
    print(f"N={N:5d} realized C_q={Cq:+.4f}  analytic power={pred:.3f}  empirical rejection={emp:.3f} (3000 surveys)")
