"""Robustness of Lambda = 0 to the relative sizes of the two order samples.
For fixed proportion tables, the QQ-constrained fit depends on the relative weight w of the A->B order
only through the common agreement a* = w a_AB + (1-w) a_BA, and the fitted cells are affine in a*.
So if the fits at the two extremes a* = a_AB and a* = a_BA both lie in conv S, every relative weighting
does (convexity). This script tests both extremes for each of the 72 adopted datasets. About 2 min."""
import numpy as np, pandas as pd
from common import cells_from_row, schuman_counts, load
from region_exact import ll_QQ, ll_P
from boundary_fit import fit
def qq_fit(p,a):
    out=[]
    for k in (p[:4],p[4:]):
        k=k/k.sum(); ag=k[0]+k[3]
        out.append(k*np.array([a/ag,(1-a)/(1-ag),(1-a)/(1-ag),a/ag]))
    return np.r_[out[0],out[1]]
d=load(); rows=[]
for _,r in d[d.ds<=72].iterrows():
    p=schuman_counts() if r.ds==72 else cells_from_row(r)
    p=np.r_[p[:4]/p[:4].sum(),p[4:]/p[4:].sum()]
    aAB=p[0]+p[3]; aBA=p[4]+p[7]
    for tag,a in (('a_AB',aAB),('a_BA',aBA)):
        n=1e4*qq_fit(p,a)                       # the fitted table itself, as pseudo-counts
        lp,m,gap,it=ll_P(n,tol=1e-9,maxit=3000); L=max(0.0,2*(ll_QQ(n)-lp))
        if L>1e-8: L,_,_,_=fit(n,K=6,restarts=6)
        rows.append((int(r.ds),tag,L))
o=pd.DataFrame(rows,columns=['ds','endpoint','Lambda']); o.to_csv('../../data/region_test_v2/order_weight_check.csv',index=False)
print(f"{len(o)} endpoint fits; largest Lambda = {o.Lambda.max():.1e}; any outside the hull (Lambda > 1e-8): {int((o.Lambda>1e-8).sum())}")
