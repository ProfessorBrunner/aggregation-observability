"""Robustness to the column-coding question. Order effects implied by each reading, and Lambda for
datasets 1-71 under the swapped reading (stage 1: Frank-Wolfe; rows that do not converge to zero quickly are
listed; their values are in swapped_stage2a.csv, computed with boundary_fit.fit)."""
import numpy as np, pandas as pd
from common import D, cells_from_row, load
from region_exact import ll_QQ, ll_P
d=load()
d['oe_def']=np.maximum((d.V1-d.W1).abs(),(d.V2-d.W2).abs())/2
d['oe_swap']=np.maximum((d.V1-d.V2).abs(),(d.W1-d.W2).abs())/2
x=d[d.ds<=71]
print("largest order effect (points): definition median %.1f max %.1f | swapped median %.1f max %.1f"%(100*x.oe_def.median(),100*x.oe_def.max(),100*x.oe_swap.median(),100*x.oe_swap.max()))
rows=[]
for _,r in x.iterrows():
    n=r.n*cells_from_row(r,swapped=True); lp,m,gap,it=ll_P(n,tol=5e-6,maxit=1500)
    rows.append((int(r.ds),max(0,2*(ll_QQ(n)-lp)),2*gap))
o=pd.DataFrame(rows,columns=['ds','L','twogap'])
need=o[(o.twogap>=1e-4)|(o.L>=1e-6)].ds.tolist()
print("not at zero under the swapped reading:",need)
print("their order effects under the swapped reading (points): min %.1f"%(100*d[d.ds.isin(need)].oe_swap.min()))
