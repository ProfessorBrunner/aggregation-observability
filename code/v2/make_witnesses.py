"""Build one feasible-mixture witness per dataset (v1.2).

For each dataset, run Frank-Wolfe with away steps (exact-oracle version of region_exact.py) and record
the final iterate as an explicit mixture of qubit respondent types (q, r, c) with weights.
Saved per dataset: the adopted counts, the types and weights, the reconstructed cell probabilities,
l_QQ (closed form), l_cand (the mixture), and the bound 0 <= Lambda <= 2 (l_QQ - l_cand).
Output: ../../data/region_test_v2/witnesses/witness_XX.json. Runtime: about one minute."""
import numpy as np, json, os
from common import D, cells_from_row, schuman_counts, load
from region_exact import cells, GM, loglik, oracle, ll_QQ
OUT=D+'witnesses/'; os.makedirs(OUT,exist_ok=True)
def fw_witness(n, tol=1e-10, maxit=100000):
    atoms=[oracle(GM.T@(n/np.maximum(cells(np.array([.5,.5,.5,.25,.25])),1e-300)))]; w=[1.0]; x=atoms[0].copy()
    safe=np.array([0.5,0.5,0.5,0.25,0.25])
    if np.any(cells(x)[n>0]<=0): atoms.append(safe); w=[0.5,0.5]; x=0.5*atoms[0]+0.5*safe
    for it in range(maxit):
        p=cells(x); g=GM.T@(n/p); s_=oracle(g); gap=float(g@(s_-x))
        if gap<tol: break
        vals=[g@a for a in atoms]; j=int(np.argmin(vals)); dA=x-atoms[j]
        if gap>=float(g@dA) or len(atoms)==1: d=s_-x; gmax=1.0; away=False
        else: d=dA; gmax=w[j]/(1-w[j]) if w[j]<1 else 1e9; away=True
        lo,hi=0.0,gmax
        for _ in range(60):
            mid=(lo+hi)/2; pm=cells(x+mid*d)
            if np.any(pm[n>0]<=0): hi=mid; continue
            if (GM.T@(n/pm))@d>0: lo=mid
            else: hi=mid
        gam=lo
        if not away:
            w=[wi*(1-gam) for wi in w]
            for k,a in enumerate(atoms):
                if np.allclose(a,s_,atol=1e-13): w[k]+=gam; break
            else: atoms.append(s_); w.append(gam)
        else:
            w=[wi*(1+gam) for wi in w]; w[j]-=gam
            if w[j]<1e-14: atoms.pop(j); w.pop(j)
        x=sum(wi*a for wi,a in zip(w,atoms))
    keep=[k for k in range(len(w)) if w[k]>1e-15]
    A=np.array([atoms[k] for k in keep]); W=np.array([w[k] for k in keep]); W=W/W.sum()
    return A,W,gap
from scipy.optimize import linprog
def caratheodory(A,W):
    """Reduce a mixture to at most 6 types with the same moment vector (a vertex of the feasible weight polytope)."""
    m=W@A; Aeq=np.vstack([A.T,np.ones(len(W))]); beq=np.r_[m,1.0]
    res=linprog(np.zeros(len(W)),A_eq=Aeq,b_eq=beq,bounds=(0,None),method='highs-ds')
    w=res.x; keep=w>1e-14
    if not res.success or keep.sum()>6: return A,W
    w=w[keep]/w[keep].sum(); return A[keep],w
d=load(); d.loc[d.ds==73,'n']=457; summary=[]
for _,r in d.iterrows():
    ds=int(r.ds)
    n=schuman_counts() if ds==72 else r.n*cells_from_row(r)
    A,W,gap=fw_witness(n)
    A,W=caratheodory(A,W)
    m=W@A; p=cells(m); lq=ll_QQ(n); lc=loglik(n,p)
    rec=dict(dataset=ds, source=('Schuman et al. 1981, Table 2' if ds==72 else 'published moments, Dzhafarov et al. 2016'),
             counts_AB=n[:4].tolist(), counts_BA=n[4:].tolist(),
             types=[dict(q=float(a[0]),r=float(a[1]),c=float(a[2])) for a in A], weights=W.tolist(),
             cells_AB=p[:4].tolist(), cells_BA=p[4:].tolist(),
             logL_QQ=lq, logL_candidate=lc, Lambda_upper_bound=max(0.0,2*(lq-lc)), fw_gap_final=gap)
    json.dump(rec,open(OUT+f'witness_{ds:02d}.json','w'),indent=1)
    summary.append((ds,len(W),2*(lq-lc)))
import pandas as pd
s=pd.DataFrame(summary,columns=['ds','n_types','two_lQQ_minus_lcand']); s.to_csv(OUT+'witness_summary.csv',index=False)
print("witnesses written:",len(s)," max 2(lQQ-lcand) = %.2e"%s.two_lQQ_minus_lcand.max()," types per witness: %d-%d"%(s.n_types.min(),s.n_types.max()))
