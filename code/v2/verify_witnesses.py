"""Independent check of the per-dataset witnesses (uses numpy only; no fitting code).

For each witness file it verifies:
  1. weights are nonnegative and sum to 1;
  2. every respondent type (q, r, c) is a valid two-dimensional projective respondent, i.e. satisfies
     Eq. (14) of the paper in the form (q-r)^2/(1-c) + (q+r-1)^2/c <= 1 (limits at c = 0, 1);
  3. the stored cell probabilities equal those implied by the mixture;
  4. l_QQ, recomputed here from the closed form, and l_cand, recomputed from the mixture, give
     2 (l_QQ - l_cand) below the tolerance, which proves Lambda = 0 up to that tolerance.
Usage: python verify_witnesses.py [tolerance]   (default 1e-9). Runtime: under a second."""
import json, glob, sys, numpy as np
tol=float(sys.argv[1]) if len(sys.argv)>1 else 1e-9
def valid(q,r,c,eps=1e-9):
    if not (-eps<=q<=1+eps and -eps<=r<=1+eps and -eps<=c<=1+eps): return False
    if c>=1-1e-12: return abs(q-r)<=1e-6
    if c<=1e-12:   return abs(q+r-1)<=1e-6
    return (q-r)**2/(1-c)+(q+r-1)**2/c<=1+eps
def ll(n,p):
    n=np.asarray(n,float); p=np.asarray(p,float); m=n>0; return float(np.sum(n[m]*np.log(p[m])))
def ll_QQ(nAB,nBA):
    nAB=np.asarray(nAB,float); nBA=np.asarray(nBA,float)
    a=(nAB[0]+nAB[3]+nBA[0]+nBA[3])/(nAB.sum()+nBA.sum())          # common agreement probability
    def fit(k):                                                       # agreement cells [0,3], disagreement [1,2]
        t=k.sum(); ag=(k[0]+k[3])/t; pe=k/t
        return pe*np.array([a/ag,(1-a)/(1-ag),(1-a)/(1-ag),a/ag])
    return ll(nAB,fit(nAB))+ll(nBA,fit(nBA))
bad=0; worst=0.0; files=sorted(glob.glob('../../data/region_test_v2/witnesses/witness_*.json'))
for f in files:
    w=json.load(open(f)); W=np.array(w['weights']); T=[(t['q'],t['r'],t['c']) for t in w['types']]
    ok_w=np.all(W>=-1e-15) and abs(W.sum()-1)<1e-12
    ok_t=all(valid(*t) for t in T)
    AB=sum(wi*np.array([q*c,q*(1-c),(1-q)*(1-c),(1-q)*c]) for wi,(q,r,c) in zip(W,T))
    BA=sum(wi*np.array([r*c,r*(1-c),(1-r)*(1-c),(1-r)*c]) for wi,(q,r,c) in zip(W,T))
    ok_c=np.allclose(AB,w['cells_AB'],atol=1e-12) and np.allclose(BA,w['cells_BA'],atol=1e-12)
    bound=2*(ll_QQ(w['counts_AB'],w['counts_BA'])-(ll(w['counts_AB'],AB)+ll(w['counts_BA'],BA)))
    worst=max(worst,bound); ok_b=bound<=tol
    if not (ok_w and ok_t and ok_c and ok_b):
        bad+=1; print(f"dataset {w['dataset']}: weights {ok_w} types {ok_t} cells {ok_c} bound {bound:.2e}")
print(f"{len(files)} witnesses checked; failures: {bad}; largest 2(l_QQ - l_cand) = {worst:.2e} (tolerance {tol:g})")
