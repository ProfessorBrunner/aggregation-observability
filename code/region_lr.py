import numpy as np, pandas as pd
rng=np.random.default_rng(424242)
# ---- atom sample: pure-state projective respondents ----
def atoms(M):
    u=rng.normal(size=(M,3)); u/=np.linalg.norm(u,axis=1,keepdims=True)
    v=rng.normal(size=(M,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
    h=rng.normal(size=(M,3)); h/=np.linalg.norm(h,axis=1,keepdims=True)
    q=(1+np.sum(h*u,1))/2; r=(1+np.sum(h*v,1))/2; c=(1+np.sum(u*v,1))/2
    return np.c_[q,r,c,q*c,r*c]
S=atoms(200000)
# cells as linear function of moment vector m=(Q,R,s,u=QX,w=RY)
# A->B: [a0b0, a0b1, a1b0, a1b1] = [u, Q-u, 1-Q-(s-u), s-u]
# B->A: [b0a0, b0a1, b1a0, b1a1] = [w, R-w, 1-R-(s-w), s-w]
def cells(m):
    Q,R,s,u,w=m
    return np.array([u,Q-u,1-Q-s+u,s-u]), np.array([w,R-w,1-R-s+w,s-w])
def cell_grad():  # d cells / d m, rows = 8 cells
    G=np.zeros((8,5))
    G[0]=[0,0,0,1,0]; G[1]=[1,0,0,-1,0]; G[2]=[-1,0,-1,1,0]; G[3]=[0,0,1,-1,0]
    G[4]=[0,0,0,0,1]; G[5]=[0,1,0,0,-1]; G[6]=[0,-1,-1,0,1]; G[7]=[0,0,1,0,-1]
    return G
Gm=cell_grad()
def loglik(counts,p):
    p=np.clip(p,1e-12,None); return float(np.sum(counts*np.log(p)))
def ll_QQ(nAB,nBA):
    # nAB order [a0b0,a0b1,a1b0,a1b1]; agreement cells idx 0,3
    tot1,tot2=nAB.sum(),nBA.sum()
    agree=nAB[0]+nAB[3]+nBA[0]+nBA[3]; s=agree/(tot1+tot2)
    def fit(n,tot):
        a=(n[0]+n[3])/tot; p=n/tot
        f=np.where([1,0,0,1],s/max(a,1e-12),(1-s)/max(1-a,1e-12)); return p*f
    return loglik(nAB,fit(nAB,tot1))+loglik(nBA,fit(nBA,tot2))
def ll_P(nAB,nBA,iters=400,return_w=False):
    counts=np.r_[nAB/nAB.sum(),nBA/nBA.sum()]*1.0
    n=np.r_[nAB,nBA]
    # start at atom maximizing likelihood-ish: nearest to empirical moment
    Q=nAB[:2].sum()/nAB.sum(); R=nBA[:2].sum()/nBA.sum()
    s=((nAB[0]+nAB[3])/nAB.sum()+(nBA[0]+nBA[3])/nBA.sum())/2
    m0=np.array([Q,R,s,nAB[0]/nAB.sum(),nBA[0]/nBA.sum()])
    j=int(np.argmin(np.linalg.norm(S-m0,axis=1))); x=S[j].copy(); w={j:1.0}
    for k in range(iters):
        p=np.r_[cells(x)]; p=np.concatenate(cells(x)); p=np.clip(p,1e-12,None)
        g=Gm.T@(n/p)                     # gradient of loglik wrt m
        j=int(np.argmax(S@g)); d=S[j]-x
        if g@d<=1e-9: break
        # line search on gamma in [0,1]
        lo,hi=0.0,1.0
        for _ in range(40):
            mid=(lo+hi)/2; pm=np.clip(np.concatenate(cells(x+mid*d)),1e-12,None)
            if (Gm.T@(n/pm))@d>0: lo=mid
            else: hi=mid
        gam=lo
        x=x+gam*d
        for a in w: w[a]*=(1-gam)
        w[j]=w.get(j,0)+gam
    val=loglik(n,np.concatenate(cells(x)))
    if return_w:
        idx=np.array([a for a in w if w[a]>1e-10]); ww=np.array([w[a] for a in idx]); return val,idx,ww/ww.sum()
    return val
def Lambda(nAB,nBA): return max(0.0,2*(ll_QQ(nAB,nBA)-ll_P(nAB,nBA)))
