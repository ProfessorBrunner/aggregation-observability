"""Direct maximum likelihood over mixtures of K pure-state qubit respondents, checked against the
exact Frank-Wolfe duality gap: Lambda_lower = Lambda - 2 gap <= Lambda_true <= Lambda."""
import numpy as np
from scipy.optimize import minimize
from region_exact import cells, GM, loglik, oracle, ll_QQ
def unit(t,p): return np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
def moments(x,K):
    A=x[:6*K].reshape(K,6); lw=x[6*K:]; w=np.exp(lw-lw.max()); w/=w.sum()
    M=[]
    for a in A:
        u=unit(a[0],a[1]); v=unit(a[2],a[3]); h=unit(a[4],a[5])
        q=(1+h@u)/2; r=(1+h@v)/2; c=(1+u@v)/2; M.append([q,r,c,q*c,r*c])
    return w@np.array(M)
def fit(n,K=6,restarts=40,seed=0):
    rng=np.random.default_rng(seed); best=(-np.inf,None)
    def negll(x):
        p=cells(moments(x,K))
        if np.any(p[n>0]<=1e-15): return 1e12
        return -loglik(n,p)
    for _ in range(restarts):
        x0=np.r_[rng.uniform(0,np.pi,6*K),rng.normal(size=K)]
        res=minimize(negll,x0,method='L-BFGS-B',options=dict(maxiter=5000))
        res=minimize(negll,res.x,method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-10,fatol=1e-12))
        if -res.fun>best[0]: best=(-res.fun,res.x)
    m=moments(best[1],K); p=cells(m); g=GM.T@(n/p); gap=float(g@(oracle(g)-m))
    lq=ll_QQ(n); L=max(0.0,2*(lq-best[0]))
    return L,max(0.0,L-2*gap),2*gap,m
