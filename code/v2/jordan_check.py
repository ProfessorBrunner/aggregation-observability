"""Theorem 4: single respondents with projectors of any rank in dimension 3-8 (Lueders update) give tables
inside conv S, i.e. Lambda = 0 in the test of Sec. V; the counterexample table is the control. Runtime: about a minute."""
import numpy as np
from region_exact import Lambda, cells
rng=np.random.default_rng(11)
def rproj(d,k):
    M=rng.normal(size=(d,k))+1j*rng.normal(size=(d,k)); Q,_=np.linalg.qr(M); return Q@Q.conj().T
def rstate(d):
    M=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); r=M@M.conj().T; return r/np.trace(r).real
def tables(P,Qp,rho):
    I=np.eye(P.shape[0]); Pc=I-P; Qc=I-Qp; seq=lambda X,Y: np.real(np.trace(Y@X@rho@X@Y))
    return np.array([seq(P,Qp),seq(P,Qc),seq(Pc,Qp),seq(Pc,Qc),seq(Qp,P),seq(Qp,Pc),seq(Qc,P),seq(Qc,Pc)])
for d,k in [(3,1),(4,2),(4,1),(5,2),(6,3),(6,2),(8,4)]:
    p=tables(rproj(d,k),rproj(d,rng.integers(1,d)),rstate(d)); L,g2,_,_=Lambda(1e5*p,tol=1e-9)
    print(f"dimension {d}, rank(P) {k}: Lambda = {L:.1e}")
Q,R,X,Y,Z,W=.90,.10,.99,.01,.01,.99; L,_,_,_=Lambda(1e5*cells(np.array([Q,R,Q*X+(1-Q)*Z,Q*X,R*Y])),tol=1e-6)
print(f"control (counterexample table): Lambda = {L:.0f}")
