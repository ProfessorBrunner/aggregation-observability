"""Supplement S2: endpoint of the detailed-balance rival over an infinite sweep.
With eps = 2g tan(theta), Eq. (S3) becomes dy/dtheta = -4z [y + tanh(c_beta c'_gamma tan theta)],
so the endpoint depends on (c_beta, c'_gamma) only through their product. Runtime: seconds."""
import numpy as np
from scipy.integrate import solve_ivp
def yinf(z,cb,cg):
    f=lambda th,y: -4*z*(y+np.tanh(cb*cg*np.tan(th)))
    return solve_ivp(f,[-np.pi/2+1e-9,np.pi/2-1e-9],[1.0],rtol=1e-11,atol=1e-13).y[0,-1]
for cg in [0.1,0.251,1.0]: print(f"z=0.2, c_beta=1, c'_gamma={cg}: y_inf = {yinf(0.2,1,cg):.4f}")
print("same product 0.5:", round(yinf(0.2,1,0.5),6), round(yinf(0.2,2,0.25),6))
