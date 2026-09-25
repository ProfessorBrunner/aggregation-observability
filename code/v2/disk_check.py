"""Eq. (19): the disk bound. Checks that Eq. (14) equals (up to rounding where c is near 0 or 1) (q-r)^2/(1-c)+(q+r-1)^2/c <= 1, and reports the
radius of the 72 datasets and of the counterexample in the coordinates of Fig. 1. Runtime: seconds."""
import numpy as np
from common import load
rng=np.random.default_rng(3); q,r,c=rng.random((3,200000))
e14=((2*q-1)**2+(2*r-1)**2-2*(2*c-1)*(2*q-1)*(2*r-1))/(1-(2*c-1)**2)
print("max |Eq.(14) - disk form| over 2e5 random triples:", float(np.max(np.abs(e14-((q-r)**2/(1-c)+(q+r-1)**2/c)))))
d=load(); d=d[d.ds<=72].copy(); Q=(1+d.V1)/2; R=(1+d.V2)/2; a=(d.sAB+d.sBA)/2
i=d.index[d.ds==72][0]; Q[i]=246/293; R[i]=185/305; a[i]=((138+44)/293+(175+42)/305)/2
rad=np.hypot((Q-R)/np.sqrt(1-a),(Q+R-1)/np.sqrt(a))
print(f"largest radius over 72 datasets: {rad.max():.3f}; counterexample: {0.8/np.sqrt(0.108):.2f}")
