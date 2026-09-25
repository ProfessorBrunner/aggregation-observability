"""Finite-start check (Sec. II and Sec. VI.C): one agent (Delta = 1, v = 5, gamma_phi = 0.02, eta = 0), started in
sigma_z = +1 at detuning x0 and read at eps = 2 Delta, for x0 = -320, -323, -330, -640 (units of Delta).
A point check of cutoff dependence, not a uniform error bound. Runtime: seconds."""
import numpy as np
from scipy.integrate import solve_ivp
D, v, g = 1.0, 5.0, 0.02
def rhs(t, y):
    rx, ry, rz, x = y
    return [-x*ry - 2*g*rx, x*rx - 2*D*rz - 2*g*ry, 2*D*ry, v]
def q_at(x0, xr=2.0, tol=1e-12):
    s = solve_ivp(rhs, (0, (xr - x0)/v), [0, 0, 1, x0], method='DOP853', rtol=tol, atol=tol)
    return (1 + s.y[2, -1])/2
qs = {x0: q_at(x0) for x0 in (-320, -323, -330, -640)}
for x0, q in qs.items(): print(f"start {x0:5d} Delta: q(eps = 2 Delta) = {q:.10f}")
print(f"largest change from the -320 Delta start: {max(abs(q - qs[-320]) for q in qs.values()):.2e}")
