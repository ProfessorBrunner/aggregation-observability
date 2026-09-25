"""
smoke_test.py -- deterministic reproduction check for simulation_v12.

  python smoke_test.py

Runs four short integrations at the paper-1 diagnostics corner (Delta = 1,
z = 0.2, kappa = 0, u_dot = 0, lambda = 0, K = 320, N = 100, gamma_phi = 0.02
Delta) and checks stored numbers.  No grid runs; wall time ~15 s on one core.
Every expected value is in reference_values.json and was produced by the same
scripts shipped here.  Exit status 0 = all checks passed.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3_models as M
import diag_dispersion as DP

REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "reference_values.json")))
OK = []
def chk(name, got, want, tol, kind="abs"):
    d = abs(got - want) if kind == "abs" else abs(got - want) / max(abs(want), 1e-300)
    ok = d <= tol
    OK.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} got {got!r:26s} want {want!r:26s} d={d:.3e} tol={tol:g}")

print("1. single agent, sd(eta) = 0 (the reference curve R(eps))")
r0 = M.simulate("focal", 1.0, 0.2, 0.0, 0.0, 900000, K=320.0, n_sample=24000,
                return_traj=True)
E, R = np.asarray(r0["x"], float), np.asarray(r0["m"], float)
j = int(np.argmin(R))
chk("m_min", float(R[j]), REF["sa_m_min"], 1e-6)
chk("eps at the deepest dip", float(E[j]), REF["sa_eps1"], 1e-4)
chk("U0 = m_inf - min R", float(DP.M_INF - R[j]), REF["sa_U0"], 1e-6)
chk("|R''(eps1)| local quadratic fit", DP.dip_curvature(E, R, E[j]), REF["sa_curv"], 1e-5)

print("2. convolution threshold (the predictor of record)")
th = DP.root_of(lambda s: DP.M_INF - DP.convolve_prediction(E, R, s)[0], 0.05, 4.0)
chk("sd(eta)/Delta at U(s) = U*", th, REF["conv_thresh"], 1e-4)

print("3. phase rate Phi' = Omega/v, from the spacing of r_z minima")
w = (E > 0) & (E < 40)
Ew, Rw = E[w], R[w]
loc = (Rw[1:-1] < Rw[:-2]) & (Rw[1:-1] < Rw[2:])
em = Ew[1:-1][loc]
sp, mid = np.diff(em), 0.5 * (em[:-1] + em[1:])
ratio = sp / (2 * np.pi / (np.sqrt(mid ** 2 + 4.0) / 5.0))
chk("mean measured/predicted spacing", float(ratio.mean()), 0.9985, 2e-3)
print(f"        (the superseded rate eps/(2v) would give {float((sp / (2*np.pi/(mid/10.0))).mean()):.4f})")

print("4. focal at sd(eta)/Delta = 1.0, seed 900000, with the crossing event")
r1 = M.simulate("focal", 1.0, 0.2, 0.0, 1.0, 900000, K=320.0, n_sample=24000,
                return_traj=True, m_cross_level=-0.5)
chk("m_min", float(r1["m"].min()), REF["f10_m_min"], 1e-6)
chk("m_end", float(r1["m"][-1]), REF["f10_m_end"], 1e-6)
chk("first downward root of m = -0.5", float(r1["t_cross_first"]), REF["f10_tcross"], 1e-4)
chk("m at that root", float(r1["m_at_cross"][0]), -0.5, 1e-12)

print("5. Null DB monotonicity at its calibrated node")
rd = M.simulate("db", 1.0, 0.2, 0.0, 1.0, 900000,
                params={"c_gamma": 0.25118864, "c_beta": 1.0},
                K=320.0, n_sample=24000, return_traj=True, return_Y=True)
y = rd["Y"]
eps = rd["x_full"][None, :] + rd["eta"][:, None]
ystar = -np.tanh(eps / 2.0)
resolvable = np.abs(ystar) < 1.0 - 1e-15
chk("min (y - y*) where y* is resolvable", float((y - ystar)[resolvable].min()),
    REF["db_gap_res_min"], 1e-14)
chk("m_min equals m_end (monotone)", float(rd["m"].min() - rd["m"][-1]), 0.0, 1e-14)

print("6. pooled reciprocity identity G2 = Cov(c,q)/[qbar(1-qbar)] at x_read = 2 Delta")
rg = M.simulate("focal", 1.0, 0.2, 0.0, 1.1, 900000, K=320.0, n_sample=24000,
                return_traj=True, return_Y=True)
xf = rg["x_full"]
k = int(np.nonzero(xf >= 2.0)[0][0])
epsg = 2.0 + rg["eta"]
rx, rz = rg["r_xyz"][0, :, k], rg["r_xyz"][2, :, k]
den = np.sqrt(epsg ** 2 + 4.0)
c, q = 0.5 * (1 + epsg / den), 0.5 * (1 + rz)
pred = float(np.cov(c, q, bias=True)[0, 1] / (q.mean() * (1 - q.mean())))
Tyy, Tyn = (q * c).mean(), (q * (1 - c)).mean()
Tny, Tnn = ((1 - q) * (1 - c)).mean(), ((1 - q) * c).mean()
meas = float(Tyy / (Tyy + Tyn) + Tny / (Tny + Tnn) - 1.0)
chk("G2 from the covariance formula", pred, REF["g2_pred"], 1e-9)
chk("identity |predicted - pooled table|", abs(pred - meas), 0.0, 1e-13)
chk("readout sample x at or after x_read", float(xf[k]), REF["readout_x_at_sample"], 1e-6)

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
