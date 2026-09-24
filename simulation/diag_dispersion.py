"""
diag_dispersion.py -- diagnostic, NO FREEZE.  Ensemble-mean undershoot below the
coherent asymptote as a function of population dispersion, at the cell-163
corner (Delta = 1, z = 0.2, kappa = 0), plus Null DB's trajectory-level match.

Additive only: `stage3_models.simulate` is imported unchanged and every number
below is read off the arrays it returns.  The one departure from the frozen
call is `n_sample`: the freeze samples 1200 output points, which puts only ~6
samples on the first post-crossing oscillation and therefore rounds off the
very quantity being measured here.  This script uses 24000 and reports the
lattice effect explicitly.

=======================================================================
DERIVATION, WRITTEN BEFORE THE RUN
=======================================================================

Exact reduction of the ensemble to a smoothing of one agent
-----------------------------------------------------------
With kappa = 0, u_dot = 0 and lambda = 0 the field is exactly linear,

    x(t) = x0 + v t,      v = Delta^2 / z,

so agent i sees eps_i(t) = xbar(t) + eta_i with xbar = x.  Every agent obeys the
same Bloch equations with the same |d eps/dt| = v and the same initial condition
r = (0,0,1) imposed at eps <= -K*Delta, deep in the diabatic region where that
state is stationary to O(Delta^2/eps^2).  Each agent's r_z is therefore ONE
universal function of its own detuning,

    r_z,i(t) = R(eps_i(t)) = R(xbar(t) + eta_i),                          (1)

and the ensemble mean is exactly a Gaussian smoothing of R in the detuning
variable:

    m(t) = (1/N) sum_i R(xbar + eta_i)  ->  (G_s * R)(xbar),   s = sd(eta).  (2)

In crossing-time language eta_i = -v t_i*, so the crossing times are Gaussian
with sd sigma_t = s/v, and smoothing in detuning by s is the same operation as
averaging the single-agent post-crossing oscillation over a Gaussian spread of
crossing times of width sigma_t.  (2) is the prediction; it has no free
constants, and it is exact in the N -> infinity limit.

Two limits of (2), which govern different parts of the curve
------------------------------------------------------------
Past the crossing the single-agent solution is a chirped oscillation about the
coherent asymptote,

    R(eps) = m_inf + A(eps) cos(Phi(eps) + phi0),   m_inf = 2 exp(-2 pi z) - 1,
    Phi(eps) = eps^2/(4v) + (2 Delta^2/v) ln(eps/Delta) + ...,
    Phi'(eps) = eps/(2v) + O(1/eps).                                      (3)

(a) PHASE SCRAMBLING.  Where A varies slowly over the kernel, expanding the
    phase to first order and integrating the Gaussian gives the damping factor

        D(eps) = exp(-0.5 (Phi'(eps) s)^2) = exp(-eps^2 s^2 / (8 v^2)),    (4)

    i.e. the oscillation survives while the spread of accumulated phase across
    the population, (eps/2) sigma_t, stays below a radian.  Applied at the
    location eps1 of the deepest single-agent excursion this predicts

        U_phase(s) = U0 * exp(-eps1^2 s^2 / (8 v^2)),  U0 = m_inf - min R.  (5)

(b) DIP CURVATURE.  (4) requires Phi'(eps) s >~ 1.  At the FIRST dip
    Phi'(eps1) = eps1/(2v) is small, so at every dispersion of interest here the
    first dip is NOT in the phase-scrambling regime.  What kills it instead is
    that the kernel reaches across a narrow dip into the pre-crossing plateau
    R ~ +1.  The leading behaviour is then set by the curvature of the dip,

        U(s) = U0 - 0.5 s^2 |R''(eps1)| + O(s^4),                          (6)

    which decays in s far faster than (5).  Prediction stated in advance: (5)
    will overestimate the surviving undershoot, (6) will track the small-s
    measurements, and the full convolution (2) is the only form valid across
    the whole range.  The crossover to (4) happens where the kernel width
    reaches the local oscillation period, s ~ 2v/eps.

Threshold on the frozen T_4
---------------------------
The frozen T_4 requires the ensemble mean to reach -0.5.  Since the mean
settles at m_inf > -0.5 for z < 0.221, T_4 at this corner is defined only while
the TRANSIENT undershoot is deep enough:

    U(s) >= U* = m_inf - (-0.5) = 0.5 - |m_inf|.                           (7)

The dispersion at which the predicted undershoot falls below the frozen -0.5
threshold is the root of U(s) = U*, reported from (2) and from (5) separately.

Finite-N floor
--------------
m is a mean over N = 100 agents, so the measured undershoot cannot follow (2) to
zero.  The sampling variance of the mean at detuning xbar is

    Var[m](xbar) = (1/N) [ (G_s * R^2)(xbar) - ((G_s * R)(xbar))^2 ],      (8)

and because the reported statistic is a MINIMUM over the trajectory it is
biased below the smooth curve by order the local sd.  (8) is evaluated at the
predicted argmin and reported alongside the across-draw scatter.
=======================================================================
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import stage3_models as M          # noqa: E402
import stage3_runs as RUN          # noqa: E402

# ---- declared design of this diagnostic -------------------------------
DELTA = 1.0
Z = 0.2
KAPPA = 0.0
K = 320.0
N_SAMPLE = 24000
SEEDS = RUN.SEEDS                                    # eight population draws
SD_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2,
           1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0)   # sd(eta)/Delta, from 0 up
DB_PARAMS = {"c_gamma": 0.25118864, "c_beta": 1.0}   # calibrated, null_db_results.npz

V = DELTA ** 2 / Z
M_INF = 2.0 * np.exp(-2.0 * np.pi * Z) - 1.0
U_STAR = M_INF + 0.5                                 # eq (7)


# ---- prediction machinery, eqs (2), (5), (6), (8) ---------------------
def single_agent_curve():
    """R(eps) and its grid, from the sd = 0 run (eta == 0 for every agent)."""
    r = M.simulate("focal", DELTA, Z, KAPPA, 0.0, SEEDS[0], K=K,
                   n_sample=N_SAMPLE, return_traj=True)
    return np.asarray(r["x"], float), np.asarray(r["m"], float), r


def _kernel(s, de, pad=8.0):
    h = int(np.ceil(pad * s / de))
    g = np.arange(-h, h + 1) * de
    k = np.exp(-0.5 * (g / s) ** 2)
    return k / k.sum(), h


def convolve_prediction(E, R, s, lo=-60.0, hi=120.0):
    """Eq (2) and eq (8): smoothed curve, its minimum, and the finite-N sd."""
    de = float(E[1] - E[0])
    w = (E >= lo) & (E <= hi)
    Ew, Rw = E[w], R[w]
    if s <= 0.0:
        i = int(np.argmin(Rw))
        return float(Rw[i]), float(Ew[i]), 0.0
    k, h = _kernel(s, de)
    c1 = np.convolve(Rw, k, mode="same")
    c2 = np.convolve(Rw ** 2, k, mode="same")
    inner = slice(h, Rw.size - h)
    i = int(np.argmin(c1[inner])) + h
    var = max(c2[i] - c1[i] ** 2, 0.0) / M.N_AGENT
    return float(c1[i]), float(Ew[i]), float(np.sqrt(var))


def phase_prediction(U0, eps1, s):
    """Eq (5): frozen-eps1 phase-scrambling form."""
    return U0 * np.exp(-(eps1 ** 2) * (s ** 2) / (8.0 * V ** 2))


def curvature_prediction(U0, curv, s):
    """Eq (6): dip-curvature form."""
    return U0 - 0.5 * (s ** 2) * curv


def dip_curvature(E, R, eps1, half=1.2):
    """|R''(eps1)| by a local quadratic fit over eps1 +- half."""
    w = np.abs(E - eps1) <= half
    c = np.polyfit(E[w] - eps1, R[w], 2)
    return float(abs(2.0 * c[0]))


def root_of(fun, lo, hi, tol=1e-6):
    """Bisection for U(s) = U*, assuming U decreasing on [lo, hi]."""
    a, b = lo, hi
    if (fun(a) - U_STAR) * (fun(b) - U_STAR) > 0:
        return np.nan
    while b - a > tol:
        mid = 0.5 * (a + b)
        if (fun(a) - U_STAR) * (fun(mid) - U_STAR) <= 0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


# ---- the run ----------------------------------------------------------
def main():
    E, R, r0 = single_agent_curve()
    j = int(np.argmin(R))
    U0, EPS1 = M_INF - R[j], float(E[j])
    CURV = dip_curvature(E, R, EPS1)

    s_conv = root_of(lambda s: M_INF - convolve_prediction(E, R, s)[0], 0.05, 4.0)
    s_phase = root_of(lambda s: phase_prediction(U0, EPS1, s), 0.05, 20.0)
    s_curv = root_of(lambda s: curvature_prediction(U0, CURV, s), 0.05, 4.0)

    print(f"m_inf = {M_INF:.6f}   U* = {U_STAR:.6f}   v = {V:g}")
    print(f"single agent: m_min = {R[j]:.6f}  U0 = {U0:.6f}  eps1 = {EPS1:.4f} "
          f" Phi'(eps1)= {EPS1/(2*V):.4f}  |R''| = {CURV:.6f}")
    print(f"threshold sd/Delta:  convolution {s_conv:.4f}   "
          f"curvature {s_curv:.4f}   phase-scrambling {s_phase:.4f}")

    rows, cols = [], ["sd", "draw", "f_mmin", "f_mend", "f_U", "f_eps_at_min",
                      "db_mmin", "db_mend", "rms_full", "rms_post", "maxabs",
                      "t_end", "n_t", "f_mmin_1200"]
    pred = []
    for s in SD_GRID:
        mc, ec, sdN = convolve_prediction(E, R, s)
        pred.append([s, M_INF - mc, ec, sdN, phase_prediction(U0, EPS1, s),
                     curvature_prediction(U0, CURV, s)])
        for jd, sd_seed in enumerate(SEEDS):
            rf = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                            n_sample=N_SAMPLE, return_traj=True)
            rd = M.simulate("db", DELTA, Z, KAPPA, s, sd_seed, params=DB_PARAMS,
                            K=K, n_sample=N_SAMPLE, return_traj=True)
            rl = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                            n_sample=1200, return_traj=True)
            t, mf, md = rf["t"], rf["m"], rd["m"]
            assert md.shape == mf.shape and np.allclose(rd["t"], t)
            post = rf["x"] >= 0.0
            d = mf - md
            rows.append([s, jd, float(mf.min()), float(mf[-1]),
                         float(M_INF - mf.min()),
                         float(rf["x"][int(np.argmin(mf))]),
                         float(md.min()), float(md[-1]),
                         float(np.sqrt(np.mean(d ** 2))),
                         float(np.sqrt(np.mean(d[post] ** 2))),
                         float(np.abs(d).max()), float(t[-1]), int(t.size),
                         float(rl["m"].min())])
        print(f"sd/Delta = {s:4.2f} done", flush=True)

    np.savez_compressed(
        "diag_dispersion.npz",
        rows=np.array(rows, float), cols=np.array(cols),
        pred=np.array(pred, float),
        pred_cols=np.array(["sd", "U_conv", "eps_at_min", "sd_finiteN",
                            "U_phase", "U_curv"]),
        E=E.astype(np.float32), R=R.astype(np.float32),
        scal=np.array([M_INF, U_STAR, V, U0, EPS1, CURV,
                       s_conv, s_curv, s_phase]),
        scal_names=np.array(["m_inf", "U_star", "v", "U0", "eps1", "curv",
                             "s_thresh_conv", "s_thresh_curv",
                             "s_thresh_phase"]))
    print("wrote diag_dispersion.npz")


if __name__ == "__main__":
    main()
