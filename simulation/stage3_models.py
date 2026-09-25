"""
stage3_models.py -- the five models of the signed freeze, one interface.

Every model is integrated under the SAME forcing, the same eta_i draw, the same
exit convention, and the same observables, so nothing in the comparison depends
on per-model bookkeeping.  Frozen specification: null_family_freeze_v4.md.

State layout
------------
Focal : y = [r_x(N), r_y(N), r_z(N), x]      (3N+1)
Nulls : y = [s(N), x]                        (N+1)   s = y_i, or p_i for Null S

Conventions carried from the freeze
-----------------------------------
* u_dot = 0, lambda = 0 in every cell (section 0.2)
* Delta is a design axis; every model parameter is a dimensionless constant
  times a power of Delta (section 0.3)
* c_gamma = 0.02 declared, so the focal model has ZERO free parameters
* exit radius E = K*Delta with a 2% margin on the terminal event, so the
  last-exiting agent gets recorded samples beyond E (carried from task (c))
"""

import numpy as np
from scipy.integrate import solve_ivp

K_TRUNC = 320.0
C_GAMMA = 0.02
N_AGENT = 100
N_SAMPLE = 1200
HORIZON_MULT = 2.5

MODELS = ("focal", "lag", "rtip", "two_state", "ising", "db")

# free dimensionless constants, in the order used by the calibrator
FREE = {"focal": (),
        "lag": ("c_w", "c_tau"),
        "rtip": ("c_w", "c_a", "c"),
        "two_state": ("c_k", "c_beta"),
        "ising": ("c_J", "c_T", "c_tau")}


def make_eta(N, sd, seed):
    return np.random.default_rng(int(seed)).normal(0.0, sd, N)


def tau_c(z, D):
    """Crossing window, closed form for u_dot = 0: v = D^2/z."""
    return max(z, np.sqrt(z)) / D


def y_max_rtip(c):
    """Largest real root of y^3 - y = -c.  Analytic, not fitted."""
    r = np.roots([1.0, 0.0, -1.0, float(c)])
    r = r[np.abs(r.imag) < 1e-12].real
    return float(np.max(np.abs(r)))


def _sig(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


# ----------------------------------------------------------------------
# right-hand sides
# ----------------------------------------------------------------------
def _rhs_focal(t, y, eta, D, kap, gphi, N):
    r = y[:3 * N].reshape(3, N)
    x = y[3 * N]
    e = x + eta
    out = np.empty_like(y)
    o = out[:3 * N].reshape(3, N)
    o[0] = -e * r[1] - 2.0 * gphi * r[0]
    o[1] = e * r[0] - 2.0 * D * r[2] - 2.0 * gphi * r[1]
    o[2] = 2.0 * D * r[1]
    out[3 * N] = _u0 + kap * r[2].mean()
    return out


def _rhs_lag(t, y, eta, D, kap, p, N):
    s = y[:N]
    e = y[N] + eta
    out = np.empty_like(y)
    out[:N] = (-np.tanh(e / (2.0 * p["c_w"] * D)) - s) * D / p["c_tau"]
    out[N] = _u0 + kap * s.mean()
    return out


def _rhs_rtip(t, y, eta, D, kap, p, N):
    s = y[:N]
    e = y[N] + eta
    out = np.empty_like(y)
    out[:N] = p["c_a"] * D * (s - s ** 3 - p["c"] * np.tanh(e / (p["c_w"] * D)))
    out[N] = _u0 + kap * (s / p["_ymax"]).mean()
    return out


def _rhs_two_state(t, y, eta, D, kap, p, N):
    pr = y[:N]
    e = y[N] + eta
    kf = p["c_k"] * D * _sig(+p["c_beta"] * e / D)
    kb = p["c_k"] * D * _sig(-p["c_beta"] * e / D)
    out = np.empty_like(y)
    out[:N] = -kf * pr + kb * (1.0 - pr)
    out[N] = _u0 + kap * (2.0 * pr - 1.0).mean()
    return out


def _rhs_db(t, y, eta, D, kap, p, N):
    """Null DB -- detailed-balance two-state rate process.

    Signed addendum `freeze_addendum_null_db.md`, eqs (DB.1)-(DB.3), built on
    the manuscript's `eq:classicalrate`.  Total rate is the Lorentzian obtained
    by adiabatic elimination; the split is biased by detailed balance:

        Gamma  = 8 g D^2 / (4 g^2 + eps^2),      g = c_gamma * D
        k_FL/k_LF = exp(c_beta eps / D),  k_FL + k_LF = Gamma

    which solves exactly to relaxation toward the detailed-balance equilibrium

        p_eq = sigmoid(-c_beta eps / D),   pdot = -Gamma (p - p_eq)

    i.e. ydot = -Gamma [y + tanh(c_beta eps / 2D)] for y = 2p - 1.  Both
    intensities are bounded by Gamma <= 2D/c_gamma, so the unbounded-Arrhenius
    stall that forced the Null-S respecification cannot recur here.
    """
    pr = y[:N]
    e = y[N] + eta
    g = p["c_gamma"] * D
    G = 8.0 * g * D ** 2 / (4.0 * g ** 2 + e ** 2)
    out = np.empty_like(y)
    out[:N] = -G * (pr - _sig(-p["c_beta"] * e / D))
    out[N] = _u0 + kap * (2.0 * pr - 1.0).mean()
    return out


def _rhs_ising(t, y, eta, D, kap, p, N):
    s = y[:N]
    e = y[N] + eta
    m = s.mean()
    out = np.empty_like(y)
    tgt = np.tanh((p["c_J"] * D * m - e) / (p["c_T"] * D))
    out[:N] = -(s - tgt) * D / p["c_tau"]
    out[N] = _u0 + kap * m
    return out


_u0 = 1.0   # rebound per cell in simulate()


def simulate(model, D, z, kap, sd, seed, params=None, N=N_AGENT,
             K=K_TRUNC, n_sample=N_SAMPLE, rtol=1e-7, atol=1e-9,
             method="DOP853", return_traj=False, return_Y=False,
             m_cross_level=None):
    """One cell, one population draw.  Returns the observable dict."""
    global _u0
    eta = make_eta(N, sd, seed)
    v0 = D ** 2 / z
    _u0 = v0
    E = K * D
    E_stop = 1.02 * E
    x0 = -E - eta.max()

    p = dict(params or {})
    if model == "rtip":
        p["_ymax"] = y_max_rtip(p["c"])

    if model == "focal":
        y0 = np.concatenate([np.zeros(N), np.zeros(N), np.ones(N), [x0]])
        rhs, ix = _rhs_focal, 3 * N
        args = (eta, D, kap, C_GAMMA * D, N)
    else:
        s0 = np.ones(N) * (p["_ymax"] if model == "rtip" else 1.0)
        y0 = np.concatenate([s0, [x0]])
        rhs = {"lag": _rhs_lag, "rtip": _rhs_rtip,
               "two_state": _rhs_two_state, "ising": _rhs_ising,
               "db": _rhs_db}[model]
        ix = N
        args = (eta, D, kap, p, N)

    span = (E - eta.min()) + (E + eta.max())
    T = HORIZON_MULT * span / v0

    def _all_out(t, yy, *_):
        return (yy[ix] + eta.min()) - E_stop
    _all_out.terminal = True
    _all_out.direction = 1.0

    # Dispersion diagnostics (2026-09-25).  ADDITIVE ONLY, opt-in, default off.
    # A NON-TERMINAL event on the ensemble mean, so the crossing instant is
    # located by brentq on the integrator's dense interpolant rather than by
    # scanning t_eval.  Non-terminal events do not influence step selection, so
    # the trajectory is the one the freeze integrates, bit for bit; only the
    # detection differs.  Verified directly: for the same (sd, seed), `m` with
    # and without `m_cross_level` compares equal under np.array_equal.  The
    # callers additionally carry the sampled `m_min` so the same trajectory can
    # be checked against previously stored sample-scanned runs; that check is
    # performed in the analysis step, not here.
    ev = [_all_out]
    if m_cross_level is not None:
        if model != "focal":
            raise ValueError("m_cross_level is implemented for the focal model only")
        lev = float(m_cross_level)

        def _m_cross(t, yy, *_):
            return yy[2 * N:3 * N].mean() - lev
        _m_cross.terminal = False
        _m_cross.direction = -1.0          # downward crossings only
        ev.append(_m_cross)

    ts = np.linspace(0.0, T, n_sample)
    sol = solve_ivp(rhs, (0.0, T), y0, args=args, t_eval=ts, rtol=rtol,
                    atol=atol, method=method, events=ev)
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    x = sol.y[ix]
    if model == "focal":
        Y = sol.y[2 * N:3 * N]
    elif model in ("two_state", "db"):
        Y = 2.0 * sol.y[:N] - 1.0
    elif model == "rtip":
        Y = sol.y[:N] / p["_ymax"]
    else:
        Y = sol.y[:N]

    out = _observables(t, x, Y, eta, D, z, kap, sd, E, v0, model)
    if m_cross_level is not None:
        # roots of m(t) - level located by the solver, not by sampling
        tc = sol.t_events[1]
        out["t_cross"] = np.asarray(tc, float)
        out["n_cross"] = int(tc.size)
        out["t_cross_first"] = float(tc[0]) if tc.size else np.nan
        out["m_at_cross"] = (np.asarray([yy[2 * eta.size:3 * eta.size].mean()
                                        for yy in sol.y_events[1]], float)
                             if tc.size else np.zeros(0))
        out["solver_status"] = int(sol.status)
    if return_Y:
        # Dispersion diagnostics (2026-09-24).  ADDITIVE ONLY, opt-in, default
        # off: exposes the per-agent array `_observables` already received, so
        # a per-agent check cannot drift from the frozen ensemble statistics.
        # Nothing above this line is recomputed or changed.
        out["Y"] = Y
        out["eta"] = eta
        out["x_full"] = x
        out["t_full"] = t
        if model == "focal":
            # Full Bloch vector per agent, (3, N, n_t) = (r_x, r_y, r_z).
            # Needed for frame projections along axes other than sigma_z;
            # sol.y is the integrator's own array, so no recomputation.
            out["r_xyz"] = sol.y[:3 * N].reshape(3, N, t.size)
    if return_traj:
        # Referee diagnostics (2026-09-23).  ADDITIVE ONLY: nothing the freeze
        # scores is recomputed or changed here.  The arrays returned are the
        # very ones `_observables` read T4 off, so a diagnostic recomputed from
        # them cannot drift from the frozen statistic.
        out["t"] = t
        out["m"] = Y.mean(axis=0)
        out["x"] = x
        out["Y_last"] = Y[:, -1]
        out["dt_out"] = float(t[1] - t[0]) if t.size > 1 else np.nan
        out["n_out"] = int(t.size)
        out["tau_c"] = float(tau_c(z, D))
    return out


def _observables(t, x, Y, eta, D, z, kap, sd, E, v0, model):
    """Everything the freeze scores, computed once per cell-draw."""
    N = eta.size
    eps = x[None, :] + eta[:, None]              # (N, n_t)
    m = Y.mean(axis=0)
    xd = np.gradient(x, t)

    # per-agent exit read and crossing diagnostics
    P = np.full(N, np.nan)
    zm = np.full(N, np.nan)
    status = []
    for i in range(N):
        ei = eps[i]
        sc = np.nonzero(np.diff(np.sign(ei)) != 0)[0]
        after = np.nonzero((np.abs(ei) >= E) & (t > (t[sc[0]] if sc.size else -1)))[0]
        if sc.size == 0:
            status.append("never_crossed")
            continue
        if sc.size > 1:
            status.append("multi_crossing")
            continue
        if after.size == 0:
            status.append("no_exit")
            continue
        k = int(sc[0])
        de = ei[k + 1] - ei[k]
        fr = 0.0 if abs(de) < 1e-300 else float(np.clip(-ei[k] / de, 0.0, 1.0))
        vv = xd[k] + fr * (xd[k + 1] - xd[k])
        zm[i] = D ** 2 / abs(vv) if abs(vv) > 1e-300 else np.nan
        P[i] = 0.5 * (1.0 + Y[i, int(after[0])])
        status.append("clean")

    clean = np.array([s == "clean" for s in status])

    # T4 exit morphology as FROZEN: m from +0.5 to -0.5, normalised by tau_c.
    T4 = np.nan
    hi = np.nonzero(m <= 0.5)[0]
    lo = np.nonzero(m <= -0.5)[0]
    if hi.size and lo.size:
        T4 = float((t[lo[0]] - t[hi[0]]) / tau_c(z, D))

    # Scale-free variant: 75%-to-25% fall time of the REALISED excursion.
    # The frozen T4's absolute thresholds are unreachable wherever the ensemble
    # mean never falls to -0.5, which is structural for the focal model at
    # small z (m_end = 2*exp(-2 pi z) - 1 > -0.5 for z < 0.221).  Reported as
    # evidence for the morphology ruling; it does NOT replace T4 unless ruled.
    T4n = np.nan
    m0, m1 = float(m[0]), float(m.min())
    if m0 - m1 > 0.2:
        a75, a25 = m0 - 0.25 * (m0 - m1), m0 - 0.75 * (m0 - m1)
        i75 = np.nonzero(m <= a75)[0]
        i25 = np.nonzero(m <= a25)[0]
        if i75.size and i25.size:
            T4n = float((t[i25[0]] - t[i75[0]]) / tau_c(z, D))

    # cross-sectional dispersion and susceptibility
    var_cs = Y.var(axis=0)
    md = np.gradient(m, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.abs(md / xd)

    return dict(
        model=model, D=D, z=z, kappa=kap, sd=sd,
        P=np.clip(P, 0.0, 1.0), z_meas=zm, clean=clean,
        frac_clean=float(clean.mean()),
        n_clean=int(clean.sum()),
        n_never=int(sum(s == "never_crossed" for s in status)),
        n_multi=int(sum(s == "multi_crossing" for s in status)),
        n_noexit=int(sum(s == "no_exit" for s in status)),
        P_cell=float(np.nanmedian(P)) if clean.any() else np.nan,
        z_meas_cell=float(np.nanmedian(zm)) if clean.any() else np.nan,
        m_end=float(m[-1]), m_start=float(m[0]),
        T4=T4, T4n=T4n,
        var_cs_max=float(np.nanmax(var_cs)),
        chi_max=float(np.nanmax(chi[np.isfinite(chi)])) if np.isfinite(chi).any() else np.nan,
        m_mean=float(m.mean()), m_var=float(m.var()),
        m_ac1=_lag1(m),
        t_end=float(t[-1]),
    )


def _lag1(a):
    """Lag-1 autocorrelation of the DETRENDED series.

    Reported for completeness only.  Under the frozen noise specification the
    dynamics are deterministic (gamma_r = 0, no annealed noise), so there are no
    fluctuations to autocorrelate and this number reflects the smooth trend, not
    an early-warning signal.  See the Stage 3 memo.
    """
    a = np.asarray(a, float)
    d = a - np.convolve(a, np.ones(11) / 11.0, mode="same")
    d = d[5:-5]
    if d.size < 10 or d.std() == 0:
        return np.nan
    return float(np.corrcoef(d[:-1], d[1:])[0, 1])
