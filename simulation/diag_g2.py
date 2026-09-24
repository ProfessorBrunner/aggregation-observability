"""
diag_g2.py -- R71 option 1: pooled reciprocity difference G2 versus dispersion.
Diagnostic, NO FREEZE.  Corner: Delta = 1, z = 0.2, kappa = 0, K = 320, N = 100.

  python diag_g2.py --shard i --nshards n

RULED DEFINITIONS (R71, option 1)
---------------------------------
First frame F  : the FOMO/loss basis, i.e. the sigma_z eigenbasis of
                 H = (1/2) eps sigma_z + Delta sigma_x.
Second frame L : the instantaneous Hamiltonian eigenbasis at the readout field.
Frame overlap  : c_i = |<F_y|L_+>|^2 = (1/2)(1 + eps_i/sqrt(eps_i^2 + 4 Delta^2)).
Propensity     : q_i = p(F = y) = (1/2)(1 + r_{z,i}) at the readout.
Renamed        : the pooled reciprocity difference is G2, not T2 -- the freeze
                 already uses T2 for the stationary frequency m(t_end).

WHAT "AT THE READOUT FIELD" HAS TO MEAN
--------------------------------------
Two readings, and only one has content.  If every agent is read at a common
FIELD VALUE eps, then c_i = c(eps) is the same for all of them: eta_i shifts
WHEN an agent reaches a given detuning, not WHICH detuning it is read at, so
Cov(c, q) collapses to output-lattice granularity (measured: 3.83e-07 at
eps = +10 Delta, 3.59e-13 at the exit radius) and G2 measures the lattice.
The ruling asks for "c_i ... from its eta_i", which requires the other reading:
a common readout TIME, at which the shared field is x_read and

    eps_i = x_read + eta_i,

so c_i disperses with sd(eta) exactly as the ruling intends.  That reading is
used throughout.  x_read is swept over a declared ladder; the two reported
primaries are x_read = 10 Delta (R39's off-resonance threshold) and
x_read = K Delta = 320 Delta (the frozen exit radius).  Post-crossing side.

THE IDENTITY BEING TESTED
-------------------------
For sharp rank-1 frames the arm-1 table of a SINGLE agent is

    p(F_y, L_y) = q_i c_i        p(F_y, L_n) = q_i (1 - c_i)
    p(F_n, L_y) = (1-q_i)(1-c_i) p(F_n, L_n) = (1-q_i) c_i

so p(L=y|F=y) + p(L=y|F=n) = c_i + (1 - c_i) = 1 for every agent: reciprocity
holds exactly, agent by agent.  Pooling breaks it.  With <.> the population
mean, <qc> = qbar cbar + Cov(c,q) and <(1-q)(1-c)> = (1-qbar)(1-cbar) + Cov, so

    G2 = P(L=y|F=y) + P(L=y|F=n) - 1
       = <qc>/qbar + <(1-q)(1-c)>/(1-qbar) - 1
       = Cov(c, q) / [ qbar (1 - qbar) ].                                 (G2)

Predicted side  : the right-hand side, from the population's (c_i, q_i).
Measured side   : the left-hand side, from the pooled 2x2 frame table built by
                  explicit per-agent state update.
They are the same number if (G2) is right, so this is a check of an identity,
not a statistical comparison.  A population of sharp-frame agents each
satisfying reciprocity exactly therefore shows an apparent POOLED reciprocity
violation of size Cov(c,q)/[qbar(1-qbar)], carried entirely by heterogeneity.

Also recorded: the arm-2 pooled agreement (L first), which the same algebra
makes equal to cbar in both arms, so the pooled QQ-style agreement difference
should vanish while G2 does not; and the fraction of agents satisfying R39's
|eps| >= 10 Delta at the readout, since the off-resonance attribution needs it.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import stage3_models as M              # noqa: E402
import sec4_order_restriction as OR    # noqa: E402
from diag_dispersion import DELTA, Z, KAPPA, K, N_SAMPLE   # noqa: E402
from diag_definedness import SD_GRID_256                   # noqa: E402

SEEDS_G2 = tuple(range(900000, 900064))       # first 64 of the declared block
X_READ = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 320.0)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1, -1]).astype(complex)
I2 = np.eye(2, dtype=complex)

COLS = ["sd", "draw", "seed", "x_read", "cbar", "qbar", "cov_cq", "var_c",
        "var_q", "G2_pred", "G2_meas", "G2_absdiff", "wbar",
        "agree_arm1", "agree_arm2", "qq_pooled", "frac_offres",
        "eps_min", "eps_max", "c_min", "c_max", "q_min", "q_max",
        "G2_state_update", "n_state_update"]


def pooled_tables(c, q, w):
    """Arm-1 and arm-2 pooled 2x2 frame tables, closed form for sharp frames."""
    a1 = np.array([[(q * c).mean(), (q * (1 - c)).mean()],
                   [((1 - q) * (1 - c)).mean(), ((1 - q) * c).mean()]])
    a2 = np.array([[(w * c).mean(), (w * (1 - c)).mean()],
                   [((1 - w) * (1 - c)).mean(), ((1 - w) * c).mean()]])
    return a1, a2


def g2_from_table(a1):
    return a1[0, 0] / a1[0].sum() + a1[1, 0] / a1[1].sum() - 1.0


def g2_by_state_update(eps, rx, ry, rz, n_max=12):
    """Same quantity from explicit Luders updates on the first n_max agents."""
    n = min(n_max, eps.size)
    T = np.zeros((2, 2))
    for i in range(n):
        rho = 0.5 * (I2 + rx[i] * SX + ry[i] * SY + rz[i] * SZ)
        H = 0.5 * eps[i] * SZ + DELTA * SX
        _, Lb = np.linalg.eigh(H)
        Lb = Lb[:, ::-1]
        a = OR.protocol(rho, OR.kraus_set(I2, 1, 1.0), OR.kraus_set(Lb, 1, 1.0),
                        0.5 * I2)
        for u in (0, 1):
            for vv in (0, 1):
                T[u, vv] += a["joint"][(u, vv)]
    T /= n
    return g2_from_table(T), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs("s3g2", exist_ok=True)

    jobs = [(s, jd, sd) for s in SD_GRID_256 for jd, sd in enumerate(SEEDS_G2)]
    mine = [j for n, j in enumerate(jobs) if n % a.nshards == a.shard]

    rows = []
    for s, jd, sd_seed in mine:
        r = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                       n_sample=N_SAMPLE, return_traj=True, return_Y=True)
        rb, eta, xf = r["r_xyz"], r["eta"], r["x_full"]
        for xr in X_READ:
            k = np.nonzero(xf >= xr)[0]          # common readout TIME
            if not k.size:
                continue
            k = int(k[0])
            eps = xr + eta                        # per-agent detuning, eq above
            rx, ry, rz = rb[0, :, k], rb[1, :, k], rb[2, :, k]
            den = np.sqrt(eps ** 2 + 4.0 * DELTA ** 2)
            c = 0.5 * (1.0 + eps / den)
            q = 0.5 * (1.0 + rz)
            w = 0.5 * (1.0 + rx * (2.0 * DELTA / den) + rz * (eps / den))
            cov = float(np.cov(c, q, bias=True)[0, 1])
            qb = float(q.mean())
            gp = cov / (qb * (1.0 - qb))
            a1, a2 = pooled_tables(c, q, w)
            gm = g2_from_table(a1)
            gs, ns = g2_by_state_update(eps, rx, ry, rz)
            rows.append([s, jd, sd_seed, xr, float(c.mean()), qb, cov,
                         float(c.var()), float(q.var()), gp, gm, abs(gp - gm),
                         float(w.mean()), float(a1[0, 0] + a1[1, 1]),
                         float(a2[0, 0] + a2[1, 1]),
                         float((a1[0, 0] + a1[1, 1]) - (a2[0, 0] + a2[1, 1])),
                         float((np.abs(eps) >= 10.0 * DELTA).mean()),
                         float(eps.min()), float(eps.max()), float(c.min()),
                         float(c.max()), float(q.min()), float(q.max()),
                         gs, float(ns)])
    np.savez_compressed(f"s3g2/g2_{a.shard}.npz",
                        rows=np.array(rows, float), cols=np.array(COLS))
    print(f"shard {a.shard}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
