"""
diag_definedness.py -- 256-draw definedness curve and DB monotonicity check.
Diagnostic, NO FREEZE.  Same corner as dispersion_memo.md v1:
Delta = 1, z = 0.2, kappa = 0, u_dot = 0, lambda = 0, K = 320, N = 100.

  python diag_definedness.py --shard i --nshards n

DECLARED DESIGN
---------------
Dispersion grid: the seventeen values of `diag_dispersion.SD_GRID` plus
1.25, 1.35, 1.45, 1.55 (twenty-one values), the four additions bracketing the
threshold measured at 8 draws (sd/Delta = 1.4314).

Draws: 256 per dispersion, seeds `range(900000, 900256)`.  These are NEW and
disjoint from the eight seeds 3141-3148 used in dispersion_memo.md v1, so the
256-draw result is an independent sample, not an extension of the old one.

Lattices: every draw is integrated twice, at 24000 output samples and at the
frozen 1200, because §5 of dispersion_memo.md v1 found the lattice moves the
threshold.  `simulate` is called twice rather than subsampled: linspace(0,T,1200)
is not a subset of linspace(0,T,24000), and re-deriving one from the other would
be a different code path from the one the freeze scores.

ITEM 3 -- DB monotonicity
-------------------------
Null DB integrates  ydot_i = -Gamma_i (y_i - y*_i)  with

    y*_i(t) = -tanh( c_beta * eps_i(t) / (2 Delta) ),   Gamma_i > 0.

The analytic monotonicity argument is: y_i starts at +1 above y*_i, y*_i is
nonincreasing because eps_i increases monotonically, so y_i can never cross it
from above, hence ydot_i < 0 for all t and y_i is monotone decreasing.  The
argument therefore RESTS on y_i(t) > y*_i(t) holding pointwise.  This script
checks that directly, per agent and per output sample, on every DB run in the
sweep: it records the minimum of (y_i - y*_i) over the whole (N, n_t) array and
counts samples violating it.  A single violation would break the argument.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import stage3_models as M          # noqa: E402
from diag_dispersion import (DELTA, Z, KAPPA, K, N_SAMPLE, SD_GRID, DB_PARAMS,
                             M_INF, U_STAR)   # noqa: E402

SD_GRID_256 = tuple(sorted(set(SD_GRID) | {1.25, 1.35, 1.45, 1.55}))
SEEDS_256 = tuple(range(900000, 900256))
N_SAMPLE_FROZEN = 1200

COLS = ["sd", "draw", "seed", "m_min", "m_end", "U", "defined",
        "m_min_1200", "U_1200", "defined_1200",
        "db_gap_min", "db_viol", "db_m_end", "db_ystar_min", "n_t",
        "db_gap_min_res", "db_viol_res", "db_n_res", "db_viol_min_abs_eps",
        "db_dy_max", "db_dy_pos_sum", "db_n_samples"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs("s3disp", exist_ok=True)

    jobs = [(s, jd, sd_seed) for s in SD_GRID_256
            for jd, sd_seed in enumerate(SEEDS_256)]
    mine = [j for n, j in enumerate(jobs) if n % a.nshards == a.shard]

    rows = []
    for s, jd, sd_seed in mine:
        rf = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                        n_sample=N_SAMPLE, return_traj=True)
        rl = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                        n_sample=N_SAMPLE_FROZEN, return_traj=True)
        rd = M.simulate("db", DELTA, Z, KAPPA, s, sd_seed, params=DB_PARAMS,
                        K=K, n_sample=N_SAMPLE, return_traj=True, return_Y=True)

        y = rd["Y"]                                   # (N, n_t), y = 2p - 1
        eps = rd["x_full"][None, :] + rd["eta"][:, None]
        ystar = -np.tanh(DB_PARAMS["c_beta"] * eps / (2.0 * DELTA))
        gap = y - ystar
        # Deep pre-crossing, tanh saturates and y* == +1.0 to double precision
        # while y is still exactly its initial +1.0, so the STRICT inequality
        # is decided by rounding, not by the dynamics.  Separate the two:
        # "resolvable" = y* distinguishable from +-1 in double precision.
        res = np.abs(ystar) < 1.0 - 1e-15
        gap_res = gap[res]
        viol_eps = eps[gap <= 0.0]
        dy = np.diff(y, axis=1)

        mm, ml = float(rf["m"].min()), float(rl["m"].min())
        rows.append([s, jd, sd_seed, mm, float(rf["m"][-1]), M_INF - mm,
                     float(mm <= -0.5), ml, M_INF - ml, float(ml <= -0.5),
                     float(gap.min()), float((gap <= 0.0).sum()),
                     float(rd["m"][-1]), float(ystar.min()), int(rf["t"].size),
                     float(gap_res.min()), float((gap_res <= 0.0).sum()),
                     float(gap_res.size),
                     float(np.abs(viol_eps).min()) if viol_eps.size else np.nan,
                     float(dy.max()), float(np.maximum(dy, 0.0).sum()),
                     float(gap.size)])
    np.savez_compressed(f"s3disp/def_{a.shard}.npz",
                        rows=np.array(rows, float), cols=np.array(COLS))
    print(f"shard {a.shard}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
