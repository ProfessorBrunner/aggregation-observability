"""
diag_continuous.py -- crossing of m = -0.5 located by EVENT ROOT-FINDING on the
integrator's dense output, not by scanning output samples.  Diagnostic, NO FREEZE.

  python diag_continuous.py --shard i --nshards n

Same corner as definedness_memo.md v1: Delta = 1, z = 0.2, kappa = 0, u_dot = 0,
lambda = 0, K = 320, N = 100, gamma_phi = 0.02 Delta.  Same 21 dispersions
(diag_definedness.SD_GRID_256) and the same 256 seeds (900000-900255).

METHOD
------
`stage3_models.simulate` gained an opt-in `m_cross_level`, which appends a
NON-TERMINAL event g(t, y) = mean(y[2N:3N]) - level with direction = -1.  SciPy
locates each root with brentq on the dense interpolant of the step that brackets
it, so the crossing instant is resolved to solver tolerance and is independent of
`t_eval`.  Non-terminal events do not influence step selection, so the integrated
trajectory is the one the freeze integrates.  Two checks, both reported rather
than asserted: `m` with and without the event compares equal under
np.array_equal for the same (sd, seed); and the sampled `m_min` carried in every
row below is compared against the stored sample-scanned arrays in
`diag_definedness.npz` in the analysis step.

`n_sample = 24000` is retained ONLY to make that comparison possible; no reported
crossing quantity depends on it.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import stage3_models as M                                      # noqa: E402
from diag_dispersion import DELTA, Z, KAPPA, K, N_SAMPLE, M_INF  # noqa: E402
from diag_definedness import SD_GRID_256, SEEDS_256              # noqa: E402

LEVEL = -0.5
COLS = ["sd", "draw", "seed", "crossed", "n_cross", "t_cross_first",
        "m_at_first_root", "m_min_sampled", "m_end", "U", "t_end", "n_t",
        "solver_status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs("s3cont", exist_ok=True)

    jobs = [(s, jd, sd) for s in SD_GRID_256 for jd, sd in enumerate(SEEDS_256)]
    mine = [j for n, j in enumerate(jobs) if n % a.nshards == a.shard]

    rows = []
    for s, jd, sd_seed in mine:
        r = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                       n_sample=N_SAMPLE, return_traj=True,
                       m_cross_level=LEVEL)
        mm = float(r["m"].min())
        rows.append([s, jd, sd_seed, float(r["n_cross"] > 0), float(r["n_cross"]),
                     r["t_cross_first"],
                     float(r["m_at_cross"][0]) if r["n_cross"] else np.nan,
                     mm, float(r["m"][-1]), M_INF - mm, float(r["t"][-1]),
                     int(r["t"].size), float(r["solver_status"])])
    np.savez_compressed(f"s3cont/cont_{a.shard}.npz",
                        rows=np.array(rows, float), cols=np.array(COLS))
    print(f"shard {a.shard}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
