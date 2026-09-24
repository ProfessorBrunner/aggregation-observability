"""
diag_dbmatch.py -- second pass of the dispersion diagnostic, NO FREEZE.

`diag_dispersion.py` reports rms(m_focal - m_db), which is flat in dispersion.
That statistic cannot say whether the two ensemble means agree in SHAPE,
because a constant level offset enters it in quadrature and dominates.  Here
the difference is decomposed on the same runs:

    d(t) = m_focal(t) - m_db(t)
    rms^2 = mean(d)^2 + rms(d - mean(d))^2
            \_ offset _/   \_____ shape _____/

Reported on the full recorded window and on the post-crossing window
(xbar >= 0), together with the largest excursion and where it occurs.  Same
cell, same seeds, same integrator settings as the first pass.
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import stage3_models as M          # noqa: E402
from diag_dispersion import (DELTA, Z, KAPPA, K, N_SAMPLE, SEEDS, SD_GRID,
                             DB_PARAMS, M_INF)   # noqa: E402


def main():
    rows = []
    cols = ["sd", "draw", "off_full", "shape_full", "rms_full",
            "off_post", "shape_post", "rms_post", "maxabs", "eps_at_maxabs",
            "corr_post", "f_mmin", "db_mmin", "db_mono_viol"]
    for s in SD_GRID:
        for jd, sd_seed in enumerate(SEEDS):
            rf = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K,
                            n_sample=N_SAMPLE, return_traj=True)
            rd = M.simulate("db", DELTA, Z, KAPPA, s, sd_seed,
                            params=DB_PARAMS, K=K, n_sample=N_SAMPLE,
                            return_traj=True)
            mf, md, xb = rf["m"], rd["m"], rf["x"]
            assert np.allclose(rf["t"], rd["t"])
            d = mf - md
            post = xb >= 0.0
            k = int(np.argmax(np.abs(d)))
            cp = float(np.corrcoef(mf[post], md[post])[0, 1])
            rows.append([s, jd,
                         float(d.mean()), float(d.std()), float(np.sqrt((d ** 2).mean())),
                         float(d[post].mean()), float(d[post].std()),
                         float(np.sqrt((d[post] ** 2).mean())),
                         float(np.abs(d).max()), float(xb[k]), cp,
                         float(mf.min()), float(md.min()),
                         float(np.maximum(np.diff(md), 0.0).sum())])
        print(f"sd/Delta = {s:4.2f} done", flush=True)

    np.savez_compressed("diag_dbmatch.npz", rows=np.array(rows, float),
                        cols=np.array(cols), m_inf=np.array([M_INF]))
    print("wrote diag_dbmatch.npz")


if __name__ == "__main__":
    main()
