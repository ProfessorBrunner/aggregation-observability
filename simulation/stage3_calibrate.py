"""
stage3_calibrate.py -- calibrate one null on the 75 frozen training cells.

  python stage3_calibrate.py --model lag --shard i --nshards n

CALIBRATION TARGET.  There is no empirical data in this exercise and none is
permitted, so the target values for T1-T4 are the FOCAL MODEL's own output on
the training cells.  That is what "can a parsimonious classical process
reproduce the surface" means operationally: each null is given 2-3 constants
and asked to match a parameter-free focal surface on the training region, then
tested on held-out cells.  Declared here because it is a design decision, not a
frozen instruction.

Objective: unweighted sum of squared differences in the four standardised
targets, each standardised by the across-training-cell standard deviation of
the focal value, so no target dominates by units.

  T1  m_end on training cells with z >= z_train = 1.0        (30 cells)
  T2  m_end on training cells with kappa = 0                 (30 cells)
  T3  (m_mean, m_var) on all 75 training cells
  T4  frozen T4 where finite for BOTH focal and null

Coarse log-spaced grid, sharded over grid points; 2 population draws per cell
at the grid stage (seeds 3141, 3142).  The winner is then re-run on the full
8-draw grid by stage3_runs.py.
"""

import argparse
import itertools
import json
import os

import numpy as np

import stage3_models as M
from stage3_runs import cells, is_train, SEEDS

CAL_SEEDS = SEEDS[:2]

GRIDS = {
    "lag":       {"c_w": np.geomspace(0.1, 30.0, 11),
                  "c_tau": np.geomspace(0.05, 20.0, 11)},
    "two_state": {"c_k": np.geomspace(0.02, 20.0, 11),
                  "c_beta": np.geomspace(0.05, 20.0, 11)},
    "rtip":      {"c_w": np.geomspace(0.2, 20.0, 7),
                  "c_a": np.geomspace(0.05, 20.0, 7),
                  "c": np.array([0.42, 0.5, 0.65, 0.85, 1.2, 2.0, 4.0])},
    "ising":     {"c_J": np.geomspace(0.02, 5.0, 7),
                  "c_T": np.geomspace(0.05, 10.0, 7),
                  "c_tau": np.geomspace(0.05, 20.0, 7)},
    # Null DB: ranges declared in freeze_addendum_null_db.md section (b),
    # c_gamma' in [1e-3, 10] and c_beta in [1e-2, 1e2], on a log grid at the
    # same 11-point refinement the other two-constant nulls used.
    "db":        {"c_gamma": np.geomspace(1e-3, 10.0, 11),
                  "c_beta": np.geomspace(1e-2, 1e2, 11)},
}


def excluded_cells():
    """Optional training-cell exclusion list (R43 robustness re-calibration).

    When handoff/exclude_cells.json exists, those cell indices are dropped from
    the calibration targets.  Used to test whether the 12-0 morphology
    head-to-head survives removing the 7 training cells that contain
    zero-clean draws -- cells whose T1-T4 targets are still defined, because
    those are ensemble-mean trajectory statistics, but where no agent
    completed a clean passage.
    """
    import os
    p = "handoff/exclude_cells.json"
    if os.environ.get("QC_EXCLUDE_ZEROCLEAN") and os.path.exists(p):
        return set(json.load(open(p))["exclude"])
    return set()


def focal_targets():
    """Per-training-cell focal values, averaged over the 8 draws."""
    import glob
    R = [np.load(f) for f in glob.glob("s3/focal_*.npz")]
    rows = np.vstack([r["rows"] for r in R])
    idx = np.vstack([r["idx"] for r in R])
    keys = list(R[0]["keys"])
    out = {}
    for i, c in enumerate(cells()):
        if (not is_train(c)) or i in excluded_cells():
            continue
        m = idx[:, 0] == i
        out[i] = {k: float(np.nanmean(rows[m, keys.index(k)]))
                  for k in ("m_end", "m_mean", "m_var", "T4")}
    return out


def objective(model, par, tg, scale):
    """Standardised SSE against the focal targets on the training cells."""
    sse = 0.0
    cl = cells()
    for i, tv in tg.items():
        c = cl[i]
        vals = []
        for sd_seed in CAL_SEEDS:
            try:
                r = M.simulate(model, c[0], c[1], c[2], c[3], sd_seed, par)
            except Exception:
                return np.inf
            vals.append(r)
        g = lambda k: float(np.nanmean([v[k] for v in vals]))
        w_t1t2 = 2.0 if (c[1] >= 1.0 or c[2] == 0.0) else 1.0
        sse += w_t1t2 * ((g("m_end") - tv["m_end"]) / scale["m_end"]) ** 2
        sse += ((g("m_mean") - tv["m_mean"]) / scale["m_mean"]) ** 2
        sse += ((g("m_var") - tv["m_var"]) / scale["m_var"]) ** 2
        if np.isfinite(tv["T4"]) and np.isfinite(g("T4")):
            sse += ((g("T4") - tv["T4"]) / scale["T4"]) ** 2
    return float(sse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[m for m in M.MODELS if m != "focal"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    tg = focal_targets()
    scale = {}
    for k in ("m_end", "m_mean", "m_var", "T4"):
        v = np.array([t[k] for t in tg.values()], float)
        v = v[np.isfinite(v)]
        scale[k] = float(v.std()) if v.std() > 1e-12 else 1.0

    g = GRIDS[a.model]
    names = list(g)
    pts = list(itertools.product(*[g[n] for n in names]))
    mine = [(i, p) for i, p in enumerate(pts) if i % a.nshards == a.shard]
    os.makedirs("s3cal", exist_ok=True)

    res = []
    for i, p in mine:
        par = dict(zip(names, [float(x) for x in p]))
        res.append([i] + [par[n] for n in names] + [objective(a.model, par, tg, scale)])
        print(f"{a.model} shard {a.shard}: pt {i} {par} sse={res[-1][-1]:.4f}", flush=True)

    import os as _os
    _tg = "X" if _os.environ.get("QC_EXCLUDE_ZEROCLEAN") else ""
    np.savez_compressed(f"s3cal/{a.model}{_tg}_{a.shard}.npz",
                        res=np.array(res, float),
                        names=np.array(names),
                        scale=np.array([json.dumps(scale)]))
    print(f"{a.model} shard {a.shard} done: {len(res)} points")


if __name__ == "__main__":
    main()
