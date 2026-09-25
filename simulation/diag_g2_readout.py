"""Recompute G2 at x_read = 2 Delta with c_i and q_i at the SAME field --
the saved output sample at or after x_read -- against the stored convention
(c at exactly x_read, q at the sample).  Diagnostic, no freeze."""
import argparse, os, sys
import numpy as np
sys.path.append(os.getcwd())
import stage3_models as M
from diag_g2 import DELTA, Z, KAPPA, K, N_SAMPLE, g2_from_table, pooled_tables
from diag_definedness import SEEDS_256

XR, SDS = 2.0, (0.3, 1.1, 3.0)
SEEDS = SEEDS_256[:64]
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int); ap.add_argument("--nshards", type=int)
a = ap.parse_args(); os.makedirs("s3g2r", exist_ok=True)
jobs = [(s, sd) for s in SDS for sd in SEEDS]
rows = []
for s, sd_seed in [j for n, j in enumerate(jobs) if n % a.nshards == a.shard]:
    r = M.simulate("focal", DELTA, Z, KAPPA, s, sd_seed, K=K, n_sample=N_SAMPLE,
                   return_traj=True, return_Y=True)
    rb, eta, xf = r["r_xyz"], r["eta"], r["x_full"]
    k = int(np.nonzero(xf >= XR)[0][0])
    xs = float(xf[k])
    rx, rz = rb[0, :, k], rb[2, :, k]
    q = 0.5 * (1.0 + rz)
    qb = float(q.mean())
    out = [s, sd_seed, xs]
    for eps in (XR + eta, xs + eta):          # stored convention, then same-field
        den = np.sqrt(eps ** 2 + 4.0 * DELTA ** 2)
        c = 0.5 * (1.0 + eps / den)
        w = 0.5 * (1.0 + rx * (2.0 * DELTA / den) + rz * (eps / den))
        cov = float(np.cov(c, q, bias=True)[0, 1])
        a1, a2 = pooled_tables(c, q, w)
        out += [cov, cov / (qb * (1.0 - qb)), g2_from_table(a1), float(c.mean()),
                float(np.var(c))]
    rows.append(out + [qb])
np.savez_compressed(f"s3g2r/g2r_{a.shard}.npz", rows=np.array(rows, float),
                    cols=np.array(["sd", "seed", "x_sample",
                                   "cov_A", "G2_pred_A", "G2_meas_A", "cbar_A", "var_c_A",
                                   "cov_B", "G2_pred_B", "G2_meas_B", "cbar_B", "var_c_B",
                                   "qbar"]))
print(f"shard {a.shard}: {len(rows)}", flush=True)
