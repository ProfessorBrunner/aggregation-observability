"""Measure the cell-163 dispersion sd(eta)/Delta = 1.0 exactly, which the
diag_dispersion / diag_definedness grids skip.  Diagnostic, no freeze."""
import argparse, os, sys
import numpy as np
sys.path.append(os.getcwd())
import stage3_models as M
from diag_dispersion import DELTA, Z, KAPPA, K, N_SAMPLE, M_INF
from diag_definedness import SEEDS_256, N_SAMPLE_FROZEN
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int); ap.add_argument("--nshards", type=int)
a = ap.parse_args(); os.makedirs("s3sd1", exist_ok=True)
mine = [s for n, s in enumerate(SEEDS_256) if n % a.nshards == a.shard]
rows = []
for sd_seed in mine:
    rf = M.simulate("focal", DELTA, Z, KAPPA, 1.0, sd_seed, K=K, n_sample=N_SAMPLE, return_traj=True)
    rl = M.simulate("focal", DELTA, Z, KAPPA, 1.0, sd_seed, K=K, n_sample=N_SAMPLE_FROZEN, return_traj=True)
    mm, ml = float(rf["m"].min()), float(rl["m"].min())
    rows.append([sd_seed, mm, float(rf["m"][-1]), M_INF - mm, float(mm <= -0.5), ml, M_INF - ml, float(ml <= -0.5)])
np.savez_compressed(f"s3sd1/sd1_{a.shard}.npz", rows=np.array(rows, float),
                    cols=np.array(["seed","m_min","m_end","U","defined","m_min_1200","U_1200","defined_1200"]))
print(f"shard {a.shard}: {len(rows)}", flush=True)
