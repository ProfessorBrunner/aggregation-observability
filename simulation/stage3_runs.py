"""
stage3_runs.py -- Stage 3 grid execution, sharded across subprocesses.

  python stage3_runs.py --model focal --shard i --nshards n [--params json]

Writes s3/<model>_<shard>.npz.  This sandbox denies POSIX semaphores, so
process pools cannot start; sharding is by index into the canonical cell list.

The focal model has ZERO free parameters (c_gamma = 0.02 declared), so its grid
is run once and never re-run.  Nulls take --params.
"""

import argparse
import json
import os

import numpy as np

import stage3_models as M

# ---- frozen cell lists, null_family_freeze_v4.md section 0.4 ----
Z_ALL = (0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.0)
Z_TR = (0.05, 0.2, 0.6, 1.5, 2.0)
DELTA = (0.25, 0.5, 1.0)
PAIRS = tuple((k, s) for k in (0.0, 0.15, 0.35) for s in (0.5, 1.0, 2.0))
P_TR = ((0.0, 0.5), (0.0, 2.0), (0.15, 1.0), (0.35, 0.5), (0.35, 2.0))
SEEDS = (3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148)

SCALARS = ("P_cell", "z_meas_cell", "frac_clean", "n_clean", "n_never",
           "n_multi", "n_noexit", "m_end", "m_start", "T4", "T4n",
           "var_cs_max", "chi_max", "m_mean", "m_var", "m_ac1", "t_end")


def cells():
    """Canonical ordered cell list: 216 = z(8) x Delta(3) x (kappa,sd)(9)."""
    return [(D, z, k, s) for D in DELTA for z in Z_ALL for (k, s) in PAIRS]


def is_train(c):
    return c[1] in Z_TR and (c[2], c[3]) in P_TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=M.MODELS)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    par = json.loads(a.params)
    cl = cells()
    mine = [(i, c) for i, c in enumerate(cl) if i % a.nshards == a.shard]
    os.makedirs("s3", exist_ok=True)

    rows, idx, Pall = [], [], []
    for i, c in mine:
        for js, sd_seed in enumerate(SEEDS[:a.seeds]):
            r = M.simulate(a.model, c[0], c[1], c[2], c[3], sd_seed, par)
            rows.append([float(r[k]) for k in SCALARS])
            idx.append([i, js, c[0], c[1], c[2], c[3]])
            Pall.append(r["P"])
        print(f"{a.model} shard {a.shard}: cell {i} D={c[0]} z={c[1]} "
              f"kap={c[2]} sd={c[3]} P={rows[-1][0]:.4f}", flush=True)

    np.savez_compressed(f"s3/{a.model}{a.tag}_{a.shard}.npz",
                        rows=np.array(rows, float),
                        idx=np.array(idx, float),
                        P=np.array(Pall, float),
                        keys=np.array(SCALARS),
                        params=np.array([json.dumps(par)]))
    print(f"{a.model} shard {a.shard} done: {len(rows)} cell-draws")


if __name__ == "__main__":
    main()
