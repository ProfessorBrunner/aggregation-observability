"""
calibration_example.py -- evaluate the Null DB objective at its best grid node.

  python calibration_example.py

Reproduces one point of the calibration grid behind the null-calibration table:
the interior minimum of the Null DB (detailed-balance two-state rate process)
objective, c_gamma = 10^-0.6 = 0.25118864, c_beta = 1.0, stored SSE = 67.034.

This calls stage3_calibrate.objective() -- the same function the grid sweep
calls -- on the 75 frozen training cells with the two calibration seeds
(3141, 3142).  It needs the focal targets in s3/focal_*.npz, shipped here; it
does NOT rerun the focal grid.  Wall time ~5 s.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import stage3_calibrate as C

BEST = {"c_gamma": 0.25118864, "c_beta": 1.0}
STORED_SSE = 67.034

g = C.GRIDS["db"]
print("Null DB grid of record:")
for k, v in g.items():
    print(f"  {k:9s} {len(v):3d} nodes  geomspace({v[0]:g}, {v[-1]:g})")
print(f"  {len(g['c_gamma']) * len(g['c_beta'])} nodes total, "
      f"{len(C.CAL_SEEDS)} seeds per cell (seeds {tuple(C.CAL_SEEDS)})")
for k, v in BEST.items():
    j = int(np.argmin(np.abs(g[k] - v)))
    assert abs(g[k][j] - v) < 1e-6 * v, (k, v, g[k][j])
    print(f"  best node: {k} = {v:.8f} is grid index {j} of {len(g[k]) - 1} "
          f"({'INTERIOR' if 0 < j < len(g[k]) - 1 else 'ON BOUNDARY'})")

tg = C.focal_targets()
scale = {}
for k in ("m_end", "m_mean", "m_var", "T4"):
    v = np.array([t[k] for t in tg.values()], float)
    v = v[np.isfinite(v)]
    scale[k] = float(v.std()) if v.std() > 1e-12 else 1.0
print(f"\nfocal targets loaded: {len(tg)} training cells")
print(f"standardisation (across-training-cell sd of the focal value):")
for k in sorted(scale):
    print(f"  {k:8s} {scale[k]:.6g}")

sse = C.objective("db", BEST, tg, scale)
print(f"\nSSE at the best node = {sse:.6f}   (stored {STORED_SSE})")
d = abs(sse - STORED_SSE)
print(f"|difference| = {d:.4f}   (stored value is quoted to 3 decimals)")
assert d < 5e-3, f"objective at the best node does not reproduce: {sse} vs {STORED_SSE}"
print("PASS")
