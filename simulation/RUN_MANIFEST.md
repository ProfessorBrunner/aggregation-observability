# simulation_v12 — run manifest

How each shipped output was produced.  Everything below is the configuration of
record; nothing here was rerun to build this package except the two checks in
`smoke_test.py` and `calibration_example.py`.

## Common corner

All eight diagnostics run at one cell of the frozen sweep (the "cell-163
corner"):

    Delta = 1, z = Delta^2/v = 0.2  (so v = 5), kappa = 0, u_dot = 0,
    lambda = 0, K = 320, N = 100 agents, gamma_phi = 0.02 Delta,
    gamma_r = 0 exactly; x(t) starts at -K Delta - max(eta) = -320 Delta - max(eta)
    and runs for 2.5 x span / v, span = 2K Delta + (max(eta) - min(eta)).

Dispersion is the one swept quantity: eta_i ~ N(0, sd^2) added to the detuning,
reported as `sd(eta)/Delta`.  `m(t)` is the ensemble mean of `r_z`.
Derived constants: `m_inf = 2 exp(-2 pi z) - 1 = -0.430781`,
`U* = m_inf + 0.5 = 0.069219`.

## Environment

Python 3.11, numpy, scipy, statsmodels (for the binomial GLM in `aggregate.py`
only).  Integrator: `scipy.integrate.solve_ivp`, DOP853, as set in
`stage3_models.py`; that file is the single source of the dynamics and was not
modified for this package except for the additive opt-in flags listed in
`README.md`.  Runs are single-threaded; the sweeps were sharded across cores
with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` set, because
28 unpinned shards oversubscribe BLAS on a 32-core machine.

## Per script

### 1. `stage3_models.py`
The model library: `focal` plus the five classical nulls (`lag`, `rtip`,
`two_state`, `ising`, `db`).  Not run directly.  `simulate(model, Delta, z,
kappa, sd_eta, seed, params=None, K=..., n_sample=..., ...)` is the only entry
point every other script uses.

### 2. `stage3_runs.py`
Defines the frozen 216-cell sweep (75 training, 141 held-out), `cells()`, `is_train()`, and the eight
population draws `SEEDS = (3141, ..., 3148)`.  Imported by
`stage3_calibrate.py` and `diag_dispersion.py`; not run here.

### 3. `stage3_calibrate.py` — the calibration grid
    python stage3_calibrate.py --model db --shard $i --nshards 24     # i = 0..23
Objective: unweighted standardised SSE against the **focal model's own output**
on the 75 frozen training cells (no empirical data anywhere).  Terms: T1
(`m_end` where z >= 1.0), T2 (`m_end` where kappa = 0), T3 (`m_mean`,
`m_var`, all cells), T4 (where finite for both), each divided by the
across-training-cell sd of the focal value; the T1/T2 term carries weight 2.0
on cells with z >= 1.0 or kappa = 0.  Calibration seeds `CAL_SEEDS =
SEEDS[:2] = (3141, 3142)`, i.e. two draws per cell at the grid stage.
Null DB grid: `c_gamma = geomspace(1e-3, 10, 11)`, `c_beta =
geomspace(1e-2, 1e2, 11)`, 121 nodes.
Inputs: focal targets in `s3/focal_*.npz` (24 shard files, shipped here).
Outputs: `s3cal/db_<shard>.npz`.
Best node, interior on both axes: `c_gamma = 0.25118864` (index 6 of 10),
`c_beta = 1.0` (index 5 of 10), SSE `67.034`.  `calibration_example.py`
re-evaluates exactly that node and reproduces `67.034011`.

### 4. `diag_dispersion.py` — undershoot against dispersion
    python diag_dispersion.py
Single process, no sharding.  `SD_GRID` = 17 dispersions
(0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0,
2.5, 3.0) x 8 seeds `(3141 ... 3148)` = 136 focal runs at `n_sample = 24000`,
each paired with a Null DB run at `DB_PARAMS = {"c_gamma": 0.25118864,
"c_beta": 1.0}`.  Also computes the three predictors in its docstring
(convolution, dip curvature, phase scrambling) and the single-agent reference
curve `R(eps)` at sd = 0.
Output: `diag_dispersion.npz` (`rows`, `pred`, `E`, `R`, `scal`).

### 5. `diag_dbmatch.py` — offset / shape decomposition
    python diag_dbmatch.py
Single process.  Same 17 x 8 design.  Splits the focal-minus-Null-DB
trajectory difference into a constant level offset and a residual shape term,
on the full sweep and on the post-crossing segment, and records the
correlation.
Output: `diag_dbmatch.npz`.

### 6. `diag_definedness.py` — T4 definedness, 256 draws, two lattices
    for i in $(seq 0 27); do python diag_definedness.py --shard $i --nshards 28 & done; wait
    python aggregate.py --merge
`SD_GRID_256` = 21 dispersions (the 17 above plus 1.25, 1.35, 1.45, 1.55) x
256 seeds `range(900000, 900256)` = 5376 draws.  Each draw is simulated twice,
at `n_sample = 24000` and at the frozen `n_sample = 1200`; the coarse lattice
is not a subsample of the fine one, so both are integrated.  The same cell also
records the Null DB monotonicity audit (`y - y*` per agent per sample, split
into all samples and the resolvable set `|y*| < 1 - 1e-15`).
Shards: `s3disp/def_<shard>.npz` -> merged `diag_definedness.npz` (5376 x 22).

### 7. `diag_continuous.py` — the m = -0.5 crossing by root-finding
    for i in $(seq 0 27); do python diag_continuous.py --shard $i --nshards 28 & done; wait
    python aggregate.py --merge
Same 21 x 256 design and the same seed block.  Uses the non-terminal
`m_cross_level = -0.5` event on the integrator's dense output (direction -1)
instead of scanning saved samples.
Shards: `s3cont/cont_<shard>.npz` -> merged `diag_continuous.npz` (5376 x 13).

### 8. `diag_sd1.py` — the sd(eta)/Delta = 1.0 point
    for i in $(seq 0 23); do python diag_sd1.py --shard $i --nshards 24 & done; wait
    python aggregate.py --merge
Exactly sd = 1.0, which both grids skip, on the same 256 seeds and both
lattices.
Shards: `s3sd1/sd1_<shard>.npz` -> merged `diag_sd1.npz` (256 x 8).

### 9. `diag_g2.py` — pooled reciprocity difference G2
    for i in $(seq 0 23); do python diag_g2.py --shard $i --nshards 24 & done; wait
    python aggregate.py --merge
21 dispersions (`SD_GRID_256`) x 64 seeds `range(900000, 900064)` (the first 64
of the declared block) x 7 readout fields `X_READ = (2, 5, 10, 20, 50, 100,
320) Delta` = 9408 population readouts.  Needs `sec4_order_restriction.py` and
the `return_Y` / `r_xyz` opt-ins of `stage3_models.py`.  For each population it
records the covariance form, the pooled 2x2 table built algebraically from all
100 agents, the explicit Lüders state-update value on the first 12 agents, and the pooled QQ arms.
Shards: `s3g2/g2_<shard>.npz` -> merged `diag_g2.npz` (9408 x 25).

### 10. `aggregate.py` — summary tables and threshold statistics
    python aggregate.py
Rebuilds the four summary CSVs from the merged archives and verifies them
against the shipped copies; recomputes the convolution/curvature/phase
thresholds, the pooled Null DB offset, the three logistic 50% points with a
2000-replicate percentile bootstrap (`rng = default_rng(20260925)`, one stream
shared across the three detectors in the order event, 24000, 1200), and the G2
scatter crossover.  Runs in about 7 s and integrates nothing.

The summary CSVs and these threshold statistics were originally assembled
interactively rather than by a script, so the eight diagnostic scripts alone
did not reproduce them.  `aggregate.py` was written for this package to close
that gap; it reproduces every column of all four CSVs to at worst 4.83e-07
(that being the printing precision of `eps_at_min_mean`, stored to 8
significant figures) and all three bootstrap intervals exactly.

## Checks

    python smoke_test.py            # 15 checks, ~11 s
    python calibration_example.py   # Null DB best node, ~16 s
    python aggregate.py             # four tables + thresholds, ~7 s
