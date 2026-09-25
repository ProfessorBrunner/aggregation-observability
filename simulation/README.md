# simulation_v12 — dynamical simulation code for paper 1 (code release v1.2)

Synthetic and analytic only.  No market data, no options data, no empirical
input of any kind: the nulls are calibrated against the focal model's own
output, not against observations.

## Contents

    stage3_models.py          the dynamics: focal model + five classical nulls
    stage3_runs.py            the frozen sweep, training cells, population seeds
    stage3_calibrate.py       null calibration on the 75 training cells
    diag_dispersion.py        undershoot against dispersion, 3 predictors
    diag_dbmatch.py           focal-vs-Null-DB offset / shape decomposition
    diag_definedness.py       T4 definedness, 256 draws, two output lattices
    diag_sd1.py               the sd(eta)/Delta = 1.0 point the grids skip
    diag_continuous.py        the m = -0.5 crossing by event root-finding
    diag_g2.py                pooled reciprocity difference G2
    aggregate.py              summary tables + threshold statistics
    smoke_test.py             15 deterministic checks, ~11 s
    calibration_example.py    Null DB objective at its best grid node, ~16 s
    reference_values.json     the expected values smoke_test.py checks
    RUN_MANIFEST.md           configuration, seeds and command per script
    OUTPUT_MAP.md             what each script and output supports
    s3/focal_*.npz            focal targets for the calibration (24 shards)
    outputs/*.csv             the stored summary tables
    outputs/diag_*.npz        the stored per-draw archives

## Quick start

    python smoke_test.py            # does this code still give the stored numbers
    python calibration_example.py   # one calibration node, reproduces SSE 67.034
    python aggregate.py             # rebuild all four summary tables and verify

Needs numpy, scipy and statsmodels (the last only for the binomial GLM in
`aggregate.py`).  Nothing here uses a GPU or more than one core; the sweeps
were sharded across cores, see `RUN_MANIFEST.md`.

## Four things a reviewer should know

**1. The two extra imports.**  The eight diagnostic scripts are not
self-contained: `stage3_calibrate.py` and `diag_dispersion.py` import
`stage3_runs.py`, and `diag_g2.py` imports `sec4_order_restriction.py`.  Both
are included here for that reason.

**2. `aggregate.py` is new.**  The four summary CSVs and the threshold
statistics quoted in the memos were originally assembled interactively rather
than by a script, so the diagnostic scripts alone did not reproduce them.
`aggregate.py` was written for this release to close that gap.  It rebuilds
every column of all four CSVs from the archives and reports the deviation from
the shipped copies; the worst is `4.83e-07`, on `eps_at_min_mean`, which is
that column's printing precision.  It reproduces the three logistic 50% points
and all three bootstrap intervals exactly.

**3. One source correction.**  `diag_dispersion.py` originally implemented the
phase-scrambling rate as `Phi' = eps/(2v)`, taking the level splitting of
`H = (1/2) eps sigma_z + Delta sigma_x` as `eps/2` instead of
`Omega = sqrt(eps^2 + 4 Delta^2)`.  The formula is corrected in the copy
shipped here, with the derivation and its verification in the docstring
(`CORRECTED 2026-09-25`).  Consequences: `Phi'(eps1)` `0.4480 -> 0.9813`, the
damping factor `exp(-eps^2 s^2 / 8 v^2) -> exp(-(eps^2 + 4 Delta^2) s^2 /
2 v^2)`, the phase threshold `3.7204 -> 1.6986`, and `U_phase` at
`sd/Delta = 1.4` `0.228086 -> 0.108072`.  The convolution and curvature limbs
are unaffected, and no measurement changes — only the predictor.
`dispersion_summary.csv` v2 already carries the corrected column;
`diag_dispersion.npz` predates the correction, so its stored `pred[:, 4]`
(`U_phase`) and `scal[8]` (`s_thresh_phase`) are superseded and `aggregate.py`
ignores them, recomputing both.

**4. `stage3_models.py` carries four opt-in flags** added after the freeze for
these diagnostics, all default-off, nothing above them recomputed:
`return_traj` (pre-existing), `return_Y` (attaches `Y`, `eta`, `x_full`,
`t_full`), `r_xyz` on the focal output, and `m_cross_level` (a non-terminal
downward-crossing event returning `t_cross`, `n_cross`, `t_cross_first`,
`m_at_cross`, `solver_status`).

## Not in this package

The figure scripts.  Figs. 3-6 and Table 1 of record are drawn by
`make_tranche_figures.py` from five aggregate archives plus `qc_style.py`, and
`fig_s3_01/02` by `make_stage3_figures.py` from the full 72-shard `s3/` set;
only the 24 focal shards needed by the calibration are shipped here.  See
`OUTPUT_MAP.md` §B, which also records that no "S1", "Table S1" or supplement
exists in the manuscript of record, and queues that as R74.
