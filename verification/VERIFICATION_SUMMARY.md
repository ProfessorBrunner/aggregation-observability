# Verification summary, revision 5 / release v1.3

Environment: Python 3.12.3, numpy 2.4.4, scipy 1.17.1, statsmodels (for `aggregate.py`). Logs in this folder: `quick_checks.log`, `full_run_part1-3.log`, `simulation_checks.log` (revision 4); `rev5_checks.log` (this revision).

## Checks

| check | command | expected | obtained | runtime | manuscript location | this revision |
|---|---|---|---|---|---|---|
| witnesses, independent check | `code/v2: python verify_witnesses.py` | 73 found, 0 failures, exit 0 | 73, 0, 4.55e−13, PASS, exit 0 (0 files: exit 1) | <1 s | Sec. V.B–C; Supp. S5 | rerun; checker hardened |
| threshold intervals, seed-block | `simulation: python seed_block_bootstrap.py` | points 1.4054, 1.3421, 1.3962 | [1.3902, 1.4206], [1.3272, 1.3571], [1.3813, 1.4113]; difference 0.00918 [0.00679, 0.01175] | 2 s | Sec. VI.C; Supp. S4 | **new; intervals changed** |
| calibration loss at the selected node | `simulation: python calibration_decomposition.py` | total 67.034 | 67.034011 = 23.96 + 15.95 + 14.74 + 12.39; timing 38 / 33 / 5 | 28 s | Supp. S3, Eq. (S6) | **new** |
| calibration losses of all five rivals at their selected nodes | `simulation: python comparator_scores.py` | Table S1: 67.0, 549.9, 550.2, 565.7, 566.7 | 67.034, 549.934, 550.152, 565.670, 566.672 | ~3 min | Supp. S3, Table S1 | **new** |
| finite-start dependence | `simulation: python start_check.py` | — | q(2Δ) varies by at most 2.92e−4 over starts −320 … −640Δ | 12 s | Sec. II; Sec. VI.C | **new** |
| power validation (quadrature benchmark) | `code: python power_validation.py` | — | median 0.0031, mean 0.0048, rms 0.0073, max 0.028 (N = 200); FP 0.0065–0.0095 | ~80 s | Sec. IV.C | **benchmark changed** (draws unchanged) |
| simulation smoke test | `simulation: python smoke_test.py` | 15/15 | 15/15 | 20 s | Sec. VI; Supp. S2–S4 | rerun |
| summary tables rebuilt | `simulation: python aggregate.py` | all tables match | worst deviation 4.829e−7 (`eps_at_min_mean`, print precision) | 13 s | Figs. 4, 5, S1 | rerun (revision 4); earlier "4.8e−9" corrected |
| calibration at the best node | `simulation: python calibration_example.py` | 67.034 | 67.034011 | 28 s | Supp. S3 | revision 4 |
| readout convention | `simulation: python diag_g2_readout.py` | stored convention reproduces to 5e−7 | same-field G₂ −0.00572, −0.0771, −0.3010 | ~30 s sharded | Sec. VI.B | revision 4 |
| endpoint, disk, Theorem 4, Λ, abortion pair, covariances, coding, order weights | as in revision 4 | as in revision 4 | reproduced (revision-4 logs) | — | Secs. III, V; Supp. S2 | unchanged |

## Not rerun

- **The full dynamical population grids and the calibration grids for all rivals.** Stored archives are reaggregated and selected trajectories recomputed. Table S1's losses are reproduced at each rival's selected node; the per-node grid losses were not archived, so the grid orderings are not rechecked (stated in the caption).
- **The 2004-replicate bootstrap for the exchanged-moment reading of dataset 72.** It supports a numerical estimate on which no conclusion rests.

## Did any headline result, figure, fitted parameter or reported uncertainty change?

- **Headline results, figures, fitted parameters:** no change.
- **Reported uncertainties: changed.** The three threshold intervals roughly double under seed-block resampling: 1.405 [1.390, 1.421], 1.342 [1.327, 1.357], 1.396 [1.381, 1.411]. The point estimates are unchanged. The detector statement changes from "overlapping, no effect on location" to a small systematic difference, 0.009Δ [0.007, 0.012].
- **Reported numbers: changed.** Sec. IV.C now uses the quadrature benchmark (max difference 0.028, was 0.040).
- **Description changes, no numerical change:**
  - The calibration objective is described as implemented.
  - Table S1 labels and constant counts are corrected, and its losses are reproduced at the selected nodes. The "five best grid points" statement is removed (no archived per-node losses).
  - The finite-start claim is replaced by a documented point check.
  - The readout-sensitivity scope is limited to the three checked dispersions.
  - The abstract's scope matches Theorem 4.
