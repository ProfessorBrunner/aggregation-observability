# Verification summary, revision 4 / release v1.2

Environment: Python 3.12.3, numpy 2.4.4, scipy 1.17.1. Full logs: `quick_checks.log`, `full_run_part1.log`, `full_run_part2.log`, `full_run_part3.log`, `simulation_checks.log` (plus the smoke-test output below). All commands run from `code/v2/` unless noted.

## Rerun for this release

| check | command | expected | obtained | runtime | manuscript location |
|---|---|---|---|---|---|
| witnesses, independent check | `python verify_witnesses.py` | 73 checked, 0 failures, max 2(ℓ_QQ−ℓ_cand) ≤ 1e−9 | 73, 0, 4.55e−13 | <1 s | Sec. V.B–C; Supp. S5 |
| endpoint depends on c_β c′_γ only | `python endpoint_check.py` | −0.0890, −0.1978, −0.3595; equal pair | as expected; −0.286239 twice | <1 s | Supp. S2, Eq. (S5) |
| disk bound, radii | `python disk_check.py` | forms agree (rounding near c = 0, 1); radii 0.872 and 2.43 | 4.2e−6; 0.872; 2.43 | 1 s | Sec. III, Eq. (19), Fig. 1 |
| Theorem 4, numerical | `python jordan_check.py` | Λ ≤ 2e−10 for all; control ~1.6e5 | ≤ 1.2e−10; 156674 | 208 s | Sec. III, Theorem 4 |
| Λ for every dataset | `python run_exact.py` | 0 datasets with Λ > 1e−6 | 0 | 83 s | Sec. V.C, Fig. 3 |
| abortion pair | `python schuman_fit.py` | published table Λ = 0; exchanged moments ≈ 9.39 (estimate) | 0.0000; 9.3873 | 9 s | Sec. V.A, V.C |
| implied covariances | `python implied_covariances.py` | median \|C_q\| 0.057, \|C_r\| 0.045, max 0.17; 62 of 72 | as expected (max 0.169) | 1 s | Sec. V.C |
| coding robustness | `python check_coding.py` | medians 4.6 / 14.0 pts; max 22.7 / 40.7; 25 rows; min 11.8 | as expected | 118 s | Sec. V.C |
| relative order sizes | `python order_weight_check.py` | 144 fits, none outside the hull | 144, max Λ 1.5e−11, 0 outside | 134 s | Sec. V.A; Supp. S5 |
| threshold 50% points (values reported) | `cd simulation && python aggregate.py` | 1.4054 [1.3981, 1.4130]; 1.3421 [1.3344, 1.3495]; root-finding 1.3962 [1.3885, 1.4035] | as expected | 13 s | Sec. VI.C; Supp. S4 |
| threshold, parametric cross-check | `python fit_definedness.py` | same 50% points; intervals within 0.001 | 1.405 [1.397, 1.412]; 1.342 [1.335, 1.350] | 14 s | not reported |
| simulation smoke test | `cd simulation && python smoke_test.py` | 15/15 pass | 15/15 | 21 s | Sec. VI; Supp. S2–S4 |
| calibration at the best node | `cd simulation && python calibration_example.py` | SSE 67.034 | 67.034011 | 28 s | Supp. S3, Table S1 |
| readout convention check | `cd simulation && python diag_g2_readout.py` | stored convention reproduces g2_summary to 5e−7; same-field G₂ within 2.5%, below scatter | −0.00572, −0.0771, −0.3010 (changes +1.5e−4, +1.4e−3, +2.0e−3) | about 30 s (sharded) | Sec. VI.B |
| summary tables rebuilt from archives | `cd simulation && python aggregate.py` | all tables match stored; offset −0.2134 ± 0.0002 | worst deviation 4.8e−9; −0.21343 ± 0.00019 | 13 s | Figs. 4, 5, S1; Sec. VI.D |
| power validation (75 cells) | `cd code && python power_validation.py` | median 0.0032, mean 0.0054, rms 0.0086, max 0.040; FP 0.0065–0.0095 | as expected (max 0.0401) | 77 s | Sec. IV.C |

## Not rerun in this release

- **The full dynamical population grids and the calibration grid (Sec. VI; Supp. S3–S4).** The stored archives (`simulation/outputs/`, `simulation/s3/`) come from the scripts in `simulation/`, with configurations and seeds in `simulation/RUN_MANIFEST.md`. For this release, `aggregate.py` rebuilt every summary table from those archives, `smoke_test.py` reran selected trajectories, and `calibration_example.py` reevaluated the calibration at its best node. All three pass.
- **The bootstrap record for the exchanged-moment reading of dataset 72** (2004 stored replicates). Not rerun; it supports a numerical estimate on which no conclusion rests.

## Did any headline result, figure, fitted parameter or reported uncertainty change?

- **Headline results, figures, fitted parameters:** no change.
- **Reported uncertainties:** one correction. The Supplement's 1200-sample interval is now [1.334, 1.350], matching the archived bootstrap (1.3344 rounds to 1.334); it read [1.335, 1.350], carried over from the parametric cross-check. The dense-sampling interval [1.398, 1.413] and the root-finding interval [1.389, 1.404] are as in revision 3 and are reproduced by the archived code.
- **Reported numbers:** one set changed. The Sec. IV.C power-validation statistics now come from the archived reimplementation (`code/power_validation.py`); the earlier values came from an implementation that is not available, and their "size inflation" claim is withdrawn.
- **Code corrections:** `simulation/diag_dispersion.py` still implemented the superseded phase rate ε/(2v). It is corrected at source, and its output column was already corrected in revision 3. `aggregate.py` recomputes the affected columns; figures and text are unaffected. `requirements.txt` now lists statsmodels, which `aggregate.py` needs.
- **Readout check (rerun, 192 focal integrations):** the pooled-reciprocity readout evaluates c_i at x_read and q_i at the first saved sample after it, 0 to 0.068Δ later depending on the draw. Recomputing G₂ with both at the same field (`simulation/diag_g2_readout.py`) moves it toward zero by at most 2.5%, below the across-draw scatter at every dispersion. The stored values −0.0059, −0.078, −0.303 become −0.0057, −0.077, −0.301; the text reports both (Sec. VI.B).
