# Which quantum-like constraints survive disorder averaging?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22948242.svg)](https://doi.org/10.5281/zenodo.22948242)

Code and derived data for:

> R. J. Brunner, *Which quantum-like constraints survive disorder averaging? Reciprocity and observability in heterogeneous populations* (manuscript submitted to Physical Review E, 2026).

## Release v1.3

v1.3 accompanies revision 5 of the manuscript. It supersedes v1.2 (commit d879cc9) in these respects:
- **Threshold intervals.** They now come from a seed-block bootstrap (`simulation/seed_block_bootstrap.py`), which respects the common population seeds across dispersions. The within-dispersion bootstraps are superseded.
- **Calibration.** `simulation/calibration_decomposition.py` decomposes the calibration loss at the selected node and counts the timing residuals. `simulation/comparator_scores.py` reproduces all five Table S1 losses at their selected nodes, from `simulation/calibration_records/`.
- **Finite start.** `simulation/start_check.py` documents the cutoff dependence.
- **Power benchmark.** `code/power_validation.py` computes the analytic benchmark from quadrature population moments; the survey draws are unchanged.
- **Witness checker.** `code/v2/verify_witnesses.py` exits with status 1 unless all 73 witnesses are found and pass.
- **Documentation.** Corrected: the 216-cell sweep, the sweep range, the 12-agent state-update check, the solver docstrings, and defaults for `diag_g2_readout.py`.

Additional checks, from `simulation/`: `python seed_block_bootstrap.py` (seconds), `python calibration_decomposition.py` (about 30 s), `python start_check.py` (seconds). Logs: `verification/rev5_checks.log`.

## Release v1.2

v1.2 accompanies revision 3 of the manuscript (paper and Supplemental Material dated September 25, 2026). It adds per-dataset proof files for the empirical result, and scripts for the three checks added in that revision.

### Quick verification (under two minutes)

```
pip install -r requirements.txt
cd code/v2
python verify_witnesses.py        # < 1 s: checks all 73 proof files independently of the fitting code
python endpoint_check.py          # seconds: Supplement S2 endpoint argument
python disk_check.py              # seconds: Eq. (19) and the radii in Fig. 1
python jordan_check.py            # about 4 min (longer on slow machines): Theorem 4, numerical check
```

Expected output: `73 witnesses checked; failures: 0; largest 2(l_QQ - l_cand) = 4.55e-13`; endpoints −0.0890, −0.1978, −0.3595 and two equal values −0.286239; largest dataset radius 0.872 and counterexample 2.43; Λ ≤ 1.2e-10 for all higher-dimensional respondents and about 1.6e5 for the control.

### Full reproduction

```
cd code && python parse_s1.py path/to/S1.pdf && python recover_n.py     # tables and sample sizes (optional; outputs are stored)
cd v2
python make_witnesses.py          # about 1 min: rebuilds the 73 proof files
python run_exact.py               # about 1 min: Lambda for every dataset
python schuman_fit.py             # about 1 min: abortion pair, published table and exchanged moments
python check_coding.py            # about 2 min: coding robustness (Sec. V)
python implied_covariances.py     # seconds: implied covariances and detectability counts (Sec. V)
python fit_definedness.py         # about 1 min: 50% point of the threshold statistic (Sec. VI, Supplement S4)
python order_weight_check.py      # about 2 min: insensitivity to relative order sizes (Sec. V)
cd ../../simulation && python smoke_test.py && python calibration_example.py && python aggregate.py && cd ../code/v2   # about 1 min: dynamical checks (Sec. VI, Supp. S3-S4)
cd .. && python power_validation.py && cd v2   # a few min: Monte Carlo check of Eq. (22) (Sec. IV.C)
cd .. && python make_fig_hull.py && python make_figs.py                 # all figures
```

No step uses random numbers except `jordan_check.py` (seed 11), `disk_check.py` (seed 3), `fit_definedness.py` (seed 7) and the stored bootstrap record, whose seeds are in the file names.

### Proof files (`data/region_test_v2/witnesses/`)

One JSON file per dataset. Each holds:
- the adopted counts, with their source (published moments; for dataset 72, Schuman et al. 1981, Table 2);
- a mixture of at most six qubit respondent types (q, r, c) with weights;
- the cell probabilities the mixture implies;
- the closed-form QQ log-likelihood, the mixture's log-likelihood, and the resulting bound 0 ≤ Λ ≤ 2(ℓ_QQ − ℓ_cand).

`verify_witnesses.py` rechecks every file using numpy alone. The mixtures come from Frank–Wolfe with away steps, reduced to at most six types by a vertex of the feasible-weight polytope (Carathéodory), which leaves the moment vector unchanged.

### Which file supports which claim

| claim | location | files |
|---|---|---|
| Λ = 0 for all 72 datasets (and Rose–Jackson) | Sec. V, Fig. 3 | `witnesses/`, `verify_witnesses.py`, `make_witnesses.py`, `run_exact.py` |
| abortion pair from the published table; exchanged-moment value 9.39 | Sec. V | `schuman_fit.py`, `bootstrap_published_moments/` |
| coding robustness (order effects, exchanged reading) | Sec. V | `check_coding.py`, `order_effect_plausibility.csv`, `swapped_stage*.csv` |
| implied covariances, 62 of 72 detectable | Sec. V | `implied_covariances.py`, `implied_covariances.csv` |
| disk bound, radii in Fig. 1 | Sec. III, Eq. (19), Fig. 1 | `disk_check.py`, `make_fig_hull.py` |
| Theorem 4, numerical check | Sec. III | `jordan_check.py` |
| endpoint depends on c_β c′_γ only | Supplement S2 | `endpoint_check.py` |
| insensitivity to relative order sizes (144 fits) | Sec. V.A | `order_weight_check.py`, `order_weight_check.csv` |
| power validation (75 cells) | Sec. IV.C | `code/power_validation.py`, `data/power_validation.csv` |
| second power check (clipped normal) | Sec. IV.C | `code/check_detectability.py`, its saved output |
| threshold 50% points and intervals | Sec. VI, Supplement S4 | points: `simulation/aggregate.py`; intervals and the detector difference: `simulation/seed_block_bootstrap.py` (seed-block resampling). The within-dispersion bootstraps in `aggregate.py` and `code/v2/fit_definedness.py` are superseded. |
| dynamical simulations, calibration, rival offset | Sec. VI, Supplement S3 | `simulation/` (see `simulation/PAPER1_MAP.md`; smoke test, calibration example, aggregation) |
| figures 2–5, S1 | Secs. IV–VI | `make_figs.py`, `data/cs_simulation/` |

The full dynamical simulations of Sec. VI were not rerun for v1.2. Their stored archives are in `simulation/outputs/` and `simulation/s3/`. `simulation/aggregate.py` rebuilds every summary table from those archives, `simulation/smoke_test.py` reruns selected trajectories, and `simulation/calibration_example.py` reevaluates the calibration at its best node. `data/cs_simulation/` holds copies of the rebuilt tables, whose only change from v1.1 is the corrected phase-scrambling column.

## Release v1.1

v1.1 accompanies the submitted manuscript. It supersedes the v1.0 region test in three ways:

- **Solver.** The likelihood-ratio statistic Λ = 2(ℓ_QQ − ℓ_P) is computed with a linear oracle for the qubit moment set, and every fit is certified by its Frank–Wolfe duality gap. v1.0 used a sampled oracle whose small positive Λ values were optimizer residue.
- **Abortion pair from its published table.** Dataset 72 is taken from Schuman, Presser and Ludwig (1981), Table 2, because two of its published moments are exchanged relative to the definition of Dzhafarov, Zhang and Kujala (2016).
- **Robustness to the coding question.** A check of every row under the exchanged reading of the moments.

The v1.0 scripts and results remain in `code/` and `data/region_test/` as the record of that analysis.

**Result:** Λ = 0 for all 72 datasets (and for the excluded Rose–Jackson pair).

> **Superseded in v1.2.** The v1.1 descriptions of an "exact" oracle and "certified" fits overstated the guarantee: the oracle is a grid with local refinement. From v1.2 the zero results rest on saved feasible-mixture witnesses checked by `code/v2/verify_witnesses.py`; nonzero values are numerical estimates.

## Contents

| path | what it is |
|---|---|
| `code/v2/region_exact.py` | linear oracle (grid plus refinement), QQ-constrained and projective-class maximum likelihoods, Λ |
| `code/v2/boundary_fit.py` | direct mixture fit for optima on the hull boundary |
| `code/v2/common.py` | table reconstruction from the published moments; Schuman et al. counts |
| `code/v2/run_exact.py` | Λ for every dataset (about one minute) |
| `code/v2/schuman_fit.py` | dataset 72 from the published table and from the exchanged moments |
| `code/v2/check_coding.py` | implied order effects and Λ under the exchanged reading |
| `code/v2/implied_covariances.py` | observed reciprocity differences and implied covariances for the 72 datasets |
| `code/v2/fit_definedness.py` | logistic 50% point of the threshold statistic with bootstrap interval |
| `code/v2/boot_published_moments.py` | record: bootstrap of dataset 72 read from its exchanged moments |
| `code/make_fig_hull.py`, `code/make_figs.py` | regenerate all figures in `figures/` |
| `code/check_detectability.py` | independent Monte Carlo check of the power boundary, with saved output |
| `code/parse_s1.py`, `code/recover_n.py` | reconstruction of the tables and recovery of sample sizes (used by v1.0 and v1.1) |
| `data/region_test_v2/` | v1.1 and v1.2 results: Λ, witnesses, coding check, order-weight check, implied covariances, bootstrap record |
| `data/cs_simulation/` | copies of the summary tables rebuilt by `simulation/aggregate.py`, used by the figure scripts |
| `docs/` | the analysis plans; v2 was written before the test reported in the paper was run |
| `simulation/` | dynamical simulation and calibration code, archives, smoke test and aggregation (see `simulation/PAPER1_MAP.md`) |

## Reproducing

```
pip install -r requirements.txt
cd code && python parse_s1.py path/to/S1.pdf && python recover_n.py   # tables and sample sizes
cd v2 && python run_exact.py && python schuman_fit.py && python check_coding.py && python implied_covariances.py
cd .. && python make_fig_hull.py && python make_figs.py
```

`S1.pdf` is Supplementary Information S1 of Dzhafarov, Zhang and Kujala, *Phil. Trans. R. Soc. A* **374**, 20150099 (2016), in the source bundle of arXiv:1504.07422; it is not redistributed here. `parse_s1.py` requires `pdftotext` (poppler). The v1.1 scripts read the reconstructed tables from `data/region_test_v2/tables73_n.csv`, so they run without the PDF.

## Licenses

Code: MIT (see `LICENSE`). Derived data and figures: CC BY 4.0. The survey moments are those published by Dzhafarov, Zhang and Kujala (2016) from data assembled by Wang, Solloway, Shiffrin and Busemeyer, *Proc. Natl. Acad. Sci. USA* **111**, 9431 (2014); the abortion-pair counts are from Schuman, Presser and Ludwig, *Public Opin. Q.* **45**, 216 (1981). Cite those works when using `data/`.

## Citation

See `CITATION.cff`.
