# Simulation outputs mapped to paper 1

"Which quantum-like constraints survive disorder averaging?", revision 4. The other documents in this folder (OUTPUT_MAP.md, RUN_MANIFEST.md) map the same outputs to memo sections of the project record; this file maps them to the paper.

| paper location | quantity | produced by | stored output |
|---|---|---|---|
| Sec. VI.B, Fig. 4(a),(b) | G₂ against dispersion and readout field; var(c); the identity check to machine precision | `diag_g2.py` (uses `sec4_order_restriction.py`), assembled by `aggregate.py` | `outputs/diag_g2.npz`, `outputs/g2_summary.csv` |
| Sec. VI.B | readout convention: c_i at x_read, q_i at the first saved sample after it (0 to 0.068Δ later, depending on the draw); G₂ with both at the same field: −0.0057, −0.077, −0.301 (at most 2.5% toward zero, below the across-draw scatter) | `diag_g2.py`; `diag_g2_readout.py` | `outputs/g2_readout_check.csv` |
| Sec. VI.C, Fig. 5 | measured undershoot (256 draws), convolution law, curvature form | `diag_definedness.py`, `diag_dispersion.py` (phase rate corrected), `aggregate.py` | `outputs/definedness_summary.csv`, `outputs/dispersion_summary.csv` |
| Sec. VI.C, Fig. 5 dotted curve | phase-scrambling form, rate √(ε²+4Δ²)/v | closed form in `code/make_figs.py` | — |
| Sec. VI.C, Supp. S4, Fig. S1 | fraction of draws reaching m = −0.5; 50% points 1.405 [1.398, 1.413] (24 000 samples), 1.342 [1.334, 1.350] (1200), 1.396 [1.389, 1.404] (root-finding) | `diag_definedness.py`, `diag_continuous.py`, `aggregate.py` (binomial GLM, 2000-replicate percentile bootstrap, seed 20260925) | `outputs/definedness_summary.csv`, `outputs/definedness_continuous.csv` |
| Sec. VI.C | settled-state range −0.40802 to −0.40786 | `diag_definedness.py`, `aggregate.py` | `outputs/definedness_summary.csv` (`m_end_mean`) |
| Sec. VI.D | rival offset −0.2134 ± 0.0002; shape term and correlation | `diag_dbmatch.py`, `aggregate.py` | `outputs/diag_dbmatch.npz` |
| Supp. S3, Table S1 | calibration of the detailed-balance rival: best node c′_γ = 10^−0.6, c_β = 1, SSE 67.03 | `stage3_calibrate.py` (uses `stage3_models.py`, `stage3_runs.py`, focal targets in `s3/`); example: `calibration_example.py` | `s3/focal_*.npz` |
| Supp. S2 | endpoint depends on c_β c′_γ only | `code/v2/endpoint_check.py` | — |

Checks, run from this folder:
- `python smoke_test.py`: 15/15 pass, about 20 s.
- `python calibration_example.py`: SSE 67.034011 against the stored 67.034, about 30 s.
- `python aggregate.py`: rebuilds the summary tables from the archives, with worst deviation 4.8×10⁻⁹, and reproduces the three bootstrap intervals; about 15 s. Requires statsmodels.
