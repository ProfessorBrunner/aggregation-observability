# Which quantum-like constraints survive disorder averaging?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22964616.svg)](https://doi.org/10.5281/zenodo.22964616)

Code and derived data for:

> R. J. Brunner, *Which quantum-like constraints survive disorder averaging? Reciprocity and observability in heterogeneous populations* (manuscript submitted to Physical Review E, 2026).

## Release v1.1

v1.1 accompanies the submitted manuscript. It supersedes the v1.0 region test in three ways:

- **Exact solver.** The likelihood-ratio statistic Λ = 2(ℓ_QQ − ℓ_P) is computed with an exact linear oracle for the qubit moment set, and every fit is certified by its Frank–Wolfe duality gap. v1.0 used a sampled oracle whose small positive Λ values were optimizer residue.
- **Abortion pair from its published table.** Dataset 72 is taken from Schuman, Presser and Ludwig (1981), Table 2, because two of its published moments are exchanged relative to the definition of Dzhafarov, Zhang and Kujala (2016).
- **Robustness to the coding question.** A check of every row under the exchanged reading of the moments.

The v1.0 scripts and results remain in `code/` and `data/region_test/` as the record of that analysis.

**Result:** Λ = 0, certified, for all 72 datasets (and for the excluded Rose–Jackson pair).

## Contents

| path | what it is |
|---|---|
| `code/v2/region_exact.py` | exact linear oracle, QQ-constrained and projective-class maximum likelihoods, Λ with duality gap |
| `code/v2/boundary_fit.py` | direct mixture fit for optima on the hull boundary, certified by the same gap |
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
| `data/region_test_v2/` | v1.1 results, including certified Λ, coding check, implied covariances and bootstrap record |
| `data/cs_simulation/` | summary tables of the dynamical simulations |
| `docs/` | the analysis plans; v2 was written before the test reported in the paper was run |
| `simulation/` | dynamical simulation code |

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
