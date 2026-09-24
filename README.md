# Aggregation and observability of quantum-like decision structure

Code and derived data for:

> R. J. Brunner, *Population heterogeneity and the observability of quantum-like decision structure: aggregation, reciprocity, and finite-sample detectability* (manuscript, 2026).

The paper asks which constraints of quantum-like decision models survive when respondents are pooled into survey tables. This repository reproduces its numerical results: the likelihood-ratio region test on the 73 published question-order datasets (Sec. V), the detectability check (Sec. IV), and all five figures. The dynamical simulations of Secs. VI and VII are in `simulation/`.

## Contents

| path | what it is |
|---|---|
| `code/region_lr.py` | the two constrained maximum likelihoods and the statistic Λ = 2(ℓ_QQ − ℓ_P) |
| `code/parse_s1.py` | step 1: reconstructs both 2×2 tables of each dataset from the published moments |
| `code/recover_n.py` | step 2: recovers per-order sample sizes from the published QQ χ² |
| `code/run_region_test.py` | step 3: Λ for all 73 datasets |
| `code/boot72.py`, `code/summarize_boot.py` | step 4: parametric bootstrap for the one dataset with appreciable Λ |
| `code/check_detectability.py` | independent Monte Carlo check of the power boundary, with its saved output |
| `code/make_figs.py` | regenerates `figures/` from `data/` |
| `data/region_test/` | reconstructed tables, recovered sample sizes, Λ per dataset, bootstrap replicates |
| `data/cs_simulation/` | summary tables of the dynamical simulations (dispersion sweep, definedness curve, pooled reciprocity difference) |
| `docs/region_test_plan_v2.md` | the analysis plan and decision rule, written before the test was run on the data |
| `docs/region_test_plan_v1_superseded.md` | the first plan, superseded because it conflated QQ failure with the projective constraint (see v2) |
| `simulation/` | dynamical simulation code (Secs. VI–VII) |

## Reproducing the region test

```
pip install -r requirements.txt
cd code
python parse_s1.py path/to/S1.pdf     # checks dataset 69 against the published table
python recover_n.py
python run_region_test.py            # about two minutes
python summarize_boot.py             # uses the stored bootstrap replicates
python make_figs.py
```

`S1.pdf` is Supplementary Information S1 of Dzhafarov, Zhang and Kujala, *Phil. Trans. R. Soc. A* **374**, 20150099 (2016), available in the source bundle of arXiv:1504.07422. It is not redistributed here. `parse_s1.py` requires `pdftotext` (poppler). The atom sample in `region_lr.py` uses a fixed seed, and the pipeline reproduces the stored `region_lr_results.csv` exactly. To regenerate the bootstrap replicates for dataset 72, run `python boot72.py 72 200 2`, `python boot72.py 72 520 3`, `python boot72.py 72 540 4` and `python boot72.py 72 540 5` (about 15 minutes in total).

## Licenses

Code: MIT (see `LICENSE`). Derived data and figures: CC BY 4.0. The underlying survey moments are those published by Dzhafarov, Zhang and Kujala (2016) from data assembled by Wang, Solloway, Shiffrin and Busemeyer, *Proc. Natl. Acad. Sci. USA* **111**, 9431 (2014); cite those works when using `data/region_test/`.

## Citation

See `CITATION.cff`. The archived release has a DOI from Zenodo.
