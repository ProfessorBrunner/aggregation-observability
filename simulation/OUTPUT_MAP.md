# simulation_v12 — what each script and output supports

Two things in this file are asserted rather than measured, and both are flagged
as such: the identity of "S1" and "Table S1", and the intended figure homes for
the dispersion material.  Everything else is a number read back from the
shipped outputs.

## A. Attribution

| Script | Output | Supports |
|---|---|---|
| `stage3_models.py` | — | the dynamics itself: focal model and the five classical nulls; every number below is one of its integrations |
| `stage3_runs.py` | — | the frozen 216-cell sweep, the 75 training cells, the eight population draws `(3141 … 3148)` |
| `stage3_calibrate.py` | `s3cal/db_*.npz` | the null-calibration grid: Null DB best node `c_gamma = 0.25118864`, `c_beta = 1.0`, SSE `67.034`, interior on both axes, against `549.93 … 566.67` for the frozen four nulls |
| `diag_dispersion.py` | `diag_dispersion.npz`, `dispersion_summary.csv`, `dispersion_perdraw.csv` | undershoot `U` against dispersion, 17 dispersions x 8 draws; the exact `kappa = 0` reduction `m = (G_s * R)(xbar)`; single-agent `m_min = -0.708449`, `U0 = 0.277668`, `eps1 = 4.4802`, `|R''(eps1)| = 0.309810`; convolution threshold `1.3858`; endpoint invariance (`m_end` spread `1.8e-04` while `U` collapses from `0.277668` to `0.004593`) |
| `diag_dispersion.py` (same run) | `dispersion_summary.csv` cols `U_pred_*` | the three predictors and their ranges of validity: convolution correct throughout; curvature better to about 0.9 and negative above 1.3388; phase better from 1.1 upward; corrected phase rate `Phi' = Omega/v`, `Phi'(eps1) = 0.9813`, so `Phi' s = 1` at `sd/Delta = 1.0191` |
| `diag_dbmatch.py` | `diag_dbmatch.npz`, `dispersion_summary.csv` cols `db_*` | **the offset −0.2134**: pooled level offset `-0.21343 +- 0.00019` over all 136 draws (range `-0.21372 … -0.21303`), dispersion-independent; shape gap falls 4.47x (`0.046298 -> 0.010360`) and correlation rises `0.7535 -> 0.9861` while total rms barely moves (`0.21801 -> 0.21373`), which is why rms alone is not a usable match statistic |
| `diag_definedness.py` | `diag_definedness.npz`, `definedness_summary.csv` | T4 definedness as a smooth sigmoid, not a cliff: `1.000` to `sd/Delta = 1.1`, `0.4922 [0.4315, 0.5531]` at 1.4, `0.0664 [0.0419, 0.1038]` at 1.6, `1/256` at 1.8 (Wilson 95%); the frozen 1200-sample lattice understates undershoot by `0.00303 … 0.01171` and moves the 50% point `1.4054 -> 1.3421` |
| `diag_definedness.py` (same run) | `definedness_summary.csv` cols `db_*` | the Null DB monotonicity audit: strict `y > y*` fails as literally stated (`min = -2.220446e-16`, 42.48% of `5.212e+09` agent-samples `<= 0`), but every violation sits at `\|eps\| >= 35.9090`, outside the tanh saturation boundary `35.2327`; on the resolvable set `min = +3.330669e-16`, zero violations in 0 of 5376 runs |
| `diag_sd1.py` | `diag_sd1.npz` | the `sd/Delta = 1.0` point the grids skip: `U = 0.144400` (sd `0.014578`, range `0.106171 … 0.180570`) against convolution `0.143872`; `m_min = -0.575181`; defined 256/256 on both lattices |
| `diag_continuous.py` | `diag_continuous.npz`, `definedness_continuous.csv` | the crossing located by root-finding on the dense output: logistic 50% point `1.3962 [1.3885, 1.4035]` against `1.4054 [1.3981, 1.4130]` for the 24000-sample scan and `1.3421 [1.3344, 1.3495]` for the 1200-sample lattice; root-finding detects **fewer** crossings than dense sampling (44 of 5376 discordant, all one-directional), because the event is tested at solver step endpoints; multiple downward crossings in 1274 of 5376 draws |
| `diag_g2.py` | `diag_g2.npz`, `g2_summary.csv` | the pooled reciprocity difference `G2 = Cov(c,q)/[qbar(1-qbar)]`: exact against the pooled table to `3.331e-16` over 9408 readouts and against the explicit 100-agent Lüders update to `1.439e-16`; `G2 = 1.943e-31` at `sd(eta) = 0`; an aggregation artifact, not a quantum signature — a classical mixture with the same `(c_i, q_i)` marginals gives the identical pooled table; pooled QQ agreement stays exactly intact (`6.661e-16`) while reciprocity breaks |
| `diag_g2.py` (same run) | `g2_summary.csv` | readout dependence: at `x_read = 2 Delta`, `G2 = -0.302935` at `sd/Delta = 3` and `-7.842e-02 +- 1.138e-02` at 1.1 with 0/64 draws wrong-signed; at `10 Delta` it is `~1e-04`, peaks `2.4499e-04` at 0.7, and across-draw scatter first exceeds the mean at `sd/Delta = 1.30`; at the frozen exit readout it is `1e-13 … 1e-12`, identically zero |
| `aggregate.py` | the four summary CSVs, the threshold statistics | rebuilds every summary column from the archives and verifies it against the shipped copies; supplies the logistic 50% points, the bootstrap intervals, the pooled offset and the G2 crossover |
| `smoke_test.py` | stdout, 15 checks | that the shipped code still produces the stored numbers |
| `calibration_example.py` | stdout | `objective("db", best node) = 67.034011` against the stored `67.034` |

## B. Figures and tables: what the record actually says

Checked, not assumed.  The manuscript of record is
`qc_bubbles_manuscript_v0_7_rev5.tex` with `qc_bubbles_manuscript_v0_8_3.pdf`
as the compiled form.  In the `.tex` there are exactly six
`\includegraphics` calls and one table:

| Float | Caption opens | Drawn by |
|---|---|---|
| Fig. 1 | the two decision frames and the avoided crossing | not in this package |
| Fig. 2 | validity region of the local expansion | not in this package |
| Fig. 3 | boundary scaling refutes the pre-registered exponent | `make_fig3_twopanel.py`, `make_tranche_figures.py` |
| **Fig. 4** | **three exact limits of the master equation** | `make_tranche_figures.py`, from `anchor_limits_d.npz` |
| **Fig. 5** | **conditional collapse for the focal model; the nulls saturate out** | `make_tranche_figures.py`, from `stage3_results.npz` |
| Fig. 6 | exit morphology on the common pair set | `make_tranche_figures.py`, from `fig6_data.npz` |
| Table 1 | exit time `T_4` in units of `tau_c` on the twelve common pairs | `make_tranche_figures.py` / `stage3_rulings.npz` |

Three consequences for the request as written:

1. **Figs. 4 and 5 are not produced by anything in this package.**  They are
   drawn by `make_tranche_figures.py` from five aggregate archives
   (`g2_map_results.npz`, `anchor_limits_d.npz`, `stage3_results.npz`,
   `stage3_rulings.npz`, `fig6_data.npz`) plus `qc_style.py`; none of those is
   in the eight-script list, and shipping the script without them would not
   run.  What this package does contain is the dynamics **behind** Fig. 5 and
   Fig. 6 — `stage3_models.py` and the frozen sweep — and the calibration that
   fixes the Null DB curve appearing in Fig. 5.
2. **There is no "S1", no "Table S1", and no supplement.**  `Table S1`,
   `S1`, and `Supplement` each return zero matches in the `.tex` and zero in
   the text of the v0.8.3 PDF.  Two files that could be meant exist:
   `fig_s3_01.pdf` and `fig_s3_02.pdf`, produced by `make_stage3_figures.py`
   from `s3/*.npz` (the full 72-shard set, of which only the 24 focal shards
   are shipped here).  A null-calibration table is likewise not a float
   anywhere; the calibration numbers appear only in running text.
3. **The dispersion, definedness and G2 material has no figure home at all.**
   It postdates v0.8.3 and no float has been cut for it.

**R74 (queued, not resolved).**  Which floats is the new material intended to
occupy, and does "S1"/"Table S1" denote `fig_s3_01` and a calibration table to
be created?  Until that is ruled, the attribution table in §A is stated
against memo sections and numbers rather than against figure numbers, and no
row here claims to produce Fig. 4, Fig. 5, or an S-numbered float.

## C. How the pooled-reciprocity readout is taken

`c_i` is evaluated at exactly `x_read` (`eps_i = x_read + eta_i`), while `q_i`
is read from the nearest saved output sample at or after `x_read`, so the two
factors of `Cov(c, q)` are not taken at the same field.

The gap is draw-specific, not a fixed offset: the integration end time varies
with the draw, so the `linspace(0, T, 24000)` output grid does too.  Over the
192 draws re-run for the check below the readout sample lands anywhere in
`x = 2.00007 … 2.06801`, i.e. it scatters across the whole saved-sample
spacing of `0.06724` in `x`.  (`x = 2.00924425` is the sample for
`sd/Delta = 1.1`, seed 900000, quoted earlier as an instance, not as a common
value.)

**Measured effect.**  Recomputed at `x_read = 2 Delta` with `c_i` moved onto
the same saved sample as `q_i`, on the same 64 seeds.  The per-agent `c_i`,
`q_i` are not retained in `diag_g2.npz`, so this required re-running the three
dispersions (192 focal integrations); the stored convention was recomputed in
the same pass and reproduces `g2_summary.csv` to `5e-07`.

| `sd(eta)/Delta` | stored (c at `x_read`) | same field | change | as % of stored | across-draw scatter |
|---|---|---|---|---|---|
| 0.3 | −0.00586702 | −0.00572167 | +1.4534e-04 | +2.48% | 7.937e-04 |
| 1.1 | −0.0784200 | −0.0770626 | +1.3574e-03 | +1.73% | 1.1216e-02 |
| 3.0 | −0.302935 | −0.300973 | +1.9626e-03 | +0.65% | 3.1729e-02 |

Largest absolute change `+1.963e-03`, at `sd/Delta = 3.0`; largest relative
change `+2.48%`, at `sd/Delta = 0.3`.  The change is toward zero at every
dispersion, and at every dispersion it is smaller than the across-draw scatter
(by factors of 5.5, 8.3 and 16.2), so no reported `G2` value or sign changes
under the consistent readout.  Largest single-draw change `4.634e-03`.

Produced by `diag_g2_readout.py`; per-dispersion results in
`outputs/g2_readout_check.csv`.
