# Dynamical simulations (Secs. VI and VII)

This folder holds the simulation code that produced `data/cs_simulation/`:

- `stage3_models.py`: agent model and integrator (coherent model and the detailed-balance rival)
- `stage3_calibrate.py`: calibration of the rival (Supplemental Material, Sec. S3)
- `diag_dispersion.py`, `diag_dbmatch.py`, `diag_sd1.py`: undershoot against dispersion and the rival comparison (Sec. VII)
- `diag_definedness.py`: definedness curve at 256 draws (Sec. VII)
- `diag_g2.py`: pooled reciprocity difference of a dynamical population (Sec. VI)

Common settings: N = 100 agents, Δ = 1, z = 0.2, κ = 0, γ_φ = 0.02Δ, start at detuning −320Δ, DOP853 at rtol 1e-7, atol 1e-9.
