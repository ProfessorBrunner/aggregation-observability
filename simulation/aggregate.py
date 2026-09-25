"""
aggregate.py -- rebuild the summary tables and the derived statistics from the
stored per-draw archives.

  python aggregate.py             # rebuild and verify against outputs/*.csv
  python aggregate.py --write     # also overwrite outputs/*.csv
  python aggregate.py --merge     # merge fresh shard dirs into outputs/diag_*.npz

WHY THIS SCRIPT EXISTS.  The diag_* scripts write per-draw archives
(outputs/diag_*.npz).  The four summary CSVs and the threshold statistics
quoted in the memos were originally assembled interactively, not by any
script, so the eight diagnostic scripts alone did not reproduce them.  This
script closes that gap: it recomputes every summary column and every derived
number from the archives and reports the deviation from the shipped CSVs.
It runs in about a minute (the 2000-replicate bootstrap dominates) and does
not integrate anything.

ONE KNOWN DEPARTURE.  dispersion_summary.csv v2 carries the CORRECTED phase
predictor Phi' = Omega/v (see diag_dispersion.py, CORRECTED 2026-09-25).
diag_dispersion.npz was written before that correction, so its stored
pred[:, 4] column holds the superseded eps/(2v) form and is ignored here; the
phase column is recomputed by diag_dispersion.phase_prediction().  The
scal[8] entry ("s_thresh_phase" = 3.72043376) in the archive is superseded for
the same reason; the corrected value is 1.6986.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import diag_dispersion as DP

Z1959 = 1.959963984540054          # normal quantile for a 95% Wilson interval
OUT = "outputs"


def nanagg(f, v, **kw):
    """Aggregate ignoring NaN; returns NaN for an all-NaN slice without warning."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return f(v, **kw) if v.size > (1 if kw.get("ddof") else 0) else np.nan


def cols(z):
    return {str(c): i for i, c in enumerate(z["cols"])}


def wilson(k, n, z=Z1959):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = (z / (1 + z2 / n)) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def fit50(s, k, n):
    """Binomial GLM on logit(p) = a + b s; returns the 50% point -a/b."""
    import statsmodels.api as sm
    m = sm.GLM(np.column_stack([k, n - k]), sm.add_constant(s),
               family=sm.families.Binomial()).fit()
    a, b = m.params
    return -a / b, m


def bootstrap50(s, A, rng, B=2000):
    """Percentile bootstrap of the 50% point, resampling draws within each
    dispersion independently.  A has shape (n_sd, n_draws) of 0/1 outcomes.
    The rng is passed in and SHARED across the three detectors, in the order
    event, 24000, 1200, because that is the stream the memo's intervals came
    from; giving each detector a freshly seeded stream shifts the last two
    intervals in the fourth decimal."""
    n = np.full(len(s), A.shape[1])
    out = []
    for _ in range(B):
        kb = np.array([A[i][rng.integers(0, A.shape[1], A.shape[1])].sum()
                       for i in range(len(s))])
        try:
            v, _ = fit50(s, kb, n)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            pass
    v = np.array(out)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return lo, hi, v


# shard directory -> merged archive, for the four sharded diagnostics.  The
# diag_* scripts write one npz per shard; the merged archives shipped in
# outputs/ are the concatenation in shard order.
SHARDS = {"s3disp": ("diag_definedness.npz", "def_"),
          "s3cont": ("diag_continuous.npz", "cont_"),
          "s3g2": ("diag_g2.npz", "g2_"),
          "s3sd1": ("diag_sd1.npz", "sd1_")}


def merge_shards():
    import glob
    for d, (out, pre) in SHARDS.items():
        fs = sorted(glob.glob(os.path.join(d, pre + "*.npz")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        if not fs:
            print(f"  {d}/: no shards, skipped")
            continue
        Z = [np.load(f) for f in fs]
        rows = np.vstack([z["rows"] for z in Z])
        np.savez_compressed(os.path.join(OUT, out), rows=rows, cols=Z[0]["cols"])
        print(f"  {d}/: {len(fs)} shards -> {OUT}/{out}  ({rows.shape[0]} rows)")


def wr(name, hdr, rows, write):
    txt = ",".join(hdr) + "\n" + "\n".join(
        ",".join(("" if not np.isfinite(x) else f"{x:.10g}") if isinstance(x, float)
                 else str(x) for x in r) for r in rows) + "\n"
    ref = os.path.join(OUT, name)
    if os.path.exists(ref):
        R = [l.split(",") for l in open(ref).read().strip().split("\n")]
        assert R[0] == hdr, f"{name}: header mismatch\n stored {R[0]}\n rebuilt {hdr}"
        A = np.array([[float(x) if x else np.nan for x in r] for r in R[1:]])
        Bm = np.array([[float(x) if (isinstance(x, str) and x) or isinstance(x, (int, float))
                        else np.nan for x in r] for r in rows], float)
        assert A.shape == Bm.shape, f"{name}: shape {A.shape} vs {Bm.shape}"
        bad = np.isfinite(A) ^ np.isfinite(Bm)
        assert not bad.any(), (f"{name}: {int(bad.sum())} cell(s) finite in one copy and "
                               f"not the other, columns "
                               f"{sorted({hdr[c] for c in np.nonzero(bad)[1]})}")
        d = np.abs(A - Bm)
        d[~np.isfinite(A) & ~np.isfinite(Bm)] = 0.0
        j = int(np.nanargmax(d))
        print(f"  {name:32s} {Bm.shape[0]:3d} rows   max|rebuilt-stored| = "
              f"{np.nanmax(d):.3e}  (column '{hdr[j % len(hdr)]}')")
    else:
        print(f"  {name:32s} {len(rows):3d} rows   (no stored copy to compare)")
    if write:
        open(ref, "w").write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    w = a.write
    if a.merge:
        merge_shards()

    ZD = np.load(os.path.join(OUT, "diag_dispersion.npz"))
    ZB = np.load(os.path.join(OUT, "diag_dbmatch.npz"))
    ZF = np.load(os.path.join(OUT, "diag_definedness.npz"))
    ZG = np.load(os.path.join(OUT, "diag_g2.npz"))
    ZC = np.load(os.path.join(OUT, "diag_continuous.npz"))

    # ---- scalars of the reference curve -------------------------------------
    E, R = ZD["E"].astype(float), ZD["R"].astype(float)
    j = int(np.argmin(R))
    U0, eps1 = DP.M_INF - R[j], E[j]
    curv = DP.dip_curvature(E, R, eps1)
    th_conv = DP.root_of(lambda s: DP.M_INF - DP.convolve_prediction(E, R, s)[0], 0.05, 4.0)
    th_curv = DP.root_of(lambda s: DP.M_INF - (R[j] + 0.5 * s * s * curv), 0.05, 4.0) \
        if True else np.nan
    om1 = np.sqrt(eps1 ** 2 + 4.0 * DP.DELTA ** 2)
    th_phase = np.sqrt(-2.0 * np.log(DP.U_STAR / U0)) * DP.V / om1
    print("reference curve (single agent, sd = 0):")
    print(f"  m_inf = {DP.M_INF:.6f}   U* = {DP.U_STAR:.6f}   U0 = {U0:.6f}")
    print(f"  eps1 = {eps1:.4f}   Omega(eps1) = {om1:.6f}   Phi'(eps1) = {om1/DP.V:.4f}"
          f"   |R''| = {curv:.6f}")
    print(f"  threshold sd/Delta:  convolution {th_conv:.4f}   curvature {th_curv:.4f}"
          f"   phase {th_phase:.4f}  (superseded stored phase {ZD['scal'][8]:.4f})")

    # ---- dispersion_summary.csv --------------------------------------------
    CD, CB = cols(ZD), cols(ZB)
    D, Bm = ZD["rows"], ZB["rows"]
    P = ZD["pred"]
    S = np.unique(np.round(D[:, CD["sd"]], 6))
    hdr = ("sd_over_Delta,U_mean,U_sd,U_min,U_max,U_pred_convolution,U_pred_curvature,"
           "U_pred_phase,sd_finiteN_pred,resid_meas_minus_conv,frac_T4_defined,"
           "frac_T4_defined_frozen_lattice,focal_m_min_mean,focal_m_end_mean,"
           "eps_at_min_mean,db_offset_post,db_shape_post,db_rms_post,db_maxabs,"
           "db_corr_post,db_m_end_mean").split(",")
    rows = []
    for s in S:
        m = np.isclose(D[:, CD["sd"]], s)
        mb = np.isclose(Bm[:, CB["sd"]], s)
        p = P[np.isclose(P[:, 0], s)][0]
        U = D[m, CD["f_U"]]
        Uc = float(p[1])
        rows.append([s, U.mean(), U.std(ddof=1) if m.sum() > 1 else 0.0, U.min(), U.max(),
                     Uc, U0 - 0.5 * s * s * curv,   # not clamped: goes negative above 1.3388
                     float(DP.phase_prediction(U0, eps1, s)), float(p[3]),
                     U.mean() - Uc,
                     float(np.mean(np.isfinite(D[m, CD["f_U"]]) & (U > DP.U_STAR))),
                     float(np.mean((DP.M_INF - D[m, CD["f_mmin_1200"]]) > DP.U_STAR)),
                     D[m, CD["f_mmin"]].mean(), D[m, CD["f_mend"]].mean(),
                     D[m, CD["f_eps_at_min"]].mean(),
                     Bm[mb, CB["off_post"]].mean(), Bm[mb, CB["shape_post"]].mean(),
                     Bm[mb, CB["rms_post"]].mean(), Bm[mb, CB["maxabs"]].mean(),
                     Bm[mb, CB["corr_post"]].mean(), D[m, CD["db_mend"]].mean()])
    print("\nrebuilt tables:")
    wr("dispersion_summary.csv", hdr, rows, w)

    off = Bm[:, CB["off_post"]]
    print(f"  Null DB offset, pooled over all {off.size} draws: "
          f"{off.mean():.5f} +- {off.std(ddof=1):.5f}  "
          f"(range {off.min():.5f} .. {off.max():.5f})")

    # ---- definedness_summary.csv -------------------------------------------
    CF = cols(ZF)
    F = ZF["rows"]
    SF = np.unique(np.round(F[:, CF["sd"]], 6))
    hdr = ("sd_over_Delta,n_draws,k_defined_24000,frac_defined_24000,wilson_lo_24000,"
           "wilson_hi_24000,k_defined_1200,frac_defined_1200,wilson_lo_1200,"
           "wilson_hi_1200,U_mean,U_sd,U_pred_convolution,resid_meas_minus_conv,"
           "sd_finiteN_pred,U_mean_1200,m_end_mean,m_end_sd,db_gap_min_all,"
           "db_gap_min_resolvable,db_viol_all,db_viol_resolvable,db_dy_max,"
           "db_m_end_mean").split(",")
    rows = []
    for s in SF:
        m = np.isclose(F[:, CF["sd"]], s)
        n = int(m.sum())
        k1, k2 = int(F[m, CF["defined"]].sum()), int(F[m, CF["defined_1200"]].sum())
        l1, h1 = wilson(k1, n)
        l2, h2 = wilson(k2, n)
        cmin, _e, sdN = DP.convolve_prediction(E, R, s)
        Uc = DP.M_INF - cmin
        rows.append([s, n, k1, k1 / n, l1, h1, k2, k2 / n, l2, h2,
                     F[m, CF["U"]].mean(), F[m, CF["U"]].std(ddof=1),
                     Uc, F[m, CF["U"]].mean() - Uc, sdN,
                     F[m, CF["U_1200"]].mean(),
                     F[m, CF["m_end"]].mean(), F[m, CF["m_end"]].std(ddof=1),
                     F[m, CF["db_gap_min"]].min(), F[m, CF["db_gap_min_res"]].min(),
                     F[m, CF["db_viol"]].sum(), F[m, CF["db_viol_res"]].sum(),
                     F[m, CF["db_dy_max"]].max(), F[m, CF["db_m_end"]].mean()])
    wr("definedness_summary.csv", hdr, rows, w)

    # ---- g2_summary.csv -----------------------------------------------------
    CG = cols(ZG)
    G = ZG["rows"]
    hdr = ("x_read,sd_over_Delta,n_draws,cbar,qbar,cov_cq,var_c,var_q,G2_pred_mean,"
           "G2_meas_mean,wbar,agree_arm1,agree_arm2,frac_offres,G2_sd,G2_min,G2_max,"
           "G2_max_absdiff_pred_meas,max_abs_pooled_qq").split(",")
    rows = []
    for xr in np.unique(np.round(G[:, CG["x_read"]], 6)):
        for s in np.unique(np.round(G[:, CG["sd"]], 6)):
            m = np.isclose(G[:, CG["x_read"]], xr) & np.isclose(G[:, CG["sd"]], s)
            if not m.any():
                continue
            g = G[m, CG["G2_meas"]]
            rows.append([xr, s, int(m.sum()),
                         G[m, CG["cbar"]].mean(), G[m, CG["qbar"]].mean(),
                         G[m, CG["cov_cq"]].mean(), G[m, CG["var_c"]].mean(),
                         G[m, CG["var_q"]].mean(), G[m, CG["G2_pred"]].mean(), g.mean(),
                         G[m, CG["wbar"]].mean(), G[m, CG["agree_arm1"]].mean(),
                         G[m, CG["agree_arm2"]].mean(), G[m, CG["frac_offres"]].mean(),
                         g.std(ddof=1), g.min(), g.max(),
                         G[m, CG["G2_absdiff"]].max(),
                         np.abs(G[m, CG["qq_pooled"]]).max()])
    wr("g2_summary.csv", hdr, rows, w)

    # ---- definedness_continuous.csv ----------------------------------------
    CC = cols(ZC)
    C = ZC["rows"]
    key24 = {(round(r[CF["sd"]], 4), int(r[CF["seed"]])): (r[CF["defined"]],
                                                           r[CF["defined_1200"]])
             for r in F}
    hdr = ("sd_over_Delta,n_draws,k_crossed_event,frac_event,wilson_lo,wilson_hi,"
           "frac_24000_scan,frac_1200_scan,n_discordant_event_vs_24000,"
           "t_cross_first_mean,t_cross_first_sd,n_downward_crossings_mean,"
           "n_downward_crossings_max,m_min_sampled_mean,"
           "m_at_first_root_maxdev").split(",")
    rows, A = [], {"event": [], "24000": [], "1200": []}
    for s in np.unique(np.round(C[:, CC["sd"]], 6)):
        m = np.isclose(C[:, CC["sd"]], s)
        n = int(m.sum())
        ev = C[m, CC["crossed"]]
        d24 = np.array([key24[(round(r[CC["sd"]], 4), int(r[CC["seed"]]))][0] for r in C[m]])
        d12 = np.array([key24[(round(r[CC["sd"]], 4), int(r[CC["seed"]]))][1] for r in C[m]])
        A["event"].append(ev); A["24000"].append(d24); A["1200"].append(d12)
        lo, hi = wilson(int(ev.sum()), n)
        fr = C[m, CC["m_at_first_root"]]
        rows.append([s, n, int(ev.sum()), ev.mean(), lo, hi, d24.mean(), d12.mean(),
                     int((ev != d24).sum()),
                     nanagg(np.nanmean, C[m, CC["t_cross_first"]]),
                     nanagg(np.nanstd, C[m, CC["t_cross_first"]], ddof=1),
                     C[m, CC["n_cross"]].mean(), C[m, CC["n_cross"]].max(),
                     C[m, CC["m_min_sampled"]].mean(),
                     nanagg(np.nanmax, np.abs(fr + 0.5))])
    wr("definedness_continuous.csv", hdr, rows, w)

    # ---- logistic 50% points with bootstrap --------------------------------
    SC = np.unique(np.round(C[:, CC["sd"]], 6))
    nn = np.full(len(SC), 256)
    print("\nlogistic 50% points (binomial GLM, 2000-replicate percentile bootstrap,"
          " rng seed 20260925):")
    rng = np.random.default_rng(20260925)
    for lab in ("event", "24000", "1200"):
        M_ = np.array(A[lab], float)
        p50, mod = fit50(SC, M_.sum(axis=1), nn)
        lo, hi, bv = bootstrap50(SC, M_, rng)
        print(f"  {lab:6s} 50% = {p50:.4f}  [{lo:.4f}, {hi:.4f}]   slope {mod.params[1]:.4f}"
              f"   deviance {mod.deviance:.1f} on {int(mod.df_resid)} df  (n_boot {bv.size})")
    print(f"  convolution prediction (not a fit): {th_conv:.4f}")

    # ---- G2 scatter crossover ----------------------------------------------
    print("\nG2 at x_read = 10 Delta, mean against across-draw scatter:")
    m10 = np.isclose(G[:, CG["x_read"]], 10.0)
    for s in np.unique(np.round(G[m10, CG["sd"]], 6)):
        m = m10 & np.isclose(G[:, CG["sd"]], s)
        g = G[m, CG["G2_meas"]]
        if 1.15 <= s <= 1.45:
            print(f"  sd/Delta {s:.2f}   mean {g.mean():.6e}   sd {g.std(ddof=1):.6e}"
                  f"   {'mean > scatter' if abs(g.mean()) > g.std(ddof=1) else 'SCATTER EXCEEDS MEAN'}"
                  f"   frac positive {np.mean(g > 0):.3f}")


if __name__ == "__main__":
    main()
