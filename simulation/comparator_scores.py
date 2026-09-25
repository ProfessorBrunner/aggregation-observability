"""Recompute each calibrated rival's loss (Supp. S3, Eq. (S6)) at its selected node, from
calibration_records/selected_nodes.json, with the same targets, scales, seeds and objective as
stage3_calibrate.py. The per-node grid losses were not archived, so this reproduces the Table S1 values
at the selected nodes but not the ordering of all grid nodes. Runtime: a few minutes."""
import json, numpy as np
import stage3_calibrate as C
sel=json.load(open('calibration_records/selected_nodes.json'))['models']
tg=C.focal_targets(); scale={}
for k in ("m_end","m_mean","m_var","T4"):
    v=np.array([t[k] for t in tg.values()],float); v=v[np.isfinite(v)]; scale[k]=float(v.std()) if v.std()>1e-12 else 1.0
for m in ("db","two_state","lag","rtip","ising"):
    par={k:float(v) for k,v in sel[m]['parameters'].items()}
    L=C.objective(m,par,tg,scale)
    print(f"{m:10s} params {par}  loss {L:.4f}  Table S1 {sel[m].get('table_s1_value')}")
