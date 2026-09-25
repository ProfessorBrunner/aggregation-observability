"""Decompose the calibration loss (Supp. S3, Eq. (S6)) at the detailed-balance rival's selected node.
Same targets, scales, seeds and simulations as stage3_calibrate.objective; reports each term and the
timing-residual bookkeeping. Runtime: about 30 s."""
import numpy as np
import stage3_calibrate as C, stage3_models as M
BEST={"c_gamma":10**-0.6,"c_beta":1.0}
tg=C.focal_targets(); scale={}
for k in ("m_end","m_mean","m_var","T4"):
    v=np.array([t[k] for t in tg.values()],float); v=v[np.isfinite(v)]; scale[k]=float(v.std()) if v.std()>1e-12 else 1.0
cl=C.cells(); terms={"end":0.0,"mean":0.0,"var":0.0,"T":0.0}; t_focal=t_used=t_dropped=0
for i,tv in tg.items():
    c=cl[i]; vals=[M.simulate("db",c[0],c[1],c[2],c[3],s,BEST) for s in C.CAL_SEEDS]
    g=lambda k: float(np.nanmean([v[k] for v in vals]))
    w=2.0 if (c[1]>=1.0 or c[2]==0.0) else 1.0
    terms["end"]+=w*((g("m_end")-tv["m_end"])/scale["m_end"])**2
    terms["mean"]+=((g("m_mean")-tv["m_mean"])/scale["m_mean"])**2
    terms["var"]+=((g("m_var")-tv["m_var"])/scale["m_var"])**2
    if np.isfinite(tv["T4"]):
        t_focal+=1
        if np.isfinite(g("T4")): t_used+=1; terms["T"]+=((g("T4")-tv["T4"])/scale["T4"])**2
        else: t_dropped+=1
print("training cells:",len(tg))
print("loss terms: "+", ".join(f"{k} {v:.4f}" for k,v in terms.items())+f"; total {sum(terms.values()):.6f}")
print(f"timing target finite for the coherent model: {t_focal}; residual included: {t_used}; omitted (rival has no crossing): {t_dropped}")
