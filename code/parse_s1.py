"""Step 1. Reconstruct the two 2x2 tables for each of the 73 datasets.

Input: S1.pdf, Supplementary Information S1 of Dzhafarov, Zhang & Kujala,
Phil. Trans. R. Soc. A 374, 20150099 (2016), arXiv:1504.07422 (not redistributed here;
obtain it from the arXiv source bundle). Requires `pdftotext` (poppler).
Output: ../data/region_test/tables73.csv

Per order, P(yes) = (1 + <.>)/2 and P(yes,yes) = (<VW> + 2 P_A + 2 P_B - 1)/4.
Check: dataset 69 reproduces the table printed in the DZK main text, P(yes,yes) = 0.3987.
"""
import re, subprocess, sys, numpy as np, pandas as pd
pdf = sys.argv[1] if len(sys.argv) > 1 else "S1.pdf"
txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
for ch in ["\xad"]: txt = txt.replace(ch, "")
for ch in ["\u2010", "\u2011", "\u2212"]: txt = txt.replace(ch, "-")
while "--" in txt: txt = txt.replace("--", "-")
rows = []
for line in txt.splitlines():
    t = line.split()
    if len(t) >= 14 and re.match(r"^-?\d\.\d{4}$", t[0]):
        rows.append(list(map(float, t[:13])) + [int(t[13]), " ".join(t[14:])])
cols = ["V1","W1","W2","V2","V1W2","V2W1","s1","ordeff","dC","dCpos","QQchi2","p05","p01","ds","label"]
d = pd.DataFrame(rows, columns=cols)
assert len(d) == 73 and d.ds.nunique() == 73, "expected 73 datasets"
def recon(r):
    Q=(1+r.V1)/2; b2=(1+r.W2)/2; P11=(r.V1W2+2*Q+2*b2-1)/4; P00=1-Q-b2+P11
    R=(1+r.V2)/2; a2=(1+r.W1)/2; P11b=(r.V2W1+2*R+2*a2-1)/4; P00b=1-R-a2+P11b
    X=P11/Q; Z=P00/(1-Q); Y=P11b/R; W=P00b/(1-R)
    return pd.Series(dict(Q=Q,X=X,Z=Z,R=R,Y=Y,W=W,sAB=Q*X+(1-Q)*Z,sBA=R*Y+(1-R)*W,A2=a2,B2=b2))
d = pd.concat([d, d.apply(recon, axis=1)], axis=1)
r = d[d.ds == 69].iloc[0]
print("dataset 69 P(yes,yes) =", round((r.V1W2 + 2*r.Q + 2*r.B2 - 1)/4, 4), "(DZK text: 0.3987)")
d.to_csv("../data/region_test/tables73.csv", index=False)
