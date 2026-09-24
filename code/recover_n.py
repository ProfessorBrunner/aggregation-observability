"""Step 2. Recover the per-order sample size of each dataset from the published QQ chi-square.

With equal halves n per order, chi2 = 2 n [KL(d1||dbar) + KL(d2||dbar)], d1, d2 the two
disagreement masses and dbar their mean. Where |d1 - d2| < 0.002 (four-decimal rounding makes
the inversion unstable) the median of the source group is used. Check: recovered Pew sizes
span 126-911 against the 125-927 stated by Dzhafarov et al.
Output: ../data/region_test/tables73_n.csv
"""
import numpy as np, pandas as pd
d = pd.read_csv("../data/region_test/tables73.csv")
kl = lambda p, q: p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
d1, d2 = 1 - d.sAB, 1 - d.sBA; db = (d1 + d2)/2
den = 2*(kl(d1, db) + kl(d2, db))
d["n_rec"] = np.where(den > 0, d.QQchi2/den, np.nan)
d["stable"] = (d1 - d2).abs() >= 0.002
d["group"] = np.where(d.ds <= 66, "Pew", np.where(d.ds <= 69, "Gallup", np.where(d.ds == 73, "Gallup", "other")))
med = d[d.stable].groupby("group").n_rec.median()
d["n"] = np.where(d.stable, d.n_rec, d.group.map(med)).round()
print("stable:", int(d.stable.sum()), "of 73; Pew range:",
      int(d[d.stable & (d.group == "Pew")].n_rec.min()), "-", int(d[d.stable & (d.group == "Pew")].n_rec.max()))
d.to_csv("../data/region_test/tables73_n.csv", index=False)
