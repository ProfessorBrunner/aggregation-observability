"""Step 4. Parametric bootstrap for a dataset with Lambda > 0 (used for dataset 72).

Usage: python boot72.py <dataset> <B> <seed>. The reported calibration of dataset 72 combines
seeds 2, 3, 4, 5 with B = 200, 520, 540, 540 (1800 replicates); combine with summarize_boot.py.
"""
import sys, json, numpy as np, pandas as pd
from region_lr import S, cells, ll_P, Lambda
ds, B, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng2 = np.random.default_rng(seed)
d = pd.read_csv("../data/region_test/region_lr_results.csv"); r = d[d.ds == ds].iloc[0]; n = int(r.n)
pAB = np.array([r.Q*r.X, r.Q*(1-r.X), (1-r.Q)*(1-r.Z), (1-r.Q)*r.Z])
pBA = np.array([r.R*r.Y, r.R*(1-r.Y), (1-r.R)*(1-r.W), (1-r.R)*r.W])
_, idx, w = ll_P(n*pAB, n*pBA, return_w=True)
qAB, qBA = cells(w @ S[idx]); qAB = np.clip(qAB, 0, 1); qBA = np.clip(qBA, 0, 1); qAB /= qAB.sum(); qBA /= qBA.sum()
L = [Lambda(rng2.multinomial(n, qAB).astype(float), rng2.multinomial(n, qBA).astype(float)) for _ in range(B)]
json.dump(dict(ds=ds, n=n, Lobs=float(r.Lambda), Lboot=L), open(f"../data/region_test/boot_{ds}_{seed}.json", "w"))
print(ds, "B", B, "exceed", int(np.sum(np.array(L) >= r.Lambda)))
