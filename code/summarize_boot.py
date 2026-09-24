"""Combine bootstrap shards for dataset 72 and apply the pre-specified Bonferroni rule (p < 0.05/73)."""
import json, glob, numpy as np
L = []; Lobs = None
for f in sorted(glob.glob("../data/region_test/boot_72_[0-9]*.json")):
    j = json.load(open(f)); L += j["Lboot"]; Lobs = j["Lobs"]
L = np.array(L); k = int(np.sum(L >= Lobs)); p = (1 + k)/(len(L) + 1)
print(f"dataset 72: Lambda = {Lobs:.3f}, B = {len(L)}, exceed = {k}, p = {p:.5f}, Bonferroni threshold = {0.05/73:.6f}")
