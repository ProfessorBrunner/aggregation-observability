"""Record of the bootstrap for dataset 72 read from its published (exchanged) moments: parametric
bootstrap from the fitted projective point (mstar_72.npy), refits (numerical estimates). Usage: python boot_published_moments.py <start_seed> <B>.
The stored replicates (seeds 1000000-1002003) are in data/region_test_v2/bootstrap_published_moments/."""
import numpy as np, sys, json
from common import D
from region_exact import cells
from boundary_fit import fit
m0=np.load(D+'mstar_72.npy'); p0=cells(m0); pA=np.clip(p0[:4],0,1); pA/=pA.sum(); pB=np.clip(p0[4:],0,1); pB/=pB.sum()
start,B=int(sys.argv[1]),int(sys.argv[2]); res=[]
for seed in range(start,start+B):
    rng=np.random.default_rng(seed); c=np.r_[rng.multinomial(298,pA),rng.multinomial(298,pB)].astype(float)
    L,Llo,g2,_=fit(c,K=4,restarts=3,seed=seed)
    if g2>1e-3: L,Llo,g2,_=fit(c,K=6,restarts=10,seed=seed+7)
    res.append((L,Llo,g2))
json.dump(dict(n=298,seeds=[start,start+B],res=res),open(D+f'bootstrap_published_moments/bootx_298_{start}.json','w'))
print("B",B,"exceed(>=9.3873):",sum(r[0]>=9.3873 for r in res))
