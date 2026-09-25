"""Certified likelihood-ratio statistic for all datasets. Datasets 1-71 (and 73): Frank-Wolfe with the
exact oracle. Dataset 72 (abortion pair): from the published table of Schuman et al. (1981), because
two of its published moments are exchanged. About one minute."""
import numpy as np, pandas as pd
from common import D, cells_from_row, schuman_counts, load
from region_exact import Lambda
from boundary_fit import fit
d=load(); d.loc[d.ds==73,'n']=457; rows=[]
for _,r in d.iterrows():
    if r.ds==72:
        L,Llo,g2,_=fit(schuman_counts(),K=6,restarts=12); rows.append((72,'Schuman table',L,g2))
    else:
        L,g2,_,_=Lambda(r.n*cells_from_row(r),tol=5e-6); rows.append((int(r.ds),'published moments',L,g2))
o=pd.DataFrame(rows,columns=['ds','source','Lambda','twogap']); o.to_csv(D+'region_exact_rerun.csv',index=False)
print("datasets with Lambda > 1e-6:",int((o.Lambda>1e-6).sum()),"| max 2*gap:",o.twogap.max())
