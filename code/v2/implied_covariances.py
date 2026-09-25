"""Observed reciprocity differences G2, G5 and implied covariances C_q, C_r for the 72 datasets."""
import numpy as np, pandas as pd
from scipy.stats import norm
from common import D, cells_from_row, schuman_counts, load
d=load(); d=d[d.ds<=72]; rows=[]
for _,r in d.iterrows():
    c=schuman_counts() if r.ds==72 else r.n*cells_from_row(r)
    nAB,nBA=c[:4],c[4:]; Q=(nAB[0]+nAB[1])/nAB.sum(); R=(nBA[0]+nBA[1])/nBA.sum()
    G2=nAB[0]/(nAB[0]+nAB[1])-nAB[3]/(nAB[2]+nAB[3]); G5=nBA[0]/(nBA[0]+nBA[1])-nBA[3]/(nBA[2]+nBA[3])
    s=((nAB[0]+nAB[3])/nAB.sum()+(nBA[0]+nBA[3])/nBA.sum())/2; N=nAB.sum()+nBA.sum()
    Cq=G2*Q*(1-Q); Cr=G5*R*(1-R)
    rows.append((int(r.ds),G2,G5,Cq,Cr,abs(Cq)*np.sqrt(N)/np.sqrt(2*s*(1-s)*Q*(1-Q)),abs(Cr)*np.sqrt(N)/np.sqrt(2*s*(1-s)*R*(1-R))))
o=pd.DataFrame(rows,columns=['ds','G2','G5','Cq','Cr','theta2','theta5'])
za=norm.ppf(1-0.05/12)
print("median |C_q| %.3f |C_r| %.3f, max %.3f; individually detectable: %d of %d"%(o.Cq.abs().median(),o.Cr.abs().median(),max(o.Cq.abs().max(),o.Cr.abs().max()),int(((o.theta2>za)|(o.theta5>za)).sum()),len(o)))
