"""Shared helpers: reconstruct both 2x2 tables from the Dzhafarov-Zhang-Kujala moments.
Definition (DZK 2016): V1 = A asked first, W2 = B asked second (order A->B);
V2 = B asked first, W1 = A asked second (order B->A).
Cells: A->B [a0b0, a0b1, a1b0, a1b1]; B->A [b0a0, b0a1, b1a0, b1a1]."""
import numpy as np, pandas as pd
D = '../../data/region_test_v2/'
def cells_from_row(r, swapped=False):
    Q=(1+r.V1)/2; B2=(1+r.W2)/2; P11=(r.V1W2+2*Q+2*B2-1)/4
    if swapped: R=(1+r.W1)/2; A2=(1+r.V2)/2
    else:       R=(1+r.V2)/2; A2=(1+r.W1)/2
    P11b=(r.V2W1+2*R+2*A2-1)/4
    return np.array([P11,Q-P11,B2-P11,1-Q-B2+P11,P11b,R-P11b,A2-P11b,1-R-A2+P11b])
def schuman_counts():
    """Schuman, Presser & Ludwig (1981), Table 2 (June 1979), A = specific item, B = general item."""
    AB=np.round(np.array([47.1,36.9,1.0,15.0])/100*293); BA=np.round(np.array([57.4,3.3,25.6,13.8])/100*305)
    return np.r_[AB,BA]
def load(): return pd.read_csv(D+'tables73_n.csv')
