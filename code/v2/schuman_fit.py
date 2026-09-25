"""Dataset 72 from its published table, and from the published (exchanged) moments (a numerical estimate)."""
import numpy as np
from common import load, cells_from_row, schuman_counts
from boundary_fit import fit
L,Llo,g2,_=fit(schuman_counts(),K=6,restarts=12); print(f"published table: Lambda={L:.4f} (gap-based lower {Llo:.4f}, not a global bound; 2gap {g2:.1e})")
r=load().query('ds==72').iloc[0]; L,Llo,g2,_=fit(r.n*cells_from_row(r),K=6,restarts=12)
print(f"published moments (exchanged): Lambda={L:.4f} (gap-based lower {Llo:.4f}, not a global bound)")
