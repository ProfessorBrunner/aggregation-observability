"""Figure 1: normalized coordinates in which the necessary projective bounds become the unit square.
x = (Q - R)/sqrt(1 - s), y = (Q + R - 1)/sqrt(s), with Q, R the first-answer yes probabilities in the
two orders and s the agreement probability. Heterogeneous two-dimensional projective populations
satisfy |x| <= 1 and |y| <= 1 (necessary conditions; the exact hull lies inside). Reciprocal-kernel
mixtures reach every QQ-satisfying table, including tables outside the square."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,'font.family':'serif'})
d=pd.read_csv('../data/region_test_v2/tables73_n.csv')
Q=(1+d.V1)/2; R=(1+d.V2)/2; s=(d.sAB+d.sBA)/2          # V2 = B asked first (Dzhafarov et al. definition)
# dataset 72 from the published table of Schuman et al. (1981)
i72=d.index[d.ds==72][0]
Q[i72]=(138+108)/293; R[i72]=(175+10)/305; s[i72]=((138+44)/293+(175+42)/305)/2
x=(Q-R)/np.sqrt(1-s); y=(Q+R-1)/np.sqrt(s)
keep=d.ds<=72
fig,ax=plt.subplots(figsize=(3.4,3.1))
ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=True,color='C0',alpha=0.12,lw=0))
ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,color='C0',lw=1))
ax.scatter(x[keep],y[keep],s=9,c='k',zorder=3,label='72 survey datasets')
cx=(0.9-0.1)/np.sqrt(1-0.892); cy=0.0
ax.scatter([cx],[cy],marker='x',s=30,c='C3',zorder=3,label='QQ table outside the class')
ax.text(0,1.07,'projective bounds',ha='center',fontsize=7.5,color='C0')
ax.set_xlim(-2.8,2.8); ax.set_ylim(-1.5,2.1); ax.set_aspect('equal')
ax.set_xlabel(r'$(\bar q-\bar r)/\sqrt{1-\bar a}$'); ax.set_ylabel(r'$(\bar q+\bar r-1)/\sqrt{\bar a}$')
ax.legend(frameon=False,loc='upper left',fontsize=6.8,handletextpad=0.3)
fig.tight_layout(); fig.savefig('../figures/fig_hull.pdf')
print("max |x|, |y| over 72 datasets:",round(float(np.abs(x[keep]).max()),3),round(float(np.abs(y[keep]).max()),3),"; counterexample x =",round(cx,2))
