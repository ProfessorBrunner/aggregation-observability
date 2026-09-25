import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,'figure.dpi':150,'font.family':'serif'})
W=3.4  # single PRE column width, inches

# ---------- Fig 1: G2 vs dispersion (bridge) ----------
g=pd.read_csv('../data/cs_simulation/g2_summary.csv')
fig,ax=plt.subplots(1,2,figsize=(2*W,2.6))
a=g[g.x_read==2]
ax[0].errorbar(a.sd_over_Delta,a.G2_meas_mean,yerr=a.G2_sd,fmt='o',ms=3,color='k',capsize=1.5,label='measured (64 draws, ±1 sd)')
ax[0].plot(a.sd_over_Delta,a.G2_pred_mean,'-',color='C0',lw=1,label=r'$C_q/[\bar q(1-\bar q)]$')
ax[0].axhline(0,color='0.6',lw=0.6)
ax[0].set_xlabel(r'$\mathrm{sd}(\eta)/\Delta$'); ax[0].set_ylabel(r'$G_2$'); ax[0].set_title(r'(a) readout $x_{\rm read}=2\Delta$',fontsize=9)
ax[0].legend(frameon=False)
for sd,c in [(0.3,'C1'),(1.1,'C0'),(3.0,'C2')]:
    b=g[np.isclose(g.sd_over_Delta,sd)].sort_values('x_read')
    ax[1].loglog(b.x_read,np.abs(b.G2_meas_mean)+1e-18,'o-',ms=3,color=c,label=fr'$|G_2|$, sd$(\eta)={sd}\Delta$')
b=g[np.isclose(g.sd_over_Delta,1.1)].sort_values('x_read')
ax[1].loglog(b.x_read,b.var_c,'s--',ms=3,color='0.4',label=r'var$(c)$, sd$(\eta)=1.1\Delta$')
ax[1].set_xlabel(r'$x_{\rm read}/\Delta$'); ax[1].set_title('(b) dependence on readout field',fontsize=9)
ax[1].legend(frameon=False,loc='lower left')
fig.tight_layout(); fig.savefig('../figures/fig_g2.pdf'); plt.close(fig)

# ---------- Fig 2: undershoot vs dispersion ----------
d=pd.read_csv('../data/cs_simulation/definedness_summary.csv'); p=pd.read_csv('../data/cs_simulation/dispersion_summary.csv')
fig,ax=plt.subplots(figsize=(W,2.6))
ax.errorbar(d.sd_over_Delta,d.U_mean,yerr=d.U_sd,fmt='o',ms=3,color='k',capsize=1.5,label='measured, 256 draws')
ax.plot(d.sd_over_Delta,d.U_pred_convolution,'-',color='C0',lw=1.2,label='convolution law')
ax.plot(p.sd_over_Delta,p.U_pred_curvature,'--',color='C1',lw=1,label='curvature form')
ss=np.linspace(0,3,301); Pp=np.sqrt(4.48**2+4)/5.0; ax.plot(ss,0.277668*np.exp(-0.5*(Pp*ss)**2),':',color='C3',lw=1.2,label='phase-scrambling form')
ax.axhline(0.069219,color='0.5',lw=0.7); ax.text(2.55,0.078,r'$U^*$',fontsize=8,color='0.3')
ax.set_ylim(-0.05,0.3); ax.set_xlabel(r'$s/\Delta=\mathrm{sd}(\eta)/\Delta$'); ax.set_ylabel(r'undershoot $U$')
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=2); fig.set_size_inches(W,3.1); fig.tight_layout(); fig.savefig('../figures/fig_undershoot.pdf'); plt.close(fig)

# ---------- Fig 3: definedness ----------
fig,ax=plt.subplots(figsize=(W,2.5))
for lat,c,lab in [('24000','k','24000 output times'),('1200','C3','1200 output times')]:
    f=d[f'frac_defined_{lat}']; lo=d[f'wilson_lo_{lat}']; hi=d[f'wilson_hi_{lat}']
    ax.errorbar(d.sd_over_Delta,f,yerr=[f-lo,hi-f],fmt='o-',ms=3,lw=0.8,color=c,capsize=1.5,label=lab)
ax.axvline(1.3858,color='C0',lw=0.8,ls='--'); ax.text(1.62,0.72,'convolution\nprediction',fontsize=7,color='C0')
ax.set_xlim(0.8,2.1); ax.set_xlabel(r'$\mathrm{sd}(\eta)/\Delta$'); ax.set_ylabel('fraction of draws\nreaching $m=-0.5$')
ax.legend(frameon=False,loc='lower left'); fig.tight_layout(); fig.savefig('../figures/fig_definedness.pdf'); plt.close(fig)

# ---------- Fig 4: detectability boundaries ----------
za=norm.ppf(1-0.05/12)
def power(Cq,N,cb,qb):
    G=Cq/(qb*(1-qb)); X=cb+(1-qb)*G; Z=cb-qb*G
    se=np.sqrt((2/N)*(X*(1-X)/qb+Z*(1-Z)/(1-qb))); return 1-norm.cdf(za-abs(G)/se)
def Cq_at(N,pi,cb,qb):
    hi=0.999*min(qb*(1-cb),cb*(1-qb))
    f=lambda c: power(c,N,cb,qb)-pi
    return brentq(f,1e-6,hi) if f(hi)>0 else np.nan
Ns=np.logspace(np.log10(150),np.log10(8000),60)
fig,ax=plt.subplots(figsize=(W,2.6))
for pi,ls in [(0.5,'-'),(0.8,'--')]:
    ax.loglog(Ns,[Cq_at(N,pi,0.7,0.5) for N in Ns],ls,color='k',label=fr'{int(pi*100)}% power, $\bar c=0.7,\ \bar q=0.5$')
lo=[np.nanmin([Cq_at(N,0.5,cb,qb) for cb in [0.5,0.7,0.9] for qb in [0.3,0.5,0.7]]) for N in Ns]
hi=[np.nanmax([Cq_at(N,0.5,cb,qb) for cb in [0.5,0.7,0.9] for qb in [0.3,0.5,0.7]]) for N in Ns]
ax.fill_between(Ns,lo,hi,color='C0',alpha=0.2,label=r'50% power, range over $\bar c\in[0.5,0.9],\ \bar q\in[0.3,0.7]$')
ax.axvline(766,color='0.5',lw=0.7); ax.axhline(0.0225,color='C3',lw=0.7,ls=':'); ax.set_xlabel(r'sample size $N$'); ax.set_ylabel(r'detectable $|C_q|$'); ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=1,fontsize=6.5); fig.set_size_inches(W,3.3)
fig.tight_layout(); fig.savefig('../figures/fig_detect.pdf'); plt.close(fig)

# ---------- Fig 5: Lambda across datasets (exact solver; abortion pair from published table) ----------
r=pd.read_csv('../data/region_test_v2/region_exact_results.csv')
L=np.maximum(r.Lambda.values,0).copy(); L[r.ds.values==72]=0.0   # published table (Schuman et al. 1981) gives Lambda = 0
fig,ax=plt.subplots(figsize=(W,2.5))
col=np.where(r.ds==73,'0.6',np.where(r.p05==1,'C3','k'))
ax.scatter(r.ds,np.where(L>1e-6,L,1e-4),s=10,c=col,zorder=3)
ax.scatter([72],[9.387],s=22,facecolors='none',edgecolors='C3',zorder=3)
ax.set_yscale('log'); ax.set_ylim(5e-5,50)
ax.annotate('abortion pair, published moments\n(two moments exchanged)',xy=(72,9.387),xytext=(22,1.2),fontsize=7,arrowprops=dict(arrowstyle='-',lw=0.6))
ax.text(1,1.6e-4,r'$\Lambda=0$ plotted at $10^{-4}$',fontsize=6.5,color='0.4')
ax.set_xlabel('dataset'); ax.set_ylabel(r'$\Lambda=2(\ell_{\rm QQ}-\ell_{\rm P})$')
fig.tight_layout(); fig.savefig('../figures/fig_lambda.pdf'); plt.close(fig)
print("figures written")
