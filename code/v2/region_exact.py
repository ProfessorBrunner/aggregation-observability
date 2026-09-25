"""Exact likelihood-ratio test for the heterogeneous two-dimensional projective class.

Moment vector m = (Q, R, s, u, w) with u = QX, w = RY. Pooled cells are linear in m.
l_QQ: closed form (QQ-constrained multinomial MLE).
l_P : max of the concave log-likelihood over conv S, S = {(q, r, c, qc, rc)} for qubit respondents,
      by Frank-Wolfe with away steps and an exact linear oracle:
      for gradient g, max over S of g.m = max over c in [0,1] of
        g3 c + (a+b)/2 + (1/2) sqrt(a^2 + b^2 + 2ab(2c-1)),   a = g1 + g4 c,  b = g2 + g5 c,
      attained at h = (a u + b v)/|a u + b v| (a pure state), so conv S needs only pure states.
Stopping: Frank-Wolfe duality gap g.(s* - x) < tol; the gap bounds l_P(opt) - l_P(x), so
Lambda is reported with an upper and lower bound.
"""
import numpy as np
CG = np.linspace(0.0, 1.0, 4001)

def cells(m):
    Q, R, s, u, w = m
    return np.array([u, Q-u, 1-Q-s+u, s-u, w, R-w, 1-R-s+w, s-w])

GM = np.array([[0,0,0,1,0],[1,0,0,-1,0],[-1,0,-1,1,0],[0,0,1,-1,0],
               [0,0,0,0,1],[0,1,0,0,-1],[0,-1,-1,0,1],[0,0,1,0,-1]], float)

def loglik(n, p):
    p = np.asarray(p, float); m = n > 0
    if np.any(p[m] <= 0): return -np.inf
    return float(np.sum(n[m]*np.log(p[m])))

def oracle(g):
    """Exact argmax over S of g.(q, r, c, qc, rc)."""
    def val(c):
        a = g[0] + g[3]*c; b = g[1] + g[4]*c
        return g[2]*c + (a+b)/2 + 0.5*np.sqrt(np.maximum(a*a + b*b + 2*a*b*(2*c-1), 0))
    v = val(CG); i = int(np.argmax(v))
    lo, hi = CG[max(i-1, 0)], CG[min(i+1, len(CG)-1)]
    for _ in range(60):                                   # golden-section refinement
        m1 = lo + 0.382*(hi-lo); m2 = lo + 0.618*(hi-lo)
        if val(m1) < val(m2): lo = m1
        else: hi = m2
    c = (lo+hi)/2
    if val(c) < v[i]: c = CG[i]
    a = g[0] + g[3]*c; b = g[1] + g[4]*c; ct = 2*c - 1
    W = np.sqrt(max(a*a + b*b + 2*a*b*ct, 0.0))
    if W < 1e-15: q = r = 0.5
    else:
        q = 0.5*(1 + (a + b*ct)/W); r = 0.5*(1 + (a*ct + b)/W)
    return np.array([q, r, c, q*c, r*c])

def ll_QQ(n):
    nAB, nBA = n[:4], n[4:]; t1, t2 = nAB.sum(), nBA.sum()
    s = (nAB[0]+nAB[3]+nBA[0]+nBA[3])/(t1+t2)
    def fit(k, t):
        a = (k[0]+k[3])/t; p = k/t
        return p*np.where([1,0,0,1], s/max(a,1e-300), (1-s)/max(1-a,1e-300))
    return loglik(nAB, fit(nAB, t1)) + loglik(nBA, fit(nBA, t2))

def ll_P(n, tol=1e-9, maxit=200000):
    """Frank-Wolfe with away steps. Returns (loglik, m, gap, iterations)."""
    nAB, nBA = n[:4], n[4:]
    Q = (nAB[0]+nAB[1])/nAB.sum(); R = (nBA[0]+nBA[1])/nBA.sum()
    x = np.array([Q, R, 0.5, Q*0.5, R*0.5])            # interior start: c = 1/2 with the observed marginals
    # represent x as a mixture of two atoms with c = 1/2 would need feasibility; start instead from an oracle atom
    g0 = np.zeros(5); g0[0] = 1e-6
    atoms = [oracle(GM.T @ (n/np.maximum(cells(x), 1e-300)))]
    w = [1.0]; x = atoms[0].copy()
    # guard: if the first atom gives a zero cell with positive count, mix toward a safe interior atom
    for it in range(1, maxit+1):
        p = cells(x)
        if np.any(p[n > 0] <= 0):
            safe = np.array([Q, R, 0.5, Q*0.5, R*0.5])
            # interior point of conv S: c = 1/2, q = Q, r = R is a qubit point iff Eq. (13) holds; use q = r = 1/2 fallback
            safe = np.array([0.5, 0.5, 0.5, 0.25, 0.25])
            atoms.append(safe); w = [wi*0.5 for wi in w] + [0.5]; x = sum(wi*a for wi, a in zip(w, atoms)); continue
        g = GM.T @ (n/p)
        s_ = oracle(g); dFW = s_ - x; gap = float(g @ dFW)
        if gap < tol: return loglik(n, cells(x)), x, gap, it
        vals = [g @ a for a in atoms]; j = int(np.argmin(vals)); dA = x - atoms[j]
        if gap >= float(g @ dA) or len(atoms) == 1:
            d = dFW; gmax = 1.0; away = False
        else:
            d = dA; gmax = w[j]/(1 - w[j]) if w[j] < 1 else 1e9; away = True
        lo, hi = 0.0, gmax                                  # bisection line search on the directional derivative
        for _ in range(60):
            mid = (lo+hi)/2; pm = cells(x + mid*d)
            if np.any(pm[n > 0] <= 0): hi = mid; continue
            if (GM.T @ (n/pm)) @ d > 0: lo = mid
            else: hi = mid
        gam = lo
        if not away:
            w = [wi*(1-gam) for wi in w]
            for k, a in enumerate(atoms):
                if np.allclose(a, s_, atol=1e-13): w[k] += gam; break
            else: atoms.append(s_); w.append(gam)
        else:
            w = [wi*(1+gam) for wi in w]; w[j] -= gam
            if w[j] < 1e-14: atoms.pop(j); w.pop(j)
        x = sum(wi*a for wi, a in zip(w, atoms))
    p = cells(x); g = GM.T @ (n/p); gap = float(g @ (oracle(g) - x))
    return loglik(n, p), x, gap, maxit

def Lambda(n, tol=1e-9):
    lq = ll_QQ(n); lp, m, gap, it = ll_P(n, tol)
    return max(0.0, 2*(lq - lp)), 2*gap, m, it
