"""
sec4_order_restriction.py -- Gate G1 derivation support.

Task (a1) of the ruling on the section-4 chassis: derive the action-level
FOMO/loss order restriction R_FL for our measurement protocol (manuscript
Appendix B item 1; Conjecture "Two-fears order restriction", sec:order) and
establish whether it is parameter-free.

THE PROTOCOL, AS FROZEN HERE
----------------------------
Matched respondents process equivalent market information through one of two
narrative sequences and then answer the SAME act/hold/exit prompt:

    arm 1 (FOMO first) :  measure F  ->  measure L  ->  measure action A
    arm 2 (loss first) :  measure L  ->  measure F  ->  measure A

F and L are the dichotomous frame activations of manuscript Eq. (3); A is the
action readout of Eq. (2) with operator Pi_A.  This is a THREE-measurement
protocol, which is what distinguishes it from the canonical two-question QQ
design of Wang & Busemeyer: there the two answers ARE the observables, here a
third measurement follows them.

WHAT THIS MODULE DOES
---------------------
Computes every sequential probability of that protocol by explicit state
update, for a general specification, so the derived identities can be checked
on random instances rather than asserted:

  * dimension d (2 = the minimal specification, >2 = degenerate frames)
  * sharp rank-1 projective frames, or unsharp (Luders POVM) frames
  * arbitrary density matrix rho0, independently chosen per arm
  * arbitrary action POVM element Pi_A, optionally order-dependent

The identities derived in the memo are the assertions in `check_instance`.
"""

import numpy as np
from scipy.linalg import sqrtm

__all__ = ["random_state", "random_effect", "frame_basis", "kraus_set",
           "protocol", "qq_value", "R_FL", "check_instance"]


# ----------------------------------------------------------------------
# random ingredients
# ----------------------------------------------------------------------
def random_state(d, rng, pure=False):
    """Haar-random density matrix (pure, or mixed via a Ginibre draw)."""
    if pure:
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        v /= np.linalg.norm(v)
        return np.outer(v, v.conj())
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    R = G @ G.conj().T
    return R / np.trace(R).real


def random_effect(d, rng):
    """Random POVM element 0 <= Pi <= I, generic (neither projector nor scalar)."""
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = G @ G.conj().T
    w = np.linalg.eigvalsh(H).max()
    return H / (w * (1.0 + 0.3 * rng.random()))     # spectrum strictly inside [0,1]


def frame_basis(d, rng):
    """Random orthonormal basis, returned as columns."""
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(G)
    return Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]


# ----------------------------------------------------------------------
# measurement channels
# ----------------------------------------------------------------------
def kraus_set(basis, split, eta=1.0):
    """Kraus operators for a dichotomous frame measurement.

    basis : (d,d) orthonormal columns
    split : number of basis vectors assigned to the "yes" outcome
            (split = 1 with d = 2 is the minimal specification: sharp rank-1)
    eta   : 1.0 = sharp projective; eta < 1 = unsharp Luders POVM with
            E_yes = eta*P_yes + (1-eta)*P_no, Kraus = sqrt(E)

    Returns [K_yes, K_no] with sum K†K = I.
    """
    d = basis.shape[0]
    P = basis[:, :split] @ basis[:, :split].conj().T
    Q = np.eye(d) - P
    if eta >= 1.0:
        return [P, Q]
    Ey = eta * P + (1.0 - eta) * Q
    En = (1.0 - eta) * P + eta * Q
    return [np.asarray(sqrtm(Ey)), np.asarray(sqrtm(En))]


PBRANCH_FLOOR = 1e-9


def _apply(K, rho):
    """Luders update.  Returns (branch probability, normalised post-state).

    A branch whose probability is below PBRANCH_FLOOR has no well-defined
    conditional state: normalising it divides by numerical noise and yields a
    non-physical matrix.  Such branches return NaN so that any restriction
    computed from them is NaN rather than a plausible-looking number.  This
    matters because it is exactly what happens as the two frames approach
    commutation -- the second frame's answer becomes deterministic given the
    first, so half the sequential cells empty out and the action-level
    restriction stops being estimable rather than becoming false.
    """
    s = K @ rho @ K.conj().T
    p = np.trace(s).real
    if p < PBRANCH_FLOOR:
        return p, np.full_like(s, np.nan)
    return p, s / p


def protocol(rho0, first, second, Pi_A):
    """Run one arm: measure `first`, then `second`, then the action.

    first, second : Kraus lists from kraus_set (the two frame measurements)
    Pi_A          : action POVM element

    Returns dict with
      joint[(i,j)]  probability of the two frame answers, in order
      act[(i,j)]    P(A | frame answers), i.e. the conditional action prob
      marg2[j]      marginal probability of the SECOND frame's answer
      P_A           unconditional action probability for this arm
    """
    joint, act = {}, {}
    for i, K1 in enumerate(first):
        p1, r1 = _apply(K1, rho0)
        for j, K2 in enumerate(second):
            p2, r2 = _apply(K2, r1)
            joint[(i, j)] = p1 * p2
            act[(i, j)] = np.trace(r2 @ Pi_A).real
    marg2 = {j: sum(joint[(i, j)] for i in range(len(first)))
             for j in range(len(second))}
    P_A = sum(joint[k] * act[k] for k in joint if joint[k] > PBRANCH_FLOOR)
    return dict(joint=joint, act=act, marg2=marg2, P_A=P_A,
                min_cell=min(joint.values()))


# ----------------------------------------------------------------------
# the two candidate restrictions
# ----------------------------------------------------------------------
def qq_value(arm_FL, arm_LF):
    """Canonical QQ combination, manuscript Eqs. (4)-(5), on the FRAME answers:

        q_FL = p(F_y,L_y) + p(F_n,L_n) - p(L_y,F_y) - p(L_n,F_n).

    Zero under the canonical projective two-question model (Wang-Busemeyer).
    Involves the action not at all.
    """
    return ((arm_FL["joint"][(0, 0)] + arm_FL["joint"][(1, 1)])
            - (arm_LF["joint"][(0, 0)] + arm_LF["joint"][(1, 1)]))


def R_FL(arm_FL, arm_LF):
    """The ACTION-level restriction derived in the memo:

        R_FL = P(A | F->L, L=y) + P(A | F->L, L=n)
             - P(A | L->F, F=y) - P(A | L->F, F=n)

    i.e. sum the conditional action probability over the two answers of
    whichever frame was activated LAST, and difference across arms.  Under the
    minimal specification (d = 2, sharp rank-1 frames, order-independent Pi_A)
    each sum equals Tr Pi_A, so R_FL = 0 identically -- with no free parameter,
    and without requiring the two arms to share rho0 or the frames to share
    any overlap.

    Conditional action probabilities are read off the LAST frame's answer;
    under the minimal specification act[(i,j)] does not depend on i, which is
    itself an assertion checked in `check_instance`.
    """
    s_FL = sum(arm_FL["act"][(0, j)] for j in (0, 1))
    s_LF = sum(arm_LF["act"][(0, j)] for j in (0, 1))
    return s_FL - s_LF


def check_instance(d=2, split=1, eta=1.0, same_rho=True, order_dep_A=False,
                   seed=0):
    """Build one random instance and return every quantity the memo asserts."""
    rng = np.random.default_rng(seed)
    BF = frame_basis(d, rng)
    BL = frame_basis(d, rng)
    F = kraus_set(BF, split, eta)
    L = kraus_set(BL, split, eta)
    Pi = random_effect(d, rng)
    Pi2 = Pi if not order_dep_A else random_effect(d, rng)
    r1 = random_state(d, rng)
    r2 = r1 if same_rho else random_state(d, rng)

    a_FL = protocol(r1, F, L, Pi)         # FOMO first, action last
    a_LF = protocol(r2, L, F, Pi2)        # loss first

    # does the first frame still matter once the second is registered?
    first_frame_effect = max(
        max(abs(a["act"][(0, j)] - a["act"][(1, j)]) for j in range(d and 2))
        for a in (a_FL, a_LF))

    return dict(
        d=d, split=split, eta=eta, same_rho=same_rho, order_dep_A=order_dep_A,
        trace_Pi=float(np.trace(Pi).real),
        qq=float(qq_value(a_FL, a_LF)),
        R_FL=float(R_FL(a_FL, a_LF)),
        order_contrast=float(a_FL["P_A"] - a_LF["P_A"]),
        first_frame_effect=float(first_frame_effect),
        sum_act_FL=float(sum(a_FL["act"][(0, j)] for j in (0, 1))),
        sum_act_LF=float(sum(a_LF["act"][(0, j)] for j in (0, 1))),
    )
