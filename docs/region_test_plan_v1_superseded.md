# Analysis plan — region test on the 73 question-order datasets (written before opening S1)

**Source.** Dzhafarov, Zhang & Kujala (2015) Supplementary S1: for each of 73 datasets, the moments ⟨V₁⟩, ⟨W₁⟩, ⟨W₂⟩, ⟨V₂⟩ (marginals: A first, A second, B second, B first) and ⟨V₁W₂⟩, ⟨V₂W₁⟩ (products, one per order). Datasets 1–66 Pew, 67–69 Gallup, 70–72 the PNAS lab/abortion sets, 73 Rose–Jackson (violates QQ; control).

**Reconstruction.** Per order, P(yes) = (1+⟨·⟩)/2 and P(yes,yes) = (⟨VW⟩ + 2P_A + 2P_B − 1)/4. From these: Q = P(A yes | A first), X = P(B yes | A yes), Z = P(B no | A no); R = P(B yes | B first), Y = P(A yes | B yes), W = P(A no | B no). Agreement masses s_AB = QX + (1−Q)Z and s_BA = RY + (1−R)W; QQ says s_AB = s_BA in the population.

**Check before use.** Dataset 67 must reproduce the public Clinton–Gore marginals (Clinton ≈ 0.50 asked first, ≈ 0.57 second; Gore ≈ 0.68 first, ≈ 0.60 second).

**Tests, in order.**
1. Necessary conditions for a heterogeneous mixture of two-dimensional projective respondents (from the corrected completeness derivation): |Q − R| ≤ √(1−s); |Q + R − 1| ≤ √s; |X − Z| ≤ √(s(1−s)/[Q(1−Q)]); |Y − W| ≤ √(s(1−s)/[R(1−R)]), with s the mean of s_AB and s_BA. Report each dataset's margin.
2. Exact criterion: membership of the moment vector (Q, R, s, QX, RY) in the convex hull of the qubit-feasible set S = {(q, r, c, qc, rc)}. Implemented by the support-function LP against a dense random sample of genuine qubit respondents; report the signed distance (negative = outside).
3. Sampling tolerance: the moments are estimates. A dataset counts as outside only if its distance exceeds what sampling noise produces for a genuine mixture at its sample size; calibrated by simulating genuine populations at the dataset's N (Pew ≈ 1500–1650 total; Gallup ≈ 1000; lab 116–118; abortion 651) and recording the distribution of the same distance.

**Kill rule.** If the number of datasets outside the hull beyond sampling tolerance is not distinguishable from the false-outside rate calibrated in step 3, the aggregate region test returns no verdict; the empirical section rests on the subgroup test alone. If a clear subset is outside, the heterogeneous two-dimensional projective class is rejected for those datasets, which is a stronger rejection than the 2016 paper's. If all are inside, the class is consistent with every dataset and confirms nothing; the necessary-condition margins are reported as the plausibility gauge.

**Not done here.** Significance tests of individual GR differences (needs counts, not proportions). Any claim about the individual level.
