---
name: review-scientific-claim
description: Adversarially review scientific prose (docs, notes, notebook markdown, paper text) for unsupported claims and propose corrected wording. Use when writing or editing any scientific statement in this repository.
---

# Review scientific claim

Review the given prose for overclaims, using `docs/RESEARCH_SPEC.md` (permitted and
prohibited claims) and `.claude/rules/scientific-contract.md` as the standard. If subagents
are available, delegate to the `claim-reviewer` agent; otherwise perform the review
directly.

For each claim in the text:

1. Classify its evidence: theorem / derivation / RS-ansatz / conjecture / empirical
   (finite-dimensional).
2. Check it against the prohibited-overclaims list: distribution-learning from MMD alone,
   sampler-free terminal laws, exact-ancestral-sampling language, asymptotic or capacity
   conclusions from finite-size runs, uniform-data symmetry (V = 0) transferred to
   hidden-manifold data.
3. If flagged, quote the sentence, explain the gap between claim and evidence, and propose
   replacement wording using the approved vocabulary ("approaches the finite-F target under
   this diagnostic", "the sampler-induced terminal law", "consistent with improved
   distributional agreement", "finite-dimensional empirical evidence").

Output: a list of {quoted claim, location, evidence class, verdict OK/OVERCLAIM/AMBIGUOUS,
proposed rewording}. Do not edit files unless the user asked for the fixes to be applied.
