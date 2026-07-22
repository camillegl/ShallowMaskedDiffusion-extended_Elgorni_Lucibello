#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices

#show figure.caption: set align(left)
#show figure.caption: set text(style: "italic")

#show: arkheion.with(
  title: [Statistical Physics of a Linear Masked Diffusion Model on Hidden-Manifold Data],
  authors: (
    (name: "Filippo Elgorni", email: "filippo.elgorni@phd.unibocconi.it", affiliation: [Bocconi University, Milan]),
    (name: "Carlo Lucibello", email: "carlo.lucibello@unibocconi.it", affiliation: [Bocconi University, Milan]),
  ),
  date: datetime.today().display("[day] [month repr:Long] [year]"),
  abstract: [#align(left)[
    We develop the statistical-physics theory of the shallow (linear-score) masked
    diffusion model trained on data from a hidden-manifold (random-feature,
    sign-channel) teacher $x = "sign"(F z)$, $F_(i nu) ~ cal(N)(0, 1\/D)$, in the
    proportional limit $D, N, M -> oo$ at fixed $gamma = N\/D$ and $alpha = M\/D$.
    The note is self-contained: we derive the finite-$F$ teacher law $P_F$ exactly
    (zero means, Wishart preactivation covariance, arcsine correlations, and an exact
    proof that the disorder-averaged law $EE_F P_F$ is uniform on the hypercube);
    we state the masked-BCE Gibbs measure; we prove that the mask-channel weights
    $V$ and bias vanish at the population level for every flip-symmetric data law,
    in particular for $P_F$ at fixed $F$; we set up the quenched replica theory with
    teacher-induced order parameters, carry the replica-symmetric calculation down
    to a closed scalar saddle-point system in terms of Marchenko--Pastur transforms;
    and we map order parameters to train/test loss, retrieval (U-turn) accuracy,
    pair correlations, memorization, and the overlap-law content of the MMD
    diagnostic. Every statement is labeled theorem, derivation, RS-ansatz,
    heuristic, or conjecture; unfinished pieces are stated as such. Section 8 maps
    each theoretical prediction to an executable Phase-4C experiment and metric.
  ]],
)

#let note(content) = highlight([NOTE: ] + content, fill: gray.lighten(80%))

#let citecolor = rgb("#93430e")
#show cite: set text(fill: citecolor)
#show link: set text(fill: blue)
#show link: underline
#show ref: set text(fill: blue)

#let extr = math.op("extr", limits: true)
#let Tr = math.op("Tr")
#let Var = math.op("Var")
#let Cov = math.op("Cov")

#let tag(s) = box(stroke: 0.5pt, inset: 2.5pt, radius: 2pt)[#text(size: 7.5pt, s)]
#let DEF = tag([def])
#let THM = tag([theorem])
#let DER = tag([derivation])
#let RS = tag([RS-ansatz])
#let GET = tag([Gaussian-equivalence])
#let MF = tag([mean-field closure])
#let HEUR = tag([heuristic])
#let CONJ = tag([conjecture])

#outline(depth: 2, indent: n => n * 1em)

= Setting, notation, and evidence labels <sec:setting>

== Model [DEF]

The generative teacher is the hidden-manifold (random-feature, sign-channel) law
@goldt2020hidden:
$ F in RR^(N times D), quad F_(i nu) ~ cal(N)(0, 1/D) "i.i.d.", quad
  z ~ cal(N)(0, I_D), quad
  x = "sign"(F z) in {-1, +1}^N, $
with the convention $"sign"(0) := +1$ (a measure-zero event, immaterial throughout).
The two control ratios are
$ gamma = N/D quad "(aspect_ratio)", quad alpha = M/D quad "(sample_ratio)", quad
  M/N = alpha/gamma quad "(visible_load, derived)", $
where $M$ is the training-set size. The learner is the upstream linear masked
diffusion model (`LinearBackbone`; `docs/ORIGINAL_ARCHITECTURE.md`), trained with
the continuous-time masked-BCE objective (Section 3).

#note[Notation discipline. In this note $alpha$ and $gamma$ are *only* the two
ratios above (code names `sample_ratio`, `aspect_ratio`). They never denote the
Rényi order or a disorder-replica index, two of the other meanings catalogued in
`docs/NOTATION.md`; replica indices here are $a, b in {1, dots, n}$. Legacy code
uses $L = N$ and a legacy `alpha` $= M\/N = alpha\/gamma$ (`visible_load`).]

== The three data laws [DEF]

Following `docs/RESEARCH_SPEC.md`, three objects are never conflated:
+ the *finite-$F$ law* $P_F$ (quenched teacher, fresh latents);
+ the *empirical training law* $hat(P)_M = (1\/M) sum_(mu=1)^M delta_(x^mu)$,
  $x^mu = "sign"(F z^mu)$, $z^mu$ i.i.d.;
+ the *disorder-averaged law* $EE_F P_F$ (annealed mixture over teachers).

== Evidence labels

Every numbered statement below carries one label, in line with
`.claude/rules/scientific-contract.md`:
#THM exact theorem (no uncontrolled step);
#DER derivation (exact given explicitly stated inputs);
#RS replica-symmetric-ansatz result;
#GET Gaussian-equivalence step (theorem of the literature under conditions not
verified here);
#MF mean-field closure (correlations neglected);
#HEUR heuristic;
#CONJ conjecture.
No asymptotic statement in this note has been verified against simulations; all
comparisons to experiment are delegated to the Phase-4C dictionary of Section 8.

= The finite-$F$ teacher law $P_F$ <sec:teacher>

This section derives the exact low-order structure of $P_F$ and separates what is
true pointwise in $F$ from what is true only after the disorder average.

== One-point statistics: zero means and the global flip symmetry

*Statement T1* #THM. _For every $F$ (off the measure-zero set where a row of $F$
vanishes) and every site $i$,_
$ EE_(x ~ P_F) [x_i] = 0, quad "and" quad P_F (-x) = P_F (x) quad "for all" x in {-1,+1}^N. $ <eq:flip>

_Proof._ $P_F (x) = integral D z thin product_(i=1)^N Theta(x_i (F z)_i)$ with
$D z = (2 pi)^(-D\/2) e^(-norm(z)^2\/2) dif z$ the standard Gaussian measure. The
map $z arrow.r -z$ preserves $D z$ and sends $(F z)_i arrow.r -(F z)_i$, hence
$"sign"((F z)_i) arrow.r -"sign"((F z)_i)$ (the tie set ${(F z)_i = 0}$ has
measure zero). The first claim follows from
$EE ["sign"((F z)_i)] = -EE ["sign"((F (-z))_i)]$; the second by applying the same
substitution inside $P_F (-x)$. $square$

_Remark._ @eq:flip is the *only* exact symmetry of $P_F$ at fixed $F$: single-site
marginals are uniform, but coordinates are not exchangeable, because the
correlation $EE[x_i x_j mid(|) F]$ depends on $(i, j)$ through the teacher
(Statement T3). Section 4 shows that @eq:flip — not coordinate exchangeability —
is the operative symmetry for the $V = 0$ question.

== Preactivation covariance: the Wishart Gram matrix

Write $h := F z in RR^N$ for the preactivations. Conditional on $F$,
$ h mid(|) F ~ cal(N)(0, G), quad G := F F^top in RR^(N times N), quad
  G_(i j) = sum_(nu=1)^D F_(i nu) F_(j nu). $ <eq:gram>

*Statement T2* #THM (random-matrix facts about $G$).
+ $EE_F [G_(i j)] = delta_(i j)$: since $EE_F [F_(i nu) F_(j nu')] =
  (1\/D) delta_(i j) delta_(nu nu')$, summing over $nu$ gives the identity.
+ $G$ is a white Wishart matrix, $G = (1\/D) X X^top$ with
  $X_(i nu) = sqrt(D) thin F_(i nu) ~ cal(N)(0, 1)$. Diagonal entries:
  $G_(i i) = norm(F_i)^2$, with $EE_F G_(i i) = 1$,
  $Var_F G_(i i) = 2\/D$. Off-diagonal entries ($i != j$): $EE_F G_(i j) = 0$,
  $Var_F G_(i j) = 1\/D$. Hence $G_(i j) = O_p (D^(-1\/2))$ entrywise.
+ In the proportional limit $D -> oo$, $gamma = N\/D$ fixed, the empirical
  spectral distribution of $G$ converges to the Marchenko--Pastur law $rho_gamma$
  @marchenko1967distribution with mean $1$, support
  $[(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2]$ (intersected with $RR_(>= 0)$),
  and an atom at $0$ of mass $1 - 1\/gamma$ when $gamma > 1$ (there are $N - D$
  exact zero modes). The first moments are
  $ lim 1/N EE_F Tr G = 1, quad lim 1/N EE_F Tr G^2 = 1 + gamma, quad
    lim 1/N EE_F Tr G^3 = 1 + 3 gamma + gamma^2. $ <eq:wishart-moments>
  Eigenvectors are asymptotically delocalized (orthogonal invariance of the white
  Wishart ensemble).

_Proof._ Entries: direct Gaussian moments. Spectrum: Marchenko--Pastur
@marchenko1967distribution; moments are the Narayana polynomials of $gamma$.
$square$

== Two-point statistics: the sign-Gaussian (arcsine) law

*Statement T3* #THM (Price's theorem @price1958useful; proof recalled in
@app:price). _For $i != j$, conditional on $F$,_
$ EE_(x ~ P_F) [x_i x_j] = 2/pi thin arcsin(rho_(i j)), quad
  rho_(i j) := G_(i j)/sqrt(G_(i i) G_(j j)) = O_p (D^(-1\/2)). $ <eq:arcsine>

Together with $EE[x_i^2] = 1$, the full correlation matrix of $x$ under $P_F$ is
therefore the _sign-kernel matrix_
$ C_(i j) (F) := EE_(x~P_F) [x_i x_j] = cases(
  1 & i = j, \
  (2\/pi) arcsin(rho_(i j)) & i != j, ) $ <eq:signkernel>
with the entrywise expansion
$ C_(i j) (F) = 2/pi thin rho_(i j) + O_p (D^(-3\/2)) quad (i != j). $ <eq:arcsine-exp>

*Statement T4* #DER (higher moments).
All odd product moments vanish pointwise in $F$ by @eq:flip. Even moments are
multivariate normal orthant probabilities of $h$; to leading order in $1\/D$,
Isserlis' theorem applied to the first Hermite coefficient of $"sign"$ gives, for
distinct $i, j, k, l$,
$ EE_(x~P_F) [x_i x_j x_k x_l] = (2/pi)^2 (rho_(i j) rho_(k l) + rho_(i k) rho_(j l)
  + rho_(i l) rho_(j k)) + O(D^(-2)) = O(D^(-1)). $ <eq:fourpoint>
So $k$-point correlations at fixed $F$ are $O(D^(-(k-2)\/2))$: pairwise structure
dominates, but only at strength $O(D^(-1\/2))$ per pair. Whether this leading
Hermite truncation is innocuous in the proportional limit is precisely the
Gaussian-equivalence question of Section 5.3.

== Finite-$F$ versus disorder-averaged versus empirical laws

*Statement T5* #THM (the annealed law is exactly uniform). _For every $N, D$ and
every $x in {-1, +1}^N$,_
$ EE_F [P_F (x)] = 2^(-N). $ <eq:annealed-uniform>

_Proof._ Insert the definition and exchange the Gaussian averages (Fubini; the
integrand is bounded):
$ EE_F [P_F (x)] = integral D z thin EE_F product_(i=1)^N Theta(x_i (F z)_i). $
Fix $z != 0$. The numbers $(F z)_i = sum_nu F_(i nu) z_nu$ are independent across
$i$ (independent rows of $F$), each centered Gaussian with variance
$norm(z)^2\/D > 0$, hence each with a continuous symmetric law. Therefore
$EE_F Theta(x_i (F z)_i) = 1\/2$ exactly, factorization over $i$ gives
$EE_F product_i Theta(x_i (F z)_i) = 2^(-N)$, and the $z$-integral (the set
${z = 0}$ is measure zero) returns $2^(-N)$. $square$

_Remark._ @eq:annealed-uniform is exact at finite $(N, D)$, not an asymptotic
statement. Consequences:
+ All annealed moments vanish: $EE_F EE_(x~P_F) [x_(i_1) dots.c x_(i_k)] = 0$ for
  every nonempty set of distinct sites. In particular $EE_F C_(i j) (F) = 0$ for
  $i != j$ — consistent with @eq:arcsine since $arcsin$ is odd and $rho_(i j)$ is
  symmetric — while the quenched correlations satisfy
  $EE_F C_(i j)^2 (F) = (2\/pi)^2 thin 1\/D + O(D^(-2))$ for $i != j$.
+ Entropically: $H(EE_F P_F) = N log 2$, while the quenched entropy density
  $s(gamma) := lim 1/N thin EE_F H(P_F) < log 2$ is computed at the RS level in
  the companion note `notes/notes_hiddenmanifold.typ`. By concavity,
  $EE_F H(P_F) <= H(EE_F P_F)$, and the deficit
  $N (log 2 - s(gamma))$ is the total correlation (multi-information) of $P_F$ —
  the information-theoretic measure of the structure the learner must capture.

*Statement T6* #DER (macroscopic total correlation at fixed $F$). _Although each
off-diagonal entry of $C(F)$ is $O(D^(-1\/2))$, the squared Frobenius norm is
extensive:_
$ 1/N norm(C(F))_F^2 = 1 + (2/pi)^2 gamma + o(1), $ <eq:corr-frobenius>
both in expectation over $F$ and (by concentration of Wishart linear spectral
statistics) almost surely in the proportional limit.

_Proof._ $norm(C(F))_F^2 = N + sum_(i != j) (2\/pi)^2 arcsin^2 rho_(i j)$. Using
$arcsin^2 rho = rho^2 + O(rho^4)$, $rho_(i j)^2 = G_(i j)^2 (1 + O_p (D^(-1\/2)))$,
and @eq:wishart-moments:
$EE_F sum_(i != j) G_(i j)^2 = EE_F norm(G)_F^2 - EE_F sum_i G_(i i)^2
 = N(1 + gamma) - N(1 + 2\/D) + o(N) = N gamma + o(N)$, which gives
@eq:corr-frobenius. $square$

Statement T6 quantifies the sense in which $P_F$ is _weakly but extensively_
correlated: no single pair carries macroscopic correlation, yet the collective
correlation content per site is $O(1)$ and grows linearly with $gamma$.

*Statement T7* #DER (support of $P_F$; Cover's theorem @cover1965geometrical).
_For Gaussian $F$ the rows are in general position almost surely, hence the
support of $P_F$ is exactly the set of sign patterns of $N$ hyperplanes in
$RR^D$, of deterministic cardinality_
$ abs("supp"(P_F)) = T(N, D) := 2 sum_(k=0)^(D-1) binom(N-1, k) quad "a.s.", $
so that $log T(N, D) \/ N -> log 2$ for $gamma <= 2$ and $-> h_2 (1\/gamma)$ for
$gamma > 2$ (Cover dichotomy; cf. `notes_hiddenmanifold.typ`, eq. (s-hartley)).
On its support, $P_F (x)$ is the Gaussian volume ("Gardner volume") of the cone
${z : "sign"(F z) = x}$ and is strongly non-uniform. For $gamma > 2$ the support
is exponentially large, $T approx e^(N h_2 (1\/gamma))$, while the training set
is only polynomially large, $M = alpha D$: the empirical coverage fraction
$M\/T(N, D) -> 0$ exponentially in $N$. Consequently
$"TV"(hat(P)_M, P_F) -> 1$ whenever the min-entropy density
$s_oo (gamma) = lim -(1\/N) log max_x P_F (x)$ is strictly positive (expected for
all finite $gamma$; the value of $s_oo$ itself is only known at the heuristic/RS
level `notes_hiddenmanifold.typ`, eq. (s-min)). _The memorization question is
therefore not whether the empirical law approximates $P_F$ — it cannot — but how
the trained model resolves the choice between the $M$ seen atoms and the
exponentially many unseen ones._

= The masked linear score model <sec:model>

== Score function [DEF]

For a masked configuration represented as a tuple $(x, m)$, $m in {0,1}^N$
($m_j = 1$: masked), the linear score assigns to each masked site $i$ the logit
$ Lambda_i (x, m) = 1/sqrt(N) sum_(j=1)^N [W_(i j) (1 - m_j) x_j + V_(i j) m_j] + b_i,
  quad p_theta (x_i = +1 mid(|) x, m) = sigma(Lambda_i), $ <eq:logit>
with $sigma(u) = (1 + e^(-u))^(-1)$ and $sigma(-u) = 1 - sigma(u)$. Parameters
$theta = (W, V, b) in RR^(N times N) times RR^(N times N) times RR^N$. The mask
token in code is the in-band scalar $0$ (masked entries of `xt` vanish), which is
equivalent to @eq:logit; the mask indicator is recomputed as `xt == 0`
(`docs/ORIGINAL_ARCHITECTURE.md`). The runtime $1\/sqrt(N)$ factor is present in
the active package (`normalization="explicit_sqrt_n"`, default) and absent in the
legacy module (init-only, discrepancy D3): at fixed $N$ the two conventions are a
reparametrization of $(W, V)$ that also rescales the effective learning rate and
regularization strength; this note uses @eq:logit as written.

== Objective: continuous-time masked BCE [DEF]

With $y_i = (x_i + 1)\/2 in {0,1}$, the binary cross-entropy at site $i$ is
$-y_i log sigma(Lambda_i) - (1 - y_i) log(1 - sigma(Lambda_i)) = -log sigma(x_i Lambda_i)$.
The upstream objective (`diffusion.py:63-94`;
`src/maskeddiffusion/objectives.py`) is
$ cal(L)(theta) = -EE_(t ~ U(0,1)) 1/t thin EE_(x ~ hat(P)_M) thin EE_(m ~ "Bern"(t)^(times N))
  sum_(i : m_i = 1) log sigma(x_i Lambda_i (x, m)), $ <eq:masked-bce>
where one $t$ is drawn per sequence and coordinates are masked independently with
probability $t$. Because $PP(m_i = 1 mid(|) t) = t$, conditioning on $m_i = 1$
cancels the $1\/t$ weight against the mask probability, giving the equivalent
per-site form
$ cal(L)(theta) = sum_(i=1)^N cal(L)_i (W_i, V_i, b_i), quad
  cal(L)_i (w, v, b) = -1/M sum_(mu=1)^M EE_(t ~ U(0,1)) thin EE_(m_(\\ i) mid(|) t, thin m_i = 1)
  log sigma(x_i^mu Lambda_i (x^mu, m)). $ <eq:perneuron>

*Statement M1* #DER (exact output factorization). The objective
@eq:masked-bce decomposes exactly as a sum of $N$ independent per-neuron problems
@eq:perneuron with disjoint parameters $(W_i, V_i, b_i)$: there is no
cross-neuron coupling anywhere in the loss. All learning statements below are
therefore statements about $N$ independent (but statistically identical, up to the
quenched teacher) scalar problems. The code normalization $1\/(L dot B)$
(discrepancy D5) and the Monte-Carlo mask sampling (`mc_samples`) do not change
the minimizer; the theory computes the expectation objective and neglects
SGD/mask-sampling noise #HEUR.

== Regularization [DEF]

Training uses an $ell_2$ penalty on all trainable parameters,
$ R(theta) = lambda/2 (norm(W)_F^2 + norm(V)_F^2 + norm(b)^2), $
implemented as `l2coeff` $= lambda\/(2 M)$ multiplying the squared norm in the
loss (`diffusion.py:36`; the active path regularizes only trainable parameters,
fixing D6). With the $M$-scaled loss in the Gibbs measure below this matches the
replica convention $-(beta lambda\/2) norm(dot)^2$ of `notes_memorization.typ`.

== Gibbs measure over parameters [DEF]

For each output site $i$ define
$ Z_i (beta) = integral dif w thin dif v thin dif b thin
  exp(-beta M cal(L)_i (w, v, b) - beta lambda/2 (norm(w)^2 + norm(v)^2 + b^2)), $ <eq:gibbs>
and $chevron.l thin dot thin chevron.r_(beta, i)$ the associated Boltzmann average.
The trained model is the zero-temperature limit
$hat(theta)_i = lim_(beta -> oo) chevron.l (w, v, b) chevron.r_(beta, i)$.

*Statement M2* #THM (convexity). Each $cal(L)_i$ is a convex function of
$(w, v, b)$: it is an expectation of $-log sigma(x_i Lambda_i) =
"softplus"(-x_i Lambda_i)$, a convex function composed with a linear form of the
parameters. For $lambda > 0$ the regularized objective is strictly convex, the
minimizer is unique, and the Gibbs measure @eq:gibbs is log-concave. For
$lambda = 0$ the minimizer set is convex (possibly empty if the data are
separable; then the infimum is approached along diverging norms).

_Consequence._ The statistical mechanics of @eq:gibbs is *not* a spin-glass
problem: replica symmetry is the rigorously expected structure for convex
problems, with exact counterparts of the saddle equations below available in
principle through the convex-Gaussian-min-max theorem @thrampoulidis2018precise
or interpolation arguments @barbier2019optimal. The replica algebra of Sections
5–6 is used here as the bookkeeping device for the disorder average; we never
invoke replica-symmetry breaking for the weights, and label the saddle
computations #RS only because the $n -> 0$ continuation itself is unrigorous as
usual.

= The $V = 0$ question <sec:vzero>

The upstream notes assert for uniform data that the mask-channel weights shrink
to zero, $mu^v = 0$ and $q^v_(a b) = 0$, with the self-flagged caveat "[Make this
statement more precise!]" (`notes_memorization.typ:299`), and all hidden-manifold
experiments freeze $V equiv 0$ as an *ablation* (`docs/RESEARCH_SPEC.md`, open
question 1; discrepancy D7). We now separate what is exact, what is heuristic,
and what is conjecture. The key observation is that the operative symmetry is the
global spin flip @eq:flip, not coordinate exchangeability.

== Exact statements

*Statement V1* #THM (population-level vanishing of the mask channel and bias).
_Let $P$ be any law on ${-1,+1}^N$ with $P(-x) = P(x)$ for all $x$; let the mask
$m$ be drawn from any distribution independent of $x$ (any $t$-schedule and any
mask correlation structure); fix any output site $i$ and any nonnegative
$t$-weighting. Then the population masked-BCE risk_
$ R(w, v, b) := -EE_(x ~ P) thin EE_m thin w(t, m) thin log sigma(x_i Lambda_i (x, m)) $ <eq:poprisk>
_satisfies the involution symmetry_
$ R(w, v, b) = R(w, -v, -b) quad "for all" (w, v, b). $ <eq:vinv>
_Consequently: (i) for $lambda > 0$ the unique population minimizer has_
$v^* = 0$, $b^* = 0$; _(ii) for $lambda = 0$ the convex set of minimizers is
invariant under $(v, b) arrow.r (-v, -b)$ and hence contains a minimizer with
$v = 0$, $b = 0$; (iii) in all cases,_
$ inf_(w, v, b) R(w, v, b) = inf_(w, thin v = 0, thin b = 0) R(w, v, b). $

_Proof._ Split the logit @eq:logit into spin-coupled and spin-blind parts,
$Lambda_i (x, m) = Lambda^w_i (x, m) + Lambda^v_i (m) + b$, with
$Lambda^w_i (x, m) = (1\/sqrt(N)) sum_j W_(i j) (1 - m_j) x_j$ odd in $x$ and
$Lambda^v_i (m) = (1\/sqrt(N)) sum_j V_(i j) m_j$ independent of $x$. Under the
joint flip $x arrow.r -x$ at fixed $m$,
$ Lambda^w_i (-x, m) = -Lambda^w_i (x, m), quad Lambda^v_i (m) arrow.r Lambda^v_i (m), $
so the per-sample loss $ell(x, m; w, v, b) := -log sigma(x_i Lambda_i (x, m))$
satisfies the pointwise identity
$ ell(-x, m; w, v, b) = -log sigma((-x_i)(-Lambda^w_i + Lambda^v_i + b))
  = -log sigma(x_i (Lambda^w_i - Lambda^v_i - b)) = ell(x, m; w, -v, -b). $ <eq:ell-flip>
Averaging over $x ~ P$ with $P = P compose "flip"$ gives @eq:vinv. The risk is
convex in $(v, b)$ (softplus composed with an affine form, then expectation), and
strictly convex as soon as $lambda > 0$ (ridge) or as soon as the mask features
$m$ are not almost-surely collinear with the spin features (positive density of
$sigma' > 0$). Statements (i)–(iii) follow: a strictly convex invariant function
has a unique minimizer, which must be fixed by the involution, hence
$v^* = -v^* = 0$, $b^* = 0$; a convex invariant minimizer set contains the
midpoint of $(w, v, b)$ and $(w, -v, -b)$, which has $v = b = 0$; and (iii) is
$R(w, v, b) = (R(w, v, b) + R(w, -v, -b))\/2 >= R(w, 0, 0)$ by convexity.
$square$

*Statement V2* #THM (corollaries; where the theorem applies).
+ _Uniform data_ $P = 2^(-N)$: flip-symmetric; the upstream claim is made precise
  at the population level.
+ _Fixed finite $F$_: $P_F$ is flip-symmetric pointwise in $F$ (@eq:flip,
  Statement T1), so the population risk under $P_F$ has $v^* = b^* = 0$ for
  *every* teacher $F$ — no disorder average and no coordinate exchangeability is
  needed. This resolves the population half of `docs/RESEARCH_SPEC.md` open
  question 1 and of discrepancy D7.
+ _Disorder-averaged data_: $EE_F P_F = 2^(-N)$ exactly (Statement T5), so the
  annealed population problem is the uniform one.
+ The argument is representation-independent: it applies verbatim to the legacy
  in-band model (`xt@W' + m@V'`), to the active `explicit_sqrt_n` model, and to
  the `RandomFeatureScore` variant's mask channel, because in all of them the
  mask channel and bias are spin-blind while the data channel is spin-odd.

*Statement V3* #THM (finite training sets: exact symmetry of expectations).
_The empirical risk on the training set $X = {x^mu}$ satisfies the pointwise
dataset-flip identity_
$ hat(R)_X (w, v, b) = hat(R)_(-X) (w, -v, -b), $
and the training-set law is flip-invariant (i.i.d. samples from the
flip-symmetric $P_F$). Hence, whenever the empirical minimizer is almost-surely
unique (e.g. $lambda > 0$), its law satisfies
$ (hat(w), hat(v), hat(b)) stretch(=)^d (hat(w), -hat(v), -hat(b)),
  quad "so in particular" quad EE_X [hat(v)] = 0, quad EE_X [hat(b)] = 0. $ <eq:vemp>
_Moreover the same conjugation applied to the gradient-flow (or full-batch
gradient-descent) dynamics initialized at $V_0 = 0$, $b_0 = 0$ shows_
$EE_X [V_t] = 0$, $EE_X [b_t] = 0$ _at every optimization time $t$: the dataset
flip conjugates the $X$-trajectory to the $(-X)$-trajectory while leaving the
$W$-dynamics invariant, and $X stretch(=)^d -X$._

_Proof._ The first identity is @eq:ell-flip summed over the training set. For the
dynamical claim: with $eta_t$ the learning-rate schedule and loss
$hat(R)_X (W, V)$, the gradients satisfy
$partial_W hat(R)_X (W, V) = partial_W hat(R)_(-X) (W, -V)$ and
$partial_V hat(R)_X (W, V) = -partial_V hat(R)_(-X) (W, -V)$ by differentiating
the pointwise identity; thus the map $(W, V) arrow.r (W, -V)$ conjugates the two
flows, and the initial condition $(W_0, 0)$ is fixed by it. Averaging over
$X stretch(=)^d -X$ gives the claim. $square$

*Statement V4* #DER (what breaks the symmetry). The theorem and its corollaries
fail — by identifiable mechanisms — in the following cases:
+ _Non-flip-symmetric data._ If $P(-x) != P(x)$ (biased spins, binarized MNIST,
  any real dataset), the involution is not a symmetry of the risk and
  $v^*, b^* != 0$ generically. The mask channel is then genuinely useful.
+ _Finite training sets._ $hat(P)_M$ is not flip-symmetric at finite $M$
  (duplicates of flipped patterns occur with negligible probability), so
  $hat(v) != 0$ for every realized training set: @eq:vemp constrains only the
  *average* over training sets.
+ _Spin-coupled mask channels._ If the architecture lets mask features multiply
  spins (e.g. a separate embedding of the mask token mixed into the data
  channel), @eq:ell-flip fails.
+ _Objectives other than masked BCE._ The involution is a property of this loss
  and this parametrization; nothing is claimed for generation quality, which is a
  property of the sampler-indexed terminal law (Section 7.6).

== Heuristics and conjectures

*Statement V5* #HEUR (magnitude of the finite-sample mask channel). At finite
$M$ the empirical minimizer picks up the sampling-noise projection of the targets
onto the mask-feature block. A standard proportional-regime noise estimate
(curvature of the population risk against empirical gradient noise) gives, per
neuron, $norm(hat(v))^2 = O(N\/M)$ with a constant set by the loss curvature and
the ridge; with the code's logged observable $q V := "mean"(V^2) dot N$
(`models.py:order_parameters`) this reads
$ q V approx c(lambda, gamma, "t-schedule") dot gamma/alpha, $ <eq:qv-scaling>
i.e. an $O(1)$ logged quantity decaying like $1\/alpha$, and exactly centered:
$EE[hat(v)] = 0$ by @eq:vemp.

*Statement V6* #RS (replica shadow of the theorem). Inside the replica-symmetric
saddle of Section 6, the mask-channel order parameters obey
$(nu, nu^(hat())) = 0$ exactly (the energetic functional is even in $nu$,
Statement E4) and the zero-temperature saddle has $q^v = delta q^v = 0$ at leading
order: the replica computation reproduces the population theorem as the
$M -> oo$ limit and is compatible with the subleading @eq:qv-scaling. Whether the
$q^v = 0$ saddle is the global one within the RS class follows from the
variance-monotonicity of the loss (Statement E5).

*Statement V7* #CONJ. The finite-sample mask channel is pure overfitting noise:
it does not improve any sampler-indexed terminal observable, and the residual
Model-vs-True MMD gap (open empirical observation, commit `2e2db70`) is not
caused by the $V equiv 0$ restriction. This is testable (Section 9, experiment
E-V1) but is *not* implied by Statement V1, which concerns only the BCE
objective.

== Summary of the $V = 0$ status

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    [*Claim*], [*Scope*], [*Status*],
    [$v^* = b^* = 0$ at the population level], [any flip-symmetric law, incl. fixed $F$], [#THM exact (V1, V2)],
    [$EE_X [hat(v)] = 0$, $EE_X [V_t] = 0$], [finite $M$, unique minimizer / flow from $V_0 = 0$], [#THM exact (V3)],
    [$q V approx c gamma \/ alpha$], [finite-$M$ magnitude], [#HEUR (V5)],
    [$q^v = 0$ at the RS saddle, $beta -> oo$], [proportional limit], [#RS (V6, Section 6.4)],
    [$V equiv 0$ harmless for generation], [terminal laws, MMD gap], [#CONJ (V7)],
  ),
  caption: [The $V = 0$ question, separated by evidence class. The population
    statement is a theorem pointwise in $F$; the finite-sample magnitude and all
    sampler-level statements remain open.],
) <tab:vzero>

= Replica setup <sec:replica>

== Goal and disorder averages

We compute the typical value of the per-neuron free entropy in the proportional
limit,
$ phi(beta) = lim_(D -> oo) 1/N thin EE_(F, Z) log Z_i (beta), quad
  gamma = N/D, thin alpha = M/D " fixed", $
where the disorder is: (i) the teacher $F$ (quenched); (ii) the training latents
$Z = {z^mu}_(mu=1)^M$ (quenched); (iii) the masks, which appear only through the
expectation objective @eq:masked-bce (annealed into $ell$ by construction); the
per-neuron factorization (Statement M1) is used throughout. The average is the
quenched one, $EE log Z$, via $EE log Z = lim_(n -> 0) (EE Z^n - 1)\/n$; the
annealed average $log EE Z$ is a different object and is not the typical one
(Section 8). Because the training points are i.i.d. given $F$, the data average
factorizes exactly:
$ EE_(F,Z) [Z_i (beta)^n] = EE_F integral product_(a=1)^n dif w^a thin dif v^a thin dif b^a thin
  e^(-beta lambda/2 sum_a (norm(w^a)^2 + norm(v^a)^2 + (b^a)^2))
  (EE_(x ~ P_F) thin e^(-beta sum_(a=1)^n ell_a (x)))^M, $ <eq:replicatedZ>
with $ell_a (x) = -EE_(t ~ U(0,1)) thin EE_(m_(\\ i) mid(|) t, thin m_i = 1)
log sigma(x_i Lambda^a_i (x, m))$ the per-datum, mask-averaged loss of replica
$a$.

== Mask central limit theorem

*Statement R1* #DER (carried over from `notes_memorization.typ`). Fix
$t, x, w^a, v^a$ and draw $m$ with $m_i = 1$. The preactivation
$z^a := Lambda^a_i (x, m)$ (without bias) is asymptotically Gaussian with
$ EE_m [z^a] = (1 - t) mu^a (x) + t nu^a, quad
  Var_m [z^a] = t(1-t)(q^a + q^(v,a)) + o(1), $
where
$ mu^a (x) := 1/sqrt(N) sum_j w^a_j x_j, quad nu^a := 1/sqrt(N) sum_j v^a_j, quad
  q^a := norm(w^a)^2/N, quad q^(v,a) := norm(v^a)^2/N. $ <eq:op-first>
_Proof sketch._ Expand the square:
$Var_m z^a = (t(1-t)\/N) sum_j (v^a_j - w^a_j x_j)^2$. The mixed term
$(2\/N) sum_j v^a_j w^a_j x_j$ has mean zero over $x ~ P_F$ pointwise in $F$
(Statement T1) and fluctuations $O(N^(-1\/2))$ under delocalization, so it drops.
$square$

The mask-averaged per-datum loss therefore depends on the weights only through
the field $mu^a (x)$ and the scalars $(nu^a, q^a + q^(v,a))$:
$ ell(mu, s^2, nu) := -EE_(t ~ U(0,1)) thin EE_(y ~ cal(N)(0,1))
  log sigma(sqrt(t(1-t) thin s^2) thin y + (1-t) mu + t nu), quad s^2 = q^a + q^(v,a). $ <eq:lossfield>

*Statement R2* #DER (two elementary properties of @eq:lossfield).
+ $ell(mu, s^2, nu)$ is increasing in $s^2$: writing $u(v) =
  EE_y ["softplus"(-(sqrt(v) y + m))]$ one finds $u'(v) =
  (1\/2) EE_y [sigma'(sqrt(v) y + m)] > 0$ by Gaussian integration by parts.
  Added mask noise strictly increases the loss.
+ Under the flip symmetry, $ell$ enters the energetic functional below only
  through a centered Gaussian field, making the latter even in $nu$; combined
  with R2(1) this is the replica-level mechanism behind the $V = 0$ theorem
  (Statement V6).

== Teacher statistics and the Gaussian-equivalence step

It remains to describe the law of the signal fields ${mu^a (x)}_(a=1)^n$ when
$x ~ P_F$. Their exact first two moments follow from Section 2:
$ EE_(x~P_F) [mu^a (x)] = 0, quad
  EE_(x~P_F) [mu^a mu^b] = 1/N sum_(j k) w^a_j w^b_k C_(j k) (F), $ <eq:mucov>
with $C(F)$ the sign-kernel matrix @eq:signkernel. Using the Hermite
decomposition of the sign function,
$"sign"(h) = kappa_1 h + kappa_2 xi + "(higher Hermite terms)"$,
$kappa_1 = EE[h thin "sign"(h)] = sqrt(2\/pi)$,
$kappa_2^2 = 1 - kappa_1^2 = 1 - 2\/pi$, @eq:mucov reads, to leading order in
$1\/D$ (cf. @eq:arcsine-exp),
$ EE [mu^a mu^b] = kappa_2^2 thin q_(a b) + kappa_1^2 thin psi_(a b) + o(1), quad
  C_(a b) := kappa_2^2 thin q_(a b) + kappa_1^2 thin psi_(a b), $ <eq:Cab>
in terms of the two overlap families
$ q_(a b) := 1/N w^a dot w^b, quad
  psi_(a b) := 1/N (w^a)^top G thin w^b = u^a dot u^b, quad
  u^a := 1/sqrt(N) F^top w^a in RR^D. $ <eq:psi-op}

*Statement R3* #GET. In the proportional limit, the empirical risk of the convex
per-neuron problem on sign-teacher data equals that on the equivalent Gaussian
covariates $tilde(x) = kappa_1 h + kappa_2 xi$, $h ~ cal(N)(0, G)$,
$xi ~ cal(N)(0, I_N)$ — i.e. the joint law of the fields ${mu^a}$ under the
Gibbs measure is Gaussian with covariance @eq:Cab. For convex generalized linear
models on random-feature data this is a theorem of the literature
@goldt2020hidden @geraceGeneralisationErrorLearning2021 @loureiro2021learning
@hu2022universality under regularity conditions not verified for the masked-BCE
objective here; we adopt it as a standard, uncontrolled-here step. Note that the
diagonal of @eq:Cab at $w^a = w^b$ is the full field variance
$C_(a a) = kappa_2^2 q_(a a) + kappa_1^2 psi_(a a)$, which differs from the mask
-noise parameter $s^2 = q_(a a) + q^(v,a a)$ of @eq:lossfield: the norm of $w$
and the teacher-projected norm play *different* roles and must not be merged.

*Statement R4* #DER (why $psi$ is a genuine order parameter). For a generic
(delocalized) $w$ independent of $F$, $psi = (1\/N) w^top G w -> q$ by
concentration, since $EE_F psi = q (1 + O(D^(-1)))$ and
$Var_F (psi mid(|) w) = 2 q^2\/D$. The Gibbs measure, however, is tilted by the
energetic term, which depends on $psi$ itself through @eq:Cab: the learned $w$
aligns with the teacher's eigenstructure, so at the saddle $psi != q$
generically. Dropping $psi$ would reduce the theory to isotropic (uniform) data
and miss the teacher alignment — cf. the conjugate-ratio statement S4 below.

== Order parameters, conjugate fields, replicated free entropy

Enforcing the definitions @eq:op-first and @eq:psi-op with Fourier-Lagrange
multipliers $(hat(q)_(a b), hat(psi)_(a b), hat(q)^v_(a b), hat(nu)_a)$ gives the
replicated free entropy (per visible site, as in `notes_memorization.typ`)
$ Phi_n = extr_(q, psi, q^v, nu, hat(q), hat(psi), hat(q)^v, hat(nu)) thin
  lr[-1/(2 n) sum_(a b) (q_(a b) hat(q)_(a b) + psi_(a b) hat(psi)_(a b) + q^v_(a b) hat(q)^v_(a b))
     - 1/n sum_a nu_a hat(nu)_a + G_S + M/N thin G_E], $ <eq:free-entropy}
with $M\/N = alpha\/gamma$ the `visible_load`, and the entropic and energetic
contributions
$ G_S = 1/(n N) log integral product_a dif w^a dif v^a dif b^a thin
  e^(-beta lambda/2 sum_a (norm(w^a)^2 + norm(v^a)^2 + (b^a)^2)
     + 1/2 sum_(a b) [hat(q)_(a b) thin w^a dot w^b + hat(psi)_(a b) (w^a)^top G w^b + hat(q)^v_(a b) thin v^a dot v^b]
     + sum_a hat(nu)_a thin 1/sqrt(N) sum_j v^a_j), $ <eq:GS}
$ G_E = 1/n log EE_({mu^a} ~ cal(N)(0, C)) thin
  e^(-beta sum_a ell(mu^a, q_(a a) + q^v_(a a), nu_a)). $ <eq:GE}
The bias $b$ enters exactly like the component of $nu$ along the all-ones mask
and is suppressed for readability; its saddle is zero by Statement V1.

_Remark._ @eq:free-entropy is the quenched object: $F$ enters through $G$ inside
the entropic volume @eq:GS (a fixed Wishart quadratic form) and through $psi$
inside the energetic term @eq:GE. Annealing over $F$ instead would replace
$P_F$ by $EE_F P_F = 2^(-N)$ (Statement T5) and collapse the theory to the
uniform-data one — a useful consistency check, not the object of interest.

= Replica-symmetric theory <sec:rs>

== Replica-symmetric parametrization

Within each overlap family we impose permutation symmetry among the $n$ replicas:
$ q_(a a) = q + delta q, quad q_(a b) = q; quad
  psi_(a a) = psi + delta psi, quad psi_(a b) = psi; quad
  q^v_(a a) = q^v + delta q^v, quad q^v_(a b) = q^v quad (a != b), $ <eq:rs-ansatz>
with the conjugates $(hat(q), delta hat(q), hat(psi), delta hat(psi), hat(q)^v,
delta hat(q)^v, hat(nu))$ defined identically, and $nu_a = nu$. By Statement M2
the problem is convex, so this ansatz is the expected-exact structure; the label
#RS refers to the unrigorous $n -> 0$ continuation, not to any glassiness
assumption.

== Entropic contribution

*Statement S1* #RS (entropic term). _Under @eq:rs-ansatz and the concentration of
Wishart linear spectral statistics (Statement T2), the entropic contribution
@eq:GS evaluates to_
#box(stroke: 0.5pt, inset: 6pt)[
$ G_S = -1/2 lr[ cal(L)_gamma (beta lambda - delta hat(q), thin -delta hat(psi))
  - cal(T)_gamma (beta lambda - delta hat(q), thin -delta hat(psi); thin hat(q), hat(psi))]
  + (hat(q)^v + hat(nu)^2)/(2 (beta lambda - delta hat(q)^v)), $ <eq:GS-RS>
]
_where the two Marchenko--Pastur transforms_
$ cal(L)_gamma (a, b) := integral rho_gamma (s) thin log(a + b s) thin dif s, quad
  cal(T)_gamma (a, b; thin c, d) := integral rho_gamma (s) thin (c + d s)/(a + b s) thin dif s, $ <eq:MP-transforms>
_are scalar functions of the Wishart spectrum $rho_gamma$, computable in closed
form from the MP Stieltjes transform (@app:mp)._

_Derivation._ With the RS blocks $hat(q)_(a b) = delta hat(q) thin delta_(a b) +
hat(q)$ and $hat(psi)_(a b) = delta hat(psi) thin delta_(a b) + hat(psi)$, the
$w$-sector of @eq:GS is a Gaussian integral with kernel
$cal(M) = beta lambda thin I_(n N) - hat(q)_"RS" times.o I_N - hat(psi)_"RS" times.o G$,
where $hat(q)_"RS" = delta hat(q) thin I_n + hat(q) thin J_n$,
$hat(psi)_"RS" = delta hat(psi) thin I_n + hat(psi) thin J_n$ with $J_n$ the
all-ones matrix. Both RS blocks are polynomials of $J_n$ and commute; the
representation splits into the subspace orthogonal to $bold(1)$ (multiplicity
$n - 1$), where $cal(M)$ acts as
$A_0 := (beta lambda - delta hat(q)) thin I_N - delta hat(psi) thin G$, and the
$bold(1)$-line (multiplicity 1), where it acts as
$A_1 = A_0 - n B$, $B := hat(q) thin I_N + hat(psi) thin G$. Hence
$ (1/(n N)) log det cal(M) = (1/(n N)) [(n - 1) log det A_0 + log det (A_0 - n B)]
  -> 1/N log det A_0 - 1/N Tr (A_0^(-1) B), $
using $log det(A_0 - n B) = log det A_0 - n thin Tr(A_0^(-1) B) + O(n^2)$. The
Gaussian integral contributes $-(1\/(2 n N)) log det cal(M)$; Wishart
concentration $(1\/N) log det(a I_N + b G) -> cal(L)_gamma (a, b)$,
$(1\/N) Tr[(a I_N + b G)^(-1)(c I_N + d G)] -> cal(T)_gamma (a, b; c, d)$ gives
the first bracket of @eq:GS-RS. The $v$-sector is the upstream one ($G$ replaced
by $I_N$, no teacher coupling) and contributes
$hat(q)^v\/(2(beta lambda - delta hat(q)^v))$; the $hat(nu)$-linear term
contributes $hat(nu)^2\/(2(beta lambda - delta hat(q)^v))$ after evaluating the
shifted Gaussian on the $bold(1)$ mode. $square$

_Remark._ For $G = I_N$ (the $gamma -> 0$ degenerate Wishart) the first bracket
reduces to $-(1\/2) log(beta lambda - delta hat(q) - delta hat(psi)) +
(hat(q) + hat(psi))\/(2(beta lambda - delta hat(q) - delta hat(psi)))$; the
logarithmic piece, absent from `notes_memorization.typ`, contributes only at
$O(log beta \/ beta)$ at zero temperature and leaves all upstream zero-temperature
saddle equations and the train-loss identity unchanged.

== Energetic contribution

*Statement S2* #RS (energetic term). _With the covariance kernel @eq:Cab and its
RS parameters_
$ C := kappa_2^2 thin q + kappa_1^2 thin psi, quad
  delta C := kappa_2^2 thin delta q + kappa_1^2 thin delta psi, $ <eq:C-RS>
_the energetic contribution @eq:GE evaluates, by the same inverse-covariance and
Hubbard--Stratonovich algebra as in `notes_memorization.typ`, to_
#box(stroke: 0.5pt, inset: 6pt)[
$ G_E = integral D X thin log integral (dif mu)/sqrt(2 pi) thin
  e^(- mu^2/(2 delta C) + (sqrt(C)/delta C) X mu - beta thin ell(mu, s^2, nu))
  - 1/2 log delta C - C/(2 delta C), quad
  s^2 = q + delta q + q^v + delta q^v, $ <eq:GE-RS>
]
_with $ell$ the mask-averaged loss field @eq:lossfield. The only change relative
to the uniform-data theory is the covariance substitution
$(q, delta q) arrow.r (C, delta C)$ in the field law; the mask noise $s^2$ keeps
the plain weight norms (Statement R3)._

_Derivation sketch._ The RS covariance matrix @eq:Cab has inverse
$C^(-1)_(a a) = (delta C + (n-1) C)\/(delta C (delta C + n C))$,
$C^(-1)_(a b) = -C\/(delta C (delta C + n C))$, determinant
$det = delta C^(n-1) (delta C + n C)$; the quadratic form in the Gaussian density
is linearized by one Hubbard--Stratonovich field $X$ per site, after which the
$n$ replicas factorize and the $n -> 0$ limit gives @eq:GE-RS. $square$

*Statement S3* #DER (structure of $G_E$). From Statement R2: (i) $G_E$ is even in
$nu$ (the Gaussian field is centered and the data law flip-symmetric); (ii) $G_E$
is non-increasing in $s^2$ at fixed $(C, delta C, nu)$, strictly so wherever
$sigma' > 0$. These two properties are the energetic input to the mask-channel
saddle (V6).

== Mask-channel sector: the $q^v = 0$ saddle

*Statement S4* #RS (mask-channel stationarity; the replica content of Statement
V6). _The stationarity equations of @eq:free-entropy in the mask-channel sector
read_
$ nu = hat(nu)/(beta lambda - delta hat(q)^v), quad
  hat(nu) = M/N thin partial_nu G_E, quad
  delta q^v = 1/(beta lambda - delta hat(q)^v), quad
  q^v + delta q^v = (hat(q)^v + hat(nu)^2)/(beta lambda - delta hat(q)^v)^2, $
$ delta hat(q)^v = 0, quad
  hat(q)^v = 2 thin M/N thin partial_(s^2) G_E = hat(q)^((w) "diag"). $
_In particular: (i) $nu = hat(nu) = 0$ exactly, since $partial_nu G_E = 0$ by the
evenness of $G_E$ (Statement S3); (ii) at zero temperature, $hat(q)^v = O(beta)$
(the diagonal/noise response), while the signal-sustaining responses
@eq:sp-energetic below are $O(beta^2)$, so $q^v = delta q^v = 0$ at leading
order: the RS saddle reproduces the population theorem of Section 4 in the
proportional limit. The finite-$M$ noise remainder is the $O(1\/N)$ correction
captured by @eq:qv-scaling._

_Derivation._ The first line is the $v$-sector entropic stationarity; the second
line is the energetic stationarity, using that $G_E$ depends on the off-diagonal
$q^v$ only trivially (the mask CLT variance involves the diagonal norms only,
Statement R1), whence $delta hat(q)^v = 0$, and that
$partial_(q^v + delta q^v) G_E = partial_(s^2) G_E$ equals the corresponding
diagonal $w$-derivative. The zero-temperature bookkeeping follows the scaling of
Section 6.5. $square$

== Zero-temperature saddle-point equations

We apply the upstream zero-temperature scalings, extended to the teacher channel:
$ delta q arrow.r delta q \/ beta, quad hat(q) arrow.r beta^2 hat(q), quad
  delta hat(q) arrow.r beta delta hat(q), quad
  delta psi arrow.r delta psi \/ beta, quad hat(psi) arrow.r beta^2 hat(psi), quad
  delta hat(psi) arrow.r beta delta hat(psi), $ <eq:zerot-scaling}
under which $phi := lim_(beta -> oo) Phi_0 \/ beta$ is finite and the mask-noise
parameter $s^2 -> q + delta q\/beta + q^v + delta q^v\/beta -> q$ (at the $q^v = 0$
saddle).

*Statement S5* #RS (saddle-point system). _The zero-temperature RS saddle is
determined by eight scalar equations in
$(q, delta q, psi, delta psi; hat(q), delta hat(q), hat(psi), delta hat(psi))$.
With the shorthands $a := lambda - delta hat(q)$, $b := delta hat(psi)$, and the
MP resolvents of @app:mp, the entropic (random-matrix) half is_
#box(stroke: 0.5pt, inset: 6pt)[
$ delta q = integral rho_gamma (s) thin 1/(a - b s) thin dif s, quad
  delta psi = integral rho_gamma (s) thin s/(a - b s) thin dif s, $ <eq:sp-entropic1>
$ q = integral rho_gamma (s) thin (hat(q) + hat(psi) s)/(a - b s)^2 thin dif s, quad
  psi = integral rho_gamma (s) thin s (hat(q) + hat(psi) s)/(a - b s)^2 thin dif s, $ <eq:sp-entropic2>
]
_and the energetic half is_
#box(stroke: 0.5pt, inset: 6pt)[
$ delta hat(q) = 2 thin M/N thin kappa_2^2 thin partial_C tilde(G)_E, quad
  delta hat(psi) = 2 thin M/N thin kappa_1^2 thin partial_C tilde(G)_E, $ <eq:sp-energetic>
$ hat(q) = 2 thin M/N thin kappa_2^2 thin partial_(delta C) tilde(G)_E, quad
  hat(psi) = 2 thin M/N thin kappa_1^2 thin partial_(delta C) tilde(G)_E, $ <eq:sp-energetic2>
]
_where_
$ tilde(G)_E (C, delta C) = integral D X thin max_mu
  lr[- mu^2/(2 delta C) + sqrt(C)/delta C thin X mu - ell(mu, q)] - C/(2 delta C), $ <eq:GE-zeroT}
_with $ell(mu, q) = -EE_t thin EE_y thin log sigma(sqrt(t(1-t) q) thin y +
(1-t) mu)$, the maximizing field $mu^* (X)$ solving the implicit equation_
$ mu^* (X)/delta C = sqrt(C)/delta C thin X - partial_mu ell(mu^* (X), q), $ <eq:mustar}
_and the derivatives given by the upstream formulas with $(q, delta q) arrow.r
(C, delta C)$:_
$ partial_C tilde(G)_E = integral D X thin lr[(X mu^* (X))/(2 delta C sqrt(C))
  - partial_q ell(mu^* (X), q)] - 1/(2 delta C), quad
  partial_(delta C) tilde(G)_E = 1/2 integral D X thin
  (mu^* (X)/delta C - X sqrt(C)/delta C)^2. $ <eq:GE-derivs}

*Statement S6* #RS (fixed conjugate ratio). _The two energetic response channels
enter with fixed Hermite weights:_
$ hat(psi)/hat(q) = delta hat(psi)/delta hat(q) = kappa_1^2/kappa_2^2
  = (2\/pi)/(1 - 2\/pi) = 2/(pi - 2) approx 1.752, $ <eq:conj-ratio}
_so the teacher channel is driven with a strength pinned by the sign nonlinearity;
the learned $W$ is generically non-isotropic ($psi > q$, Statement R4), with
$psi = q$ only at $gamma = 0$._

== Consistency checks

*Statement S7* #DER (uniform-data reduction, $gamma arrow.r 0$). For
$G arrow.r I_N$ (the degenerate Wishart), $psi equiv q$ identically,
$kappa_1^2 + kappa_2^2 = 1$ gives $C = q$, $delta C = delta q$, the two conjugate
pairs merge into $hat(q)_"eff" = hat(q) + hat(psi)$,
$delta hat(q)_"eff" = delta hat(q) + delta hat(psi)$, and
@eq:sp-entropic1–@eq:sp-energetic2 collapse to the upstream saddle system of
`notes_memorization.typ` (their eq. (SP)) with $alpha_"legacy" = M\/N$.

*Statement S8* #DER (empty-data limit, $alpha arrow.r 0$). The energetic
responses vanish, $hat(q) = hat(psi) = delta hat(psi) = 0$,
$delta hat(q) = lambda$; then @eq:sp-entropic1 gives $delta q = 1\/lambda$ and
@eq:sp-entropic2 gives $q = psi = 0$, matching the upstream $alpha = 0$
solution ($delta q = 1\/lambda$).

== Status of the computation and remaining unsolved integrals

Exact within the replica framework: the output factorization (M1), the mask CLT
(R1), the Wishart transforms (T2, S1 given the RS parametrization), the convexity
consequences (M2), and the reduction checks (S7, S8). Uncontrolled or unfinished:
+ #GET the replacement of the sign-teacher field law by its Gaussian equivalent
  (Statement R3) — a theorem of the literature under conditions not verified for
  this objective.
+ #RS the $n arrow.r 0$ continuation itself.
+ The MP transforms @eq:MP-transforms and the resolvents of
  @eq:sp-entropic1–@eq:sp-entropic2 have closed forms via the MP Stieltjes
  transform (@app:mp), but the coupled system
  @eq:sp-entropic1–@eq:sp-energetic2 together with the one-dimensional Gaussian
  quadrature @eq:GE-zeroT–@eq:mustar has *not been solved numerically*: this is
  the computational task of Phase 4C (experiment E-S1), extending
  `julia-code/hiddenmanifold`.
+ The order $lambda arrow.r 0^+$ versus $beta arrow.r oo$ (at $lambda = 0$ and
  separable data the minimizer escapes to infinite norm; cf. the upstream
  $delta q = 1\/lambda$ divergence) is left at the level of Statement M2's
  caveat.
+ Nothing in this section touches the joint-law consistency of the trained
  conditionals (Section 7.6).

= From order parameters to observables <sec:observables>

All formulas below are evaluated at the saddle $(q^*, delta q^*, psi^*, delta
psi^*)$ of Statement S5 (with $C^*, delta C^*$ from @eq:C-RS and $q^v = 0$).

== Train and test loss

*Statement O1* #RS (train loss). _At the saddle, the per-datum training loss is
the energetic integrand at the maximizing field,_
$ cal(L)_"train" = EE_(t ~ U(0,1)) thin EE_(y ~ cal(N)(0,1)) thin
  lr[-log sigma(sqrt(t(1-t) thin q^*) thin y + (1-t) thin mu^* (X))]_(X ~ cal(N)(0,1)), $ <eq:ltrain}
_and equivalently, from $(1\/N) log Z_i = -beta [(alpha\/gamma) cal(L)_"train" +
(lambda\/2) q^*]$ at the minimizer, the thermodynamic identity_
$ cal(L)_"train" = -gamma/alpha thin (phi + lambda q^* \/ 2), $ <eq:ltrain-id}
_with $phi = lim_(beta -> oo) Phi_0\/beta$ the zero-temperature free entropy per
visible site._

*Statement O2* #RS (test loss). _For a fresh datum $x ~ P_F$ (same quenched $F$,
fresh latent), the field $mu(x) = (1\/sqrt(N)) w dot x$ is Gaussian with variance
$C^*$ (Statement R3), so_
$ cal(L)_"test" = EE_(t ~ U(0,1)) thin EE_(mu ~ cal(N)(0, C^*)) thin EE_(y ~ cal(N)(0,1))
  [-log sigma(sqrt(t(1-t) thin q^*) thin y + (1-t) thin mu)]. $ <eq:ltest}
_The generalization (memorization) gap $Delta = cal(L)_"test" -
cal(L)_"train" >= 0$ is a direct prediction for the masked-BCE generalization
curve in $(alpha, gamma, lambda)$._

== Time-sliced accuracy and U-turn reconstruction

*Statement O3* #RS (time-sliced test accuracy; upstream derivation with
$(q, delta q) arrow.r (C, delta C)$). _The probability of correctly resolving a
masked site of a fresh datum, given a fraction $1 - t$ of correctly unmasked
tokens, is_
$ cal(E)_t^"fair" = integral D X thin D y thin
  sigma(sqrt(t(1-t) q^*) thin y + (1-t) mu^* (X)), $ <eq:acc-fair}
$ cal(E)_t^"greedy" = integral D X thin
  Phi(mu^* (X)/sqrt(q^*) thin sqrt((1-t)/t)), $ <eq:acc-greedy}
_with $mu^* (X)$ from @eq:mustar and $Phi$ the standard normal CDF. The noise
under the square root is the plain norm $q^*$ (mask randomness), while $mu^*$
depends on $(C^*, delta C^*)$ (Statement R3)._

*Statement O4* #MF (U-turn overlap; uncontrolled closure). _Neglecting the
correlations developed between committed tokens during sequential generation
(the same closure as `notes_memorization.typ`, §"time-integrated accuracy"), the
magnetization $m_t = c_t - e_t$ relative to a training datum, initialized at
$m_(t_0) = 1 - t_0$, obeys_
$ (dif m_t)/(dif t) = 1 - 2 cal(E)_t (m_t), quad
  cal(E)_t (m) = integral D X thin D y thin
  sigma(sqrt(q^* ((1-t) - m^2)) thin y + m thin mu^* (X)), $ <eq:uturn-ode}
_and the U-turn overlap is $q_U^(t_0) = m_0$. This is the mean-field closure; it
is not a consequence of the saddle equations and its error is unquantified._

== Overlap laws and the information content of the MMD diagnostic

*Statement O5* #DER (true-true overlap law). _For $x, y$ i.i.d. from $P_F$
(quenched $F$), the normalized Hamming overlap $q(x, y) = x dot y \/ N$
satisfies, pointwise in $F$,_
$ EE[q] = 0, quad Var(q) = 1/N^2 norm(C(F))_F^2
  = 1/N (1 + kappa_1^4 thin gamma) + o(1/N), $ <eq:overlap-tt}
_by Statement T6; and $q$ is asymptotically Gaussian (CLT for the weakly
correlated summands $x_i y_i$) #HEUR, so $q => cal(N)(0, (1 + kappa_1^4
gamma)\/N)$. The excess over the uniform-data value $1\/N$ is the teacher-induced
correlation of Statement T6._

*Statement O6* #DER (kernel concentration and the scale of the MMD). _The MMD
kernel $k_lambda (x, y) = exp(-lambda (1 - q)\/2)$ depends on $(x, y)$ only
through the overlap. For any pair of laws whose overlaps concentrate at $0$ with
variance $sigma^2\/N$,_
$ EE k_lambda = e^(-lambda\/2) thin e^(lambda^2 sigma^2 \/ (8 N)) thin (1 + o(1)), $
_and the unbiased MMD$^2$ between sample sets $X, Y$ is_
$ "MMD"^2_lambda (X, Y) = e^(-lambda\/2) thin lambda^2/(8 N) thin
  (sigma^2_(X X) + sigma^2_(Y Y) - 2 sigma^2_(X Y)) + o(1/N). $ <eq:mmd-scaling}
_Three consequences:_
+ _At fixed $lambda$, MMD$^2$ is an $O(1\/N)$ quantity in the proportional limit:
  the entire signal of this diagnostic is subleading. This is consistent with the
  repository's finite-size regime and with negative unbiased estimates being
  legitimate (`docs/RESEARCH_SPEC.md`), and it predicts an $N$-collapse of all
  MMD curves at fixed $lambda$ (experiment E-O4)._
+ _The kernel is blind to everything except the law of the overlap: two
  distributions with the same overlap law are indistinguishable to this MMD at
  any order in $1\/N$. In particular, perfect memorization (terminal law $=$
  $hat(P)_M$) and perfect generalization (terminal law $= P_F$) share the same
  leading overlap law — training points are themselves i.i.d. $P_F$ samples — so
  the fixed-$lambda$ MMD cannot separate these extremes at leading order._
+ _The memorization signature lives in the model-model_ self_-overlap law: if the
  terminal law collapses onto memorized patterns, two independently generated
  samples collide on the same pattern with nonzero probability, giving
  $EE[q_"mm"] > 0 = EE[q_"tt"]$ and inflating $EE k("mm")$. The two-point
  generalization signature is instead $Delta sigma^2 := N thin (Var q_"mt" - Var
  q_"tt")$. Both are directly measurable, kernel-free observables (E-O4)._

The model-side overlap laws ($q_"mm"$, $q_"mt"$) depend on the terminal law
$P_(theta, A, k)$ and are not computed here (Section 7.6); only the true-side law
(O5) is exact.

== Memorization and retrieval metrics

*Statement O7* #HEUR (memorization observables). The repository's memorization
diagnostics — nearest-training overlap (`nearest_training_overlap`,
`nearest_training_excess`), top-$K$ overlaps, and the Hamming memorized fraction
— are functions of the terminal law and of the retrieval channel. Within the
present theory: (i) the map $mu^* (X)$ of @eq:mustar is the single-site retrieval
channel and @eq:uturn-ode its sequential closure; (ii) from-scratch collapse onto
training points requires the full terminal law (O6(c) gives its overlap-law
signature); (iii) the upstream conjecture that perfect retrieval coexists with
full memorization for $M = O(L\/log L)$ (`notes_memorization.typ`) is unchanged
by the teacher and remains #CONJ.

== Pair correlations

Teacher side: exact, $C_(i j) (F) = (2\/pi) arcsin(rho_(i j))$ (Statement T3),
and the code exposes it as `teacher.correlation_matrix()`
(`src/maskeddiffusion/teacher.py`). Model side: the terminal-law correlation
$EE_(P_(theta, A, k)) [x_i x_j]$ requires the joint law (open). What the saddle
determines is the _conditional-level_ structure: the linear response
$partial Lambda_i \/ partial x_j = W_(i j)\/sqrt(N)$ and the singular spectrum of
$W$, which is the deformed Wishart law implicit in the resolvent
@eq:sp-entropic1–@eq:sp-entropic2 (experiment E-S3) #RS.

== Sampler-indexed terminal laws

The trained object is a collection of single-site masked conditionals
$p_theta (x_i mid(|) x, m)$; any generative use composes them through a named
sampler $A$ resolving $k$ tokens per step, producing the terminal law
$P_(theta, A, k)$ (`docs/RESEARCH_SPEC.md`). The present theory makes
conditional-level predictions (losses O1–O2, accuracies O3) that are
sampler-free, and terminal-level predictions only through the uncontrolled
closures O4 and O6. Whether the trained conditionals are mutually consistent with
a joint law is open (`docs/RESEARCH_SPEC.md`, open question 3); until resolved,
no sampler here is exact ancestral sampling, and every generated-sample
observable must carry its $(A, k)$ index.

= Limits: order and meaning <sec:limits>

+ *Proportional limit.* $D -> oo$ at fixed $(gamma, alpha, lambda)$, with
  $N = "round"(gamma D)$ and $M = "round"(alpha D)$ (`docs/NOTATION.md` rounding
  rules; the realized integers define the realized `visible_load`). Every
  asymptotic statement in this note holds in this limit only. All repository
  measurements are at finite $(D, N, M)$; nothing here is a description of any
  recorded run.
+ *Optimization time $-> oo$.* Taken first, at fixed $lambda > 0$: by Statement
  M2 the minimizer is unique and algorithm-independent (SGD noise and
  `mc_samples = 1` mask noise are neglected #HEUR). Formally this is
  $beta -> oo$ in @eq:gibbs. The limit $lambda -> 0^+$ is taken *after*
  $beta -> oo$; at $lambda = 0$ with separable data the minimizer does not exist
  (norm divergence, Statement M2 caveat).
+ *Number of generated samples $-> oo$.* Taken after training, at fixed
  $(D, N, M, theta)$: by the law of large numbers over sampler randomness,
  generated-sample averages converge to expectations under the sampler-indexed
  terminal law $P_(theta, A, k)$. This limit is independent of, and does not
  commute with, the proportional limit.
+ *Replica continuation.* $n -> 0$ is the bookkeeping limit for
  $EE_(F,Z) log Z_i$; it is unrigorous as usual and labeled #RS wherever used.
+ *Quenched versus annealed — two distinct levels.* (i) At the level of the
  learning problem: the typical behavior is the quenched average
  $EE_(F,Z) log Z_i$ computed here; the annealed $log EE_(F,Z) Z_i$ is a
  different object (it weights atypical datasets) and is not computed.
  Self-averaging — concentration of $(1\/N) log Z_i$ and of the norm observables
  over $(F, Z)$ in the proportional limit — is expected for this convex problem
  (proven for related regularized convex GLMs @thrampoulidis2018precise) but is
  #CONJ here; it predicts repeat-to-repeat fluctuations $O(D^(-1\/2))$,
  experiment E-L1. (ii) At the level of the data law: $P_F$ (quenched),
  $EE_F P_F = 2^(-N)$ (annealed, exact by Statement T5), and $hat(P)_M$
  (empirical) are three different objects; averaging over explicit repeats is the
  only correct way to approach the second.
+ *Teacher entropy input.* Where the companion note's RS/ansatz values
  $s(gamma)$, $s_oo (gamma)$ are invoked (Statements T6–T7 remark), they inherit
  the RS status of `notes_hiddenmanifold.typ` and are never presented as
  established.

= Phase-4C experimental dictionary <sec:dictionary>

Each prediction above is mapped below to an executable experiment using the
active package (`src/maskeddiffusion/`, CLIs `maskeddiffusion-train`/`-sample`/
`-evaluate`, TOML configs with contract names `latent_dim`, `aspect_ratio`,
`sample_ratio`, `l2reg`, `v_policy`, `bias_policy`, `sampler_name`,
`tokens_per_step`, `n_generate`, `base_seed`). Execution notes: several
`base_seed` repeats per cell for quenched averaging; artifacts recorded with
manifests per `docs/adr/003-artifact-format.md`; protected notebooks and pinned
reference CSVs are never rerun or regenerated (E-L2 compares against pinned
values only). Status letters refer to the evidence labels of Section 1.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 4pt,
    align: left,
    [*ID*], [*Prediction (source, status)*], [*Executable experiment*], [*Metric / expected signature*],
    [E-F1], [arcsine law at fixed $F$ (T3, #THM)], [teacher + samples; compare empirical correlations against $(2\/pi) arcsin(hat(rho)_(i j))$ with $hat(rho)$ from `teacher.correlation_matrix()`], [`correlation_error` (metrics/correlations.py); residual $O(D^(-1\/2))$],
    [E-F2], [annealed uniformity (T5, #THM)], [average one/two-point marginals over explicit teacher repeats], [`empirical_mean_spin`, `empirical_pair_correlation` $-> 0$ at rate $1\/sqrt("repeats")$],
    [E-F3], [macroscopic correlation (T6, #DER)], [compute $norm(C(F))_F^2 \/ N$ per repeat over a $gamma$-grid], [ratio to $1 + (2\/pi)^2 gamma -> 1$ as $D$ grows],
    [E-F4], [Cover support vs coverage (T7, #DER)], [count distinct training patterns; compare $log(\# "distinct")\/N$ with $h_2 (1\/gamma)$ envelope], [coverage $M\/T(N,D) -> 0$; distinct-count curve],
    [E-V1], [$v^* = 0$ population theorem + $q V approx c gamma\/alpha$ (V1–V5, #THM/#HEUR)], [`v_policy="trainable"` vs `"frozen_zero"` over `sample_ratio` grid at fixed `aspect_ratio`, several `base_seed` repeats], [logged `qV`; sign-symmetry of $V$ entries across repeats ($EE[V] = 0$); test-BCE parity between policies],
    [E-V2], [bias vanishing (V1, #THM)], [`bias_policy="trainable"`], [$"mean"(|b|)$, $norm(b)^2\/N -> 0$; sign symmetry],
    [E-V3], [symmetry-breaking control (V4, #THM)], [same runs on flip-symmetrized vs spin-biased synthetic data], [`qV` contrast: $approx 0$ symmetrized, $> 0$ biased],
    [E-S1], [RS saddle train/test loss (S5, O1–O2, #RS)], [solve @eq:sp-entropic1–@eq:sp-energetic2 numerically (extend `julia-code/hiddenmanifold`); train over $(gamma, alpha, lambda)$ grid], [train/validation masked BCE vs saddle curves; gap $Delta$ of @eq:ltest],
    [E-S2], [teacher alignment $psi > q$ (R4, S6, #RS)], [from checkpoints + teacher artifact: $hat(psi) = "mean"_i norm(F^top W_i)^2 \/ N$ vs `qW`], [$hat(psi)\/hat(q)$ vs $gamma$; $-> 1$ as $gamma -> 0$],
    [E-S3], [deformed $W$-spectrum (S1, S6, #RS)], [eigenvalues of $W^top W \/ N$ from checkpoints vs MP-resolvent prediction from the saddle], [spectral distance; bulk deformation with $gamma$],
    [E-O1], [time-sliced accuracy (O3, #RS)], [one-shot unmasking accuracy vs $t$ at fixed $(gamma, alpha, lambda)$], [$cal(E)_t^"fair"$, $cal(E)_t^"greedy"$ curves vs @eq:acc-fair–@eq:acc-greedy],
    [E-O2], [U-turn trajectories (O4, #MF)], [`maskeddiffusion-sample` U-turn protocol from masked training data, initial fraction $t_0$], [$q_U^(t_0)$ vs ODE @eq:uturn-ode; $m_f$ vs $m_i$],
    [E-O3], [true-true overlap law (O5, #DER)], [fresh-$z$ $P_F$ pairs through the same $F$; overlap histogram], [Gaussian fit; $N dot Var(q) -> 1 + (2\/pi)^2 gamma$],
    [E-O4], [MMD diagnostic power (O6, #DER)], [`maskeddiffusion-evaluate` over $N$ at fixed $lambda in {4, 8}$; record biased/unbiased `model_vs_true`, `true_vs_true`, `train_vs_true`, `model_vs_train`; add overlap-moment reporters], [$O(1\/N)$ collapse of all MMD$^2$; $Delta sigma^2$; model-model collision mass],
    [E-O5], [memorization observables (O7, #HEUR/#CONJ)], [`nearest_training_overlap`, `nearest_training_excess`, top-3 overlaps, Hamming memorized fraction vs `sample_ratio`], [memorization curves; from-scratch collapse rate via model-model overlaps],
    [E-O6], [sampler dependence (§7.6, structural)], [one checkpoint, all `sampler_name` values incl. `tokens_per_step > 1`], [per-sampler MMD and overlap laws; terminal laws named $P_(theta, A, k)$],
    [E-L1], [self-averaging (§8(5), #CONJ)], [repeats over (teacher, dataset, training) seeds at several $D$], [variance of test BCE/MMD vs $D$: $O(1\/D)$],
    [E-L2], [$gamma -> 0$ crossover (S7, #DER)], [small-`aspect_ratio` runs; compare with pinned legacy uniform-data reference values (`docs/REFERENCE_RESULTS_MANIFEST.md`; no reruns)], [continuity of $q$, train/test loss toward uniform-data values],
  ),
  caption: [Phase-4C dictionary: every theoretical prediction of this note mapped
    to an executable experiment and metric. Status labels as in Section 1.],
) <tab:p4c>

#bibliography("bibliography.bib")

#show: arkheion-appendices

= Price's theorem and the arcsine law <app:price>

*Statement.* For $(h_1, h_2)$ centered jointly Gaussian with unit variances and
correlation $rho$,
$ EE ["sign"(h_1) thin "sign"(h_2)] = 2/pi thin arcsin(rho). $

_Proof._ Price's theorem @price1958useful: for a bivariate normal pair,
$partial_rho thin EE [g(h_1) g(h_2)] = EE [g''(h_1) g''(h_2)]$-type recursion;
specializing to $g = "sign"$ gives $partial_rho thin EE ["sign"(h_1)
"sign"(h_2)] = 2 thin phi_2 (0, 0; rho)$, with $phi_2$ the bivariate normal
density, $phi_2 (0, 0; rho) = (2 pi sqrt(1 - rho^2))^(-1)$. Hence
$partial_rho thin EE = 1\/(pi sqrt(1 - rho^2))$, and integrating from $0$ (where
the signs are independent and the expectation vanishes) to $rho$ gives
$EE = (1\/pi) arcsin(rho) dot 2 = (2\/pi) arcsin(rho)$. $square$

Applied to $h = F z$ conditional on $F$: $(h_i, h_j)$ is jointly Gaussian with
variances $G_(i i), G_(j j)$ and covariance $G_(i j)$; normalizing gives
$rho_(i j) = G_(i j)\/sqrt(G_(i i) G_(j j))$ and Statement T3 follows. The same
theorem with four signs and Isserlis' formula gives the leading term of
Statement T4.

= Marchenko--Pastur transforms <app:mp>

Let $rho_gamma$ be the Marchenko--Pastur law of Statement T2 and
$ m_gamma (z) := integral rho_gamma (s)/(s - z) thin dif s, quad
  z in CC \\ [(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2], $
its Stieltjes transform, the unique solution with $m_gamma (z) ~ -1\/z$ at
infinity of
$ gamma thin z thin m_gamma (z)^2 + (z + gamma - 1) thin m_gamma (z) + 1 = 0. $ <eq:mp-quadratic}

The transforms of Statement S1 and the resolvents of Statement S5 are all
rational functionals of $m_gamma$ evaluated at the real point $zeta = -a\/b$
($b != 0$; the $b = 0$ cases follow by continuity):
$ cal(T)_gamma (a, b; thin c, d) = d/b + (c - d a/b) thin (1/b) thin m_gamma (-a/b), $ <eq:T-formula}
$ cal(L)_gamma (a, b) = log a + integral_0^(b\/a) lr[ 1/u - 1/u^2 thin m_gamma (-1/u) ] dif u,
  quad (a > 0), $ <eq:L-formula}
obtained from $s\/(1 + u s) = 1\/u - (1\/u^2)\/(s + 1\/u)$ and
$integral rho (s + 1\/u)^(-1) dif s = m_gamma (-1\/u)$. The resolvents of
@eq:sp-entropic1–@eq:sp-entropic2 are
$ integral rho_gamma (s)/(a - b s) dif s = -1/b thin m_gamma (-a/b), quad
  integral rho_gamma (s) thin s/(a - b s) dif s = -1/b - a/b^2 thin m_gamma (-a/b), $
and the squared-resolvent moments follow by differentiating these in $(a, b)$:
$integral rho (a - b s)^(-2) dif s = -partial_a [(1\/b) m_gamma (-a\/b)]$-type
identities. Numerically, @eq:mp-quadratic is solved at the needed real points and
the remaining one-dimensional integral @eq:L-formula is done by quadrature; this
closes the entropic half of the saddle. (The atom of $rho_gamma$ at $s = 0$ for
$gamma > 1$ contributes its mass explicitly; the formulas above hold with
$m_gamma$ the full transform including the atom.)

= Wishart moment computations <app:wishart>

For $G = (1\/D) X X^top$, $X_(i nu) ~ cal(N)(0,1)$, $X in RR^(N times D)$:
+ $EE Tr G = (1\/D) sum_(i, nu) EE X_(i nu)^2 = N$.
+ $EE Tr G^2 = (1\/D^2) sum_(i j) sum_(nu nu') EE [X_(i nu) X_(j nu) X_(j nu')
  X_(i nu')]$. Wick contraction gives three pairings; the nonzero contributions
  are $N D^2 + N^2 D + N D$, so $EE Tr G^2 = N(1 + gamma + 1\/D)$, and the
  proportional limit gives $1 + gamma$.
+ The third moment $1 + 3 gamma + gamma^2$ follows likewise from the six Wick
  pairings of a length-6 trace (Narayana numbers).
+ Frobenius/diagonal split used in Statement T6:
  $EE norm(G)_F^2 = EE Tr G^2 = N(1 + gamma) + O(1)$, and
  $EE sum_i G_(i i)^2 = N thin EE[G_(1 1)^2] = N(1 + 2\/D)$, hence
  $EE sum_(i != j) G_(i j)^2 = N gamma + O(1)$.
+ Overlap variance (Statement O5): with $x, y$ i.i.d. $P_F$,
  $Var(q mid(|) F) = (1\/N^2) sum_(i j) EE[x_i x_j mid(|) F] thin
  EE[y_i y_j mid(|) F] = (1\/N^2) norm(C(F))_F^2$, and Statement T6 gives the
  density $1 + kappa_1^4 gamma$.
