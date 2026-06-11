#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/drafting:0.2.2": inline-note, margin-note
#import "@preview/mannot:0.3.0": markrect

#show figure.caption: set align(left)
#show figure.caption: set text(style: "italic")

#show: arkheion.with(
  title: [Entropy of the Hidden-Manifold Output Distribution],
  authors: (
    (name: "Carlo Lucibello", email: "carlo.lucibello@unibocconi.it", affiliation: [Bocconi University, Milan]),
  ),
  date: datetime.today().display("[day] [month repr:Long] [year]"),
  abstract: [#align(left)[We compute the quenched entropy of the output distribution of a hidden-manifold model with a generic channel $P_"out"$, using the replica method. The full derivation is presented down to the replica-symmetric saddle-point equations.]],
)

#let note(content) = highlight([NOTE: ] + content, fill: gray.lighten(80%))

#let citecolor = rgb("#93430e")
#show cite: set text(fill: citecolor)
#show link: set text(fill: blue)
#show link: underline
#show ref: set text(fill: blue)

#let extr = math.op("extr", limits: true)
#let Tr = math.op("Tr")
#let Pout = $P_"out"$

#outline(depth: 2, indent: n => n * 1em)

= Introduction
== TODO (Agents should not edit this section)
- Outer-1RSB ansatz set up in @app:1rsb (breaks the disorder index $alpha$, keeps the $(n+1)$-block RS); explicit replicon/stability computation around the $q_0 = 0$ RS saddle for the sign channel still pending.
- Add Bayesian interpretation of overlaps.
- Add countinuous activations case, e.g. erf(h) + noise
- Compute Fisher information?

== The hidden manifold model

The *hidden manifold model* (HMM) @goldt2020hidden is a generative framework for high-dimensional structured data that plays a central role in the statistical-physics theory of supervised learning.
Observations $x in RR^N$ are produced by projecting a low-dimensional Gaussian latent code $z in RR^D$ through a random linear map $F in RR^(N times D)$, followed by an entry-wise output channel $Pout$:
$ z ~ cal(N)(0, I_D), quad x_i ~ Pout (dot mid(|) (F z)_i), quad F_(i mu) ~ cal(N)(0, 1/D) quad "i.i.d." $
The aspect ratio $gamma = N/D$ interpolates between the under-determined ($gamma < 1$) and over-determined ($gamma > 1$) regimes and controls the richness of the output distribution $p_F (x)$.

The HMM captures the empirical observation that real data concentrates near a lower-dimensional manifold, while retaining Gaussian-equivalence properties that make the replica method analytically tractable @goldt2020hidden.
It has become a standard framework for deriving exact asymptotics of generalization in the proportional limit $N, D -> oo$ at fixed $gamma$.
@geraceGeneralisationErrorLearning2021 used the replica method to compute the exact generalization error for ridge regression and random features under the HMM, establishing a rigorous Gaussian equivalence theorem that reduces the non-linear model to an effective Gaussian problem.
@loureiro2021learning extended this analysis to generic feature maps and fully characterized the learning-curve phenomenology, including the double-descent peak and the under-to-over-parametrized crossover.

== Setup and goals

These notes address a complementary question: rather than studying how well a learner can recover a target function, we study the *information content of the output distribution $p_F (x)$ itself*.
Let
$ p_F (x) = integral D z thin product_(i=1)^N Pout (x_i | (F z)_i), quad D z = (2 pi)^(-D/2) e^(-|z|^2 / 2) dif z, $
with $F_(i mu) ~ cal(N)(0, 1/D)$, quenched, and $N, D -> oo$ at fixed
$ gamma = N/D. $

We compute the *quenched entropy*
$ S = EE_F [H_F (X)], quad H_F (X) = -integral dif x thin p_F (x) log p_F (x), $
and its Rényi generalization $S_alpha = EE_F [H_alpha (X)]$ for $alpha > 0$, using the replica method.

== Outline

The derivation proceeds as follows.

*Section 2* reduces the Shannon entropy to a replicated partition function, introduces the overlap order parameters via Hubbard--Stratonovich, and derives the replica-symmetric (RS) saddle-point equations.
A key structural feature is the Bayes-optimal Nishimori collapse $q_0^* = 0$, $q_1^* = q^*$, with conjugates $hat(q)_0^* = hat(q)_d^* = 0$ and $hat(q)_1^* = hat(q)^*$, which reduces the five RS parameters to a single overlap $q^*$ satisfying a Fisher-information self-consistency equation.
A dedicated subsection clarifies what is exact and what is an ansatz in the *outer* (disorder-replica, $m$) space: at $n = 0$ the identity $Z_0 (F) = 1$ makes outer RS automatic at the order needed for the Shannon entropy, whereas the inner-block RS assumption is a Bayes-optimal ansatz justified by Nishimori exchangeability and the standard local-stability hypothesis. At finite Rényi order the situation reverses — RS is genuinely an ansatz, parity symmetry forces $q_0 = 0$ on the RS branch but does not exclude outer 1RSB, and a replicon check is needed.

*Section 3* extends the machinery to the quenched Rényi entropy $S_alpha$ at finite order $alpha = n + 1$.
First steps toward an 1RSB computation are done in @app:1rsb.
*Section 4* works out the noiseless sign channel in detail, recovering the Cover dichotomy @eq:s-hartley at $alpha -> 0$, providing closed or semi-closed expressions for the Shannon, collision, and min-entropy, and reducing the saddle-point system to a single scalar equation under the empirically observed $q_0 = 0$ ansatz; @fig:renyi-profile gives the full numerical Rényi profile $s_alpha(gamma)$ across $alpha in [0.02, 10]$.

Heavier algebraic details are collected in the appendices.

= Replica computation of the entropy

== Replica representation of the entropy

=== Replicas of the entropy at fixed $F$

Since $integral dif x thin p_F (x) = 1$, the identity
$ -integral dif x thin p_F (x) log p_F (x) = -lr(partial_n integral dif x thin p_F (x)^(n+1) mid(|))_(n=0) $
holds. Define, for integer $n >= 0$,
$ Z_n (F) := integral dif x thin p_F (x)^(n+1). $ <eq:Zn>
Because $Z_0 (F) = 1$, we have $log Z_0 (F) = 0$, and so
$ H_F (X) = -lr(partial_n log Z_n (F) mid(|))_(n=0). $

Expanding the $(n+1)$-th power introduces $n+1$ latent replicas $z^a in RR^D$, $a = 0, dots, n$:
$ Z_n (F) = integral dif x thin product_(a=0)^n D z^a thin product_(i=1)^N product_(a=0)^n Pout (x_i | (F z^a)_i). $
The same observation $x_i$ is shared by all replicas: this coupling is the structural source of the overlap order parameters.

=== Quenched average: second replica index

We need $S = -EE_F partial_n log Z_n (F)|_(n=0)$. Introduce a disorder-replica index $alpha = 1, dots, m$:
$ EE_F log Z_n (F) = lim_(m -> 0) 1/m thin log EE_F [Z_n (F)^m], $
so that
$ S = -lr(partial_n thin lim_(m -> 0) 1/m thin log EE_F [Z_n (F)^m] mid(|))_(n=0). $ <eq:S-replica>

For integer $m$,
$ Z_n (F)^m = integral product_(alpha=1)^m dif x^alpha thin product_(alpha,a) D z^(alpha a) thin product_(i=1)^N product_(alpha,a) Pout (x_i^alpha | (F z^(alpha a))_i). $
We collect the replica indices into $r = (alpha, a)$, $r = 1, dots, M$, $M = m(n+1)$.

== Gaussian disorder average and order parameter

=== Gaussian average over $F$

The disorder $F$ enters only through the linear forms $h_i^r := (F z^r)_i$. For fixed $z$'s, $(h_i^r)$ is jointly Gaussian with covariance
$ EE_F [h_i^r h_j^s] = delta_(i j) thin Q_(r s), quad Q_(r s) := 1/D thin z^r dot z^s, $
and is i.i.d. across the $N$ sites. Thus
$ EE_F [Z_n (F)^m] = integral product_r D z^r thin [Z_"out" (Q)]^N, $ <eq:EFZmid>
$ Z_"out" (Q) := integral product_alpha dif x^alpha integral dif mu_Q (h) thin product_(alpha,a) Pout (x^alpha | h^(alpha a)), $
where $dif mu_Q (h)$ is the centered $M$-dim. Gaussian measure with covariance $Q$.

=== Hubbard–Stratonovich and the prior action

The integrand of @eq:EFZmid depends on $z$ only via the symmetric matrix $Q_(r s) = z^r dot z^s\/D$. Inserting a delta-function and Fourier representation introduces the conjugate $hat(Q)$, after which the $z$-integral factorizes over the $D$ components of $z^r$ and is Gaussian:
$ integral product_r D z^r thin exp(1/2 sum_(r s) hat(Q)_(r s) thin z^r dot z^s) = [det (I - hat(Q))]^(-D/2). $
Collecting all $D$-extensive contributions,
$ EE_F [Z_n (F)^m] = integral dif Q dif hat(Q) thin exp{D thin cal(S)_(m,n) (Q, hat(Q))}, $
$ cal(S)_(m,n) (Q, hat(Q)) = cal(S)_"prior" (Q, hat(Q)) + gamma thin cal(S)_"out" (Q), $ <eq:Smn>
$ cal(S)_"prior" = -1/2 Tr(Q hat(Q)) - 1/2 log det(I - hat(Q)), quad cal(S)_"out" = log Z_"out" (Q). $

In the limit $D -> oo$, saddle-point evaluation gives $1/D thin log EE_F [Z_n (F)^m] -> extr_(Q, hat(Q)) cal(S)_(m,n)$, and the entropy density is
$ s := lim_(N -> oo) S/N = -1/gamma thin lr(partial_n thin lim_(m -> 0) 1/m thin extr_(Q, hat(Q)) cal(S)_(m,n) mid(|))_(n=0). $ <eq:s-density>

== The replica-symmetric calculation

=== The RS ansatz

We impose invariance under permutations of the disorder replicas $alpha$ and, separately, under permutations of the entropy replicas $a$ within each block, giving
$ Q_((alpha,a),(beta,b)) = cases(
  1 quad & alpha = beta\, a = b,
  q_1 & alpha = beta\, a != b,
  q_0 & alpha != beta,
) quad
$
$
hat(Q)_((alpha,a),(beta,b)) = cases(
  hat(q)_d quad & alpha = beta\, a = b,
  hat(q)_1 & alpha = beta\, a != b,
  hat(q)_0 & alpha != beta.
) $
The diagonal of $Q$ is $1$ because the Gaussian prior fixes $|z^r|^2\/D -> 1$.

The matrix $hat(Q)$ has three eigenvalues coming from invariant subspaces (within-block antisymmetric, between-block antisymmetric, fully symmetric); see @app:spectrum for the diagonalization. The result is
$ log det (I - hat(Q)) = m n log A + (m - 1) log B + log C, $ <eq:logdet>
with
$ A &= 1 - hat(q)_d + hat(q)_1, \
B &= 1 - hat(q)_d - n hat(q)_1 + (n+1) hat(q)_0, \
C &= 1 - hat(q)_d - n hat(q)_1 - (m-1)(n+1) hat(q)_0. $
The trace term enumerates immediately to
$ Tr(Q hat(Q)) = m(n+1) hat(q)_d + m thin n(n+1) thin q_1 hat(q)_1 + m(m-1)(n+1)^2 thin q_0 hat(q)_0. $ <eq:trace>

=== Output action under the RS ansatz

The RS structure of $Q$ admits the hierarchical Gaussian decomposition
$ h^(alpha a) = sqrt(q_0) thin u + sqrt(q_1 - q_0) thin v^alpha + sqrt(1 - q_1) thin w^(alpha a), $ <eq:hierarchical>
with $u, {v^alpha}, {w^(alpha a)}$ mutually independent standard Gaussians; the covariance reproduces $1, q_1, q_0$ on the three RS strata. Define
$ tilde(cal(P))(x mid(|) z) := integral D w thin Pout (x mid(|) z + sqrt(1 - q_1) w), $ <eq:Ptilde>
$ cal(P)(x mid(|) u, v) := tilde(cal(P))(x mid(|) sqrt(q_0) u + sqrt(q_1-q_0) v). $ <eq:Pdef>
Both are normalized in $x$. Because the $w^(alpha a)$ are independent across $a$, the $a$-product factorizes into $[cal(P)(x^alpha | u, v^alpha)]^(n+1)$, and the disorder replicas decouple as well. We obtain
$ Z_"out" (q_0, q_1) = integral D u thin [cal(K)_n (u; q_0, q_1)]^m, quad
cal(K)_n := integral D v integral dif x thin [cal(P)(x | u, v)]^(n+1). $ <eq:Kn>

=== Limit $m -> 0$

Since $cal(K)_0 = 1$, expanding $exp(m log cal(K)_n)$ gives
$ 1/m thin cal(S)_"out"^"RS" thin |_(m -> 0) = integral D u thin log cal(K)_n (u; q_0, q_1) =: Phi_n (q_0, q_1). $ <eq:Phi>

For the prior, expanding @eq:logdet for small $m$ (using $C - B = -m(n+1)hat(q)_0$, see @app:m-zero) gives
$ 1/m log det(I - hat(Q)) thin |_(m -> 0) = n log A + log B_0 - ((n+1) hat(q)_0) / B_0, $
$ B_0 := 1 - hat(q)_d - n hat(q)_1 + (n+1) hat(q)_0, $
and the trace term divides cleanly by $m$. Combining,
$ 1/m thin cal(S)_"prior"^"RS" thin |_(m -> 0) =& -1/2 [(n+1) hat(q)_d + n(n+1) q_1 hat(q)_1 - (n+1)^2 q_0 hat(q)_0] \
&- 1/2 [n log A + log B_0 - ((n+1) hat(q)_0) / B_0]. $ <eq:Sprior-m0>

Set
$ phi_n := lim_(m -> 0) 1/m thin cal(S)_(m,n)^"RS" = (cal(S)_"prior"^"RS" \/ m)|_(m -> 0) + gamma thin Phi_n (q_0, q_1), $
so that
$ s = -1/gamma thin lr(partial_n thin extr thin phi_n mid(|))_(n=0). $ <eq:s-extr>

== The Shannon limit $n -> 0$

=== Output term: conditional output entropy

In @eq:Pdef, $cal(P)$ depends on $(u, v)$ only through $z := sqrt(q_0) u + sqrt(q_1-q_0) v$. With $u, v$ standard normal, $z ~ cal(N)(0, q_1)$. Hence
$ cal(K)_n (u) = integral D v integral dif x thin [tilde(cal(P))(x mid(|) z(u, v))]^(n+1). $
At $n = 0$, the $x$-integral gives $1$, so $cal(K)_0 = 1$ and $Phi_0 = 0$. Differentiating once,
$ partial_n integral dif x thin [tilde(cal(P))(x|z)]^(n+1) thin |_(n=0) = -H(tilde(cal(P))(dot|z)), $
so 
$
partial_n cal(K)_n (u)|_(n=0) = -EE_v [H(tilde(cal(P))(dot|z(u,v)))]
$ 
and
#box(stroke: 0.5pt, inset: 6pt)[
  $ partial_n Phi_n |_(n=0) = -EE_(z ~ cal(N)(0, q_1)) [H(tilde(cal(P))(dot mid(|) z))]. $ <eq:dnPhi>
] 
This is the conditional output entropy at common-cause variance $q_1$.

=== Prior term

Differentiating @eq:Sprior-m0 in $n$ at fixed parameters and evaluating at $n = 0$ (envelope theorem at the saddle), one obtains, with $B := 1 - hat(q)_d + hat(q)_0$,
$ partial_n thin (1/m cal(S)_"prior"^"RS") |_(n=0) = -1/2 [hat(q)_d + q_1 hat(q)_1 - 2 q_0 hat(q)_0] - 1/2 [log A - hat(q)_1 / B + (hat(q)_0 (hat(q)_0 - hat(q)_1)) / B^2]. $ <eq:dnSprior>
The detailed expansion is given in @app:n-zero-prior.

== Saddle-point equations and final formula

=== RS saddle equations and the Bayes-optimal collapse

The bare $n^0$ stationarity equations of $phi_n$ are degenerate (since $log Z_0 = 0$ identically forces $phi_0 = 0$ at the saddle). The physical equations come from the order-$n$ piece. Carrying out the standard expansion (@app:RS-saddle), one finds for the Bayes-optimal entropy of the hidden-manifold model the Nishimori-type collapse
$ q_0^* = 0, quad q_1^* = q^*, $
leaving a single scalar overlap $q^* in [0, 1]$.

The collapse has a transparent meaning once one identifies the role each replica plays in the Bayes-optimal setup. The marginal $p_F (x)$ is the law of $x$ in the *teacher-student* generative model
$ z^* ~ cal(N)(0, I_D), quad x_i ~ Pout (dot mid(|) (F z^*)_i), $
with the same Gaussian prior on $z^*$ as in @eq:Zn — sampling $x$ from $p_F$ is equivalent to first drawing a teacher $z^*$ from the prior and then $x$ from the channel. Splitting one factor of $p_F (x)$ off in @eq:Zn,
$ Z_n (F) = EE_(z^* ~ "prior",thin x | z^*, F) [product_(a=1)^n integral D z^a thin product_i Pout (x_i mid(|) (F z^a)_i)]. $
The replica index $a = 0$ plays the role of the *teacher* — it generates $x$ via the leftover $p_F (x)$ factor. The remaining replicas $a = 1, dots, n$ are *students* drawn from the posterior $p(z mid(|) x, F) prop p(z) thin Pout (x|z, F)$ conditioned on the *same* observation.

*Nishimori symmetry within a block.* The joint density of $(z^0 := z^*, z^1, dots, z^n, x)$ given $F$ factors as
$ p(z^0, dots, z^n, x mid(|) F) = (product_(a=0)^n p(z^a) thin Pout (x mid(|) F z^a)) \/ p_F (x)^n, $
which is *manifestly symmetric* under arbitrary permutations of $(z^0, dots, z^n)$. This is the Nishimori identity in the matched teacher-student model: a posterior sample is statistically indistinguishable from the teacher itself, so the $n+1$ replicas are exchangeable. Their pairwise overlaps therefore coincide,
$ EE [z^a dot z^b \/ D] = q^* "for all" a != b, $
and force $q_1 = q^*$ to equal the teacher-student magnetization. Within a block there is no separate "teacher-student" vs. "student-student" overlap; only one number survives.

*Independence across blocks: $q_0 = 0$.* It is tempting to argue simply that "teachers from different blocks are i.i.d. draws from a centered prior, so $EE[z^(*, alpha) dot z^(*, beta)\/D] = 0$". That conclusion is correct but the premise is more subtle: the teachers' marginal under the replicated model is generically a *tilted* prior, and only collapses to the bare centered Gaussian at $n = 0$. Three independent ingredients are needed.

*(i) Cross-block conditional independence given $F$.* The replicated weight factorises blockwise,
$ rho({z^((alpha, a))}, {x^alpha} mid(|) F) = product_alpha rho_alpha (z^((alpha, 0)), dots, z^((alpha, n)), x^alpha mid(|) F), quad rho_alpha prop product_(a=0)^n p(z^((alpha, a))) thin Pout (x^alpha mid(|) F z^((alpha, a))), $
so cross-block expectations decouple into products of within-block marginals,
$ EE [z^((alpha, a)) dot z^((beta, b)) \/ D mid(|) F] = EE[z^((alpha, a)) mid(|) F]^top thin EE[z^((beta, b)) mid(|) F] \/ D, quad alpha != beta. $

*(ii) Within-block exchangeability.* By the Nishimori symmetry just established, every replica in a block has the same marginal under $rho_alpha$ as the teacher: $EE[z^((alpha, a)) mid(|) F] = EE[z^(*, alpha) mid(|) F]$ for every $a in {0, dots, n}$.

*(iii) Collapse of the teacher marginal to the bare prior at $n = 0$.* Integrating $rho_alpha$ over $x^alpha$ and the other $n$ replicas,
$ rho_alpha (z^(*, alpha) mid(|) F) prop p(z^(*, alpha)) integral dif x^alpha thin Pout (x^alpha mid(|) F z^(*, alpha)) thin p_F (x^alpha)^n. $ <eq:tilted-marginal>
At $n = 0$ the tilting factor $p_F (x)^n$ disappears, the $x$-integral is identically $1$, and the marginal collapses to the bare centered prior $cal(N)(0, I_D)$, *independently of $F$*. Hence $EE[z^(*, alpha) mid(|) F] = 0$ pointwise in $F$, and combined with (i)–(ii),
$ EE [z^((alpha, a)) dot z^((beta, b)) \/ D] = EE_F [0 dot 0 \/ D] = 0, quad alpha != beta, $
which is precisely $q_0 = 0$.

The crucial role of $n = 0$ enters in step (iii): for $n != 0$ the marginal @eq:tilted-marginal is a genuine $F$-dependent tilt of the prior, the within-block magnetization $EE[z^(*, alpha) mid(|) F]$ does not collapse pointwise, and the cross-block average $q_0 = EE_F norm(EE[z^(*,alpha) mid(|) F])^2\/D$ becomes the $F$-variance of the magnetization — generically nonzero.

*Symmetric channels: $q_0 = 0$ at any $n$.* The vanishing of $q_0$ extends to all $n$ when the channel carries a $z arrow.r -z$ symmetry: there exists an involution $sigma$ on the output alphabet such that
$ Pout (sigma x mid(|) -y) = Pout (x mid(|) y) quad "for all" y in RR. $ <eq:channel-symmetry>
Examples: the sign channel of Section 4 satisfies this with $sigma x = -x$, as do additive symmetric-noise channels (AWGN, symmetric Bernoulli flips, etc.). Combined with the centred prior $p(-z) = p(z)$, the substitution $z arrow.r -z$ in the prior integral $p_F (x) = integral p(z) thin Pout (x mid(|) F z) thin dif z$ yields $p_F (sigma x) = p_F (x)$, and the same change of variables on @eq:tilted-marginal,
$ rho_alpha (-z^* mid(|) F) prop p(-z^*) thin integral dif x thin Pout (x mid(|) -F z^*) thin p_F (x)^n stretch(=)^(x arrow.r sigma x) p(z^*) thin integral dif x thin Pout (x mid(|) F z^*) thin p_F (x)^n prop rho_alpha (z^* mid(|) F), $
makes the within-block teacher marginal an *even* density in $z^*$, pointwise in $F$. Hence $EE[z^* mid(|) F] = 0$ identically, and steps (i)–(ii) give $q_0 = 0$ for every $n$. This is a symmetry, not a saddle accident — it explains why the noiseless sign channel of @sec:sign exhibits $q_0 = 0$ also at $n = 1, 2, dots$ (cf. @sec:sign-renyi-q0). It is, however, *not* a generic property: an asymmetric output channel breaks @eq:channel-symmetry and reinstates $q_0 != 0$ at finite $n$.

*When the collapse fails.* The teacher-student rewriting is exact only at $n = 0$. For Rényi entropies at $n != 0$, $Z_n$ probes the *tilted* measure $p_F (x)^(n+1) thin dif x$ rather than the marginal $p_F$: the extra factors $p_F (x)^n$ no longer admit an interpretation as posterior expectations against an underlying Bayes-optimal model, since $p_F (x)^n thin dif x$ is not a probability measure for $n != 0$ and there is no generative law in which $x$ acts as an observation drawn from a teacher. In the present RS parametrization the Nishimori-type collapse at $n=0$ reads
$ q_0 = 0, quad q_1 = q, quad hat(q)_d = 0, quad hat(q)_0 = 0, quad hat(q)_1 = hat(q), $
with $q, hat(q)$ the surviving Bayes-optimal scalars (cf. @app:RS-saddle). These are properties of the order-$n$ stationarity *at* $n = 0$, not of the system itself; in particular the standard Nishimori identity $hat(m) = hat(q)$ between teacher-student and student-student conjugate overlaps requires a parametrization that introduces those overlaps separately, whereas here teacher and student replicas have already been symmetrized inside the $(n+1)$-block. Consequently all five RS parameters $(q_0, q_1, hat(q)_d, hat(q)_0, hat(q)_1)$ evolve independently in the general Rényi setting of Section 3, and the saddle has $q_0 != 0$ generically.

The saddle-point system reduces to
#box(stroke: 0.5pt, inset: 6pt)[
$ cases(
hat(q) = gamma thin cal(I)(q),
q = hat(q) / (1 + hat(q)),
) $ <eq:RS-eqs>
]
where
$ cal(I)(q) := EE_(z~ cal(N)(0, q)) [integral dif x thin (partial_z tilde(cal(P))(x|z))^2 / (tilde(cal(P))(x|z))] $ <eq:fisher>
is the *Fisher information* of the channel $tilde(cal(P))(dot|z)$ with respect to its conditioning variable $z$. Eliminating $hat(q)$,
$ q = (gamma thin cal(I)(q)) / (1 + gamma thin cal(I)(q)). $

=== Final RS formula for the entropy

Putting together the output piece @eq:dnPhi and the simplified prior piece, the quenched entropy density is
#box(stroke: 0.5pt, inset: 8pt)[
$ s = EE_(z ~ cal(N)(0, q^*)) [H(tilde(cal(P))(dot mid(|) z))] + 1/(2 gamma) [-log(1 - q^*) - q^*], $ <eq:final-RS>
]
where $q^*$ solves @eq:RS-eqs and $tilde(cal(P))(x | z)$ is given by @eq:Ptilde with $q_1 = q^*$.
Equivalently, using $log(1 + hat(q)^*) = -log(1 - q^*)$ on the saddle, the prior bracket reads $log(1 + hat(q)^*) - q^*$. The expression is most cleanly obtained from the finite-$n$ Rényi formula @eq:Sprior-renyi-star: with the Bayes-optimal scaling $q_1, hat(q)_1 = O(1)$ and $q_0, hat(q)_0, hat(q)_d = O(n)$, expansion of @eq:Sprior-renyi-star at order $n$ gives
$ lr(cal(S)_"prior"^"RS" \/ m mid(|))_(m -> 0)^* = n thin [-1/2 thin q^* hat(q)^* + 1/2 thin hat(q)^* - 1/2 log(1 + hat(q)^*)] + O(n^2), $
and $s = -gamma^(-1) partial_n[dots] - partial_n Phi_n |_(n=0)$ together with $q^* = hat(q)^* / (1 + hat(q)^*)$ produces the bracket above. The detailed expansion is collected in @app:RS-saddle.

The first term is the *conditional entropy* of $x$ given the common Gaussian "signal" $z$ of variance $q^*$; the residual variance $1 - q^*$ is integrated against the channel via $w$. The second term is the prior contribution: it is non-negative on $q^* in [0, 1)$ (with $-log(1-q^*) - q^* >= 0$, vanishing only at $q^* = 0$) and diverges as $q^* -> 1$.

=== Consistency check: conditional entropy and mutual information

Decompose $H_F (X) = H_F (X | Z) + I_F (Z; X)$. The first term is *easy*: by independence across $i$ and $(F z)_i -> cal(N)(0, 1)$,
$ lim_(N -> oo) 1/N EE_F H_F (X | Z) = EE_(h ~ cal(N)(0,1)) [H(Pout (dot mid(|) h))]. $
This equals the $q^* -> 1$ limit of the first term in @eq:final-RS, where $tilde(cal(P))(x|z) -> Pout (x|z)$. The replica calculation thus evaluates the mutual-information piece $1/N EE_F I_F (Z; X)$, which is captured by the prior bracket of @eq:final-RS:
$ 1/N thin EE_F I_F (Z; X) -> 1/(2 gamma) thin [-log(1 - q^*) - q^*]. $
This term is non-negative for $q^* in [0, 1)$ and *diverges* as $q^* -> 1$, consistent with the fact that for an informative channel the latent $z$ becomes effectively decodable from $x$ and the mutual information per site, while remaining $O(1\/gamma)$, picks up the unbounded $-log(1-q^*)$ piece reflecting the perfect alignment of teacher and posterior. (The earlier statement that the mutual-information term vanishes at $q^* = 1$ was based on a spurious $(1 - q^*)$ factor in the prior saddle and is incorrect.)

=== Replica symmetry: what is exact and what is an ansatz <sec:outer-RS>

The replicated action @eq:Smn carries *two* distinct replica indices. The index $a = 0, dots, n$ comes from the power $p_F (x)^(n+1)$ inside $Z_n (F)$. The index $alpha = 1, dots, m$ is introduced only to compute the quenched average
$ EE_F log Z_n (F) = lim_(m -> 0) 1/m thin log EE_F Z_n (F)^m. $
Replica symmetry in these two directions has different meanings. The $a$-replicas form the internal "molecule" associated with one copy of $Z_n$, while the $alpha$-replicas are the usual quenched replicas of the partition function. A Parisi instability of the quenched free entropy should therefore first be looked for in the *outer* $alpha$-direction; internal breaking of the $(n+1)$-block is a secondary check.

*Outer RS is essentially automatic at $n = 0$.* The integer-replica object satisfies
$ Z_0 (F) = integral dif x thin p_F (x) = 1 quad "pointwise in" F. $
Hence $Z_n (F) = 1 + n A_F + O(n^2)$ with $A_F = -H_F (X)$, and
$ EE_F Z_n (F)^m = 1 + m n thin EE_F A_F + O(n^2, m^2). $
After $m -> 0$ and differentiation at $n = 0$, only the one-block term survives. Correlations among different disorder replicas can enter only at $O(n^2)$ — i.e. in higher cumulants of $A_F$ — and therefore drop out of the Shannon derivative. Outer RS at $n = 0$ is thus not a physical assumption: it is forced/trivial at the order needed for the Shannon entropy.

*$q_0 = 0$ from the one-block marginal.* The vanishing of the RS inter-block overlap admits a direct derivation. For a single $Z_n$-block, the one-replica marginal is
$ rho_n (z mid(|) F) = (p(z)) / (Z_n (F)) integral dif x thin Pout (x mid(|) F z) thin p_F (x)^n , $ <eq:rho-n>
so the RS cross-block overlap is
$ q_0 (n) = EE_F norm(bar(z)_n (F))^2 \/ D, quad bar(z)_n (F) := integral dif z thin z thin rho_n (z mid(|) F). $
At $n = 0$ the tilt $p_F (x)^n$ disappears, $rho_0 (z mid(|) F) = p(z)$, and the centred prior gives $bar(z)_0 (F) = 0$ pointwise in $F$, hence $q_0 = 0$. The conjugate variables in the present RS parametrization then satisfy
$ hat(q)_0 = 0, quad hat(q)_d = 0, quad hat(q)_1 = hat(q), $
rather than $hat(q)_0 = hat(q)_1$ (cf. @app:RS-saddle).

*Inner RS as a Bayes-optimal ansatz.* The remaining nontrivial assumption is the internal, Bayes-optimal RS ansatz inside one $(n+1)$-block. After singling out one factor of $p_F (x)$, the $a = 0$ replica plays the role of the teacher $z^*$ and the others are posterior samples from $p(z mid(|) x, F)$. In the matched model the Nishimori identity makes teacher and posterior samples exchangeable, which equates the teacher--student and student--student overlap moments and justifies a single overlap $q$ at the RS saddle. Exchangeability does *not*, however, by itself prove that the full overlap distribution is a delta — Parisi RSB ensembles are still exchangeable after averaging over the random hierarchy. The RS formula @eq:final-RS should therefore be understood as exact when the Nishimori saddle is stable against replicon/RSB perturbations. For the dense Gaussian HMM with simple log-concave or convex channels this is the expected situation; for more structured or hard channels the replicon (or a 1RSB perturbation) should be checked.

*Finite Rényi order $n != 0$.* The problem is no longer Bayes-optimal in the original sense: the measure is tilted by $p_F (x)^n$ and the Nishimori collapse is lost. The five-parameter RS saddle $(q_0, q_1, hat(q)_d, hat(q)_0, hat(q)_1)$ is then a genuine ansatz, not a consequence of symmetry. If the channel has the parity symmetry @eq:channel-symmetry, the one-block marginal @eq:rho-n is even in $z$, hence $bar(z)_n (F) = 0$ and the RS cross-block parameter satisfies
$ q_0 (n) = 0 quad "on the RS branch, for every" n. $
This justifies the $q_0 = 0$ reduction for the sign channel (@sec:sign-renyi-q0). It is, however, weaker than full outer RS: a parity-symmetric spin glass can still have a nontrivial overlap distribution. The correct claim is that *symmetry forces the RS $q_0$-branch to zero*, not that symmetry proves outer RS at all Rényi orders. Outer Parisi RSB at finite $n$ remains a genuine question that requires a stability or interpolation argument; the appropriate first 1RSB ansatz, breaking the *outer* index $alpha$, is set up in @app:1rsb.

*Near $alpha = 1$.* If the Shannon RS saddle is locally stable, RS persists for $alpha$ in a neighbourhood of $1$ by continuity. Far from $alpha = 1$ — especially as $alpha -> 0$ or $alpha -> oo$, where the Rényi entropy probes rare support/peak structure — outer RSB and large-deviation effects become more plausible and a stability check is needed.

The clean message, summarized:
$ #box(stroke: 0.5pt, inset: 6pt)[
  "Shannon: outer RS is exact/irrelevant; inner RS is Nishimori-stable if the Bayes saddle is stable."
] $
$ #box(stroke: 0.5pt, inset: 6pt)[
  "Finite Rényi: RS is an ansatz; parity can force " q_0 = 0 " but does not prove full RS."
] $

= Quenched Rényi entropy at finite $n$

== Setup

The same replica machinery yields the *quenched Rényi entropy* of order $alpha = n+1$ (with $alpha > 0$, $alpha != 1$),
$ S_alpha = EE_F [H_alpha (X)], quad H_alpha (X) = 1/(1-alpha) thin log integral dif x thin p_F (x)^alpha = -1/n thin log Z_n (F), $
with $Z_n (F)$ exactly the integer-replica object defined in @eq:Zn. The Shannon entropy of Section 2 is recovered as $alpha -> 1$ ($n -> 0$) since $-n^(-1) log Z_n -> -partial_n log Z_n |_(n=0)$.

The quenched average still requires the second replica trick $EE_F log Z_n = lim_(m->0) m^(-1) log EE_F [Z_n^m]$, so steps 2.2–2.3 carry over verbatim: the disorder-averaged action @eq:Smn, the Hubbard–Stratonovich step, and the $m -> 0$ expansion of @eq:Sprior-m0 are unchanged. The only difference is that we keep $n$ finite instead of differentiating at $n = 0$. Defining
$ phi_n (q_0, q_1, hat(q)_d, hat(q)_0, hat(q)_1) := lr(cal(S)_"prior"^"RS" \/ m mid(|))_(m -> 0) + gamma thin Phi_n (q_0, q_1) $
exactly as in @eq:Phi and @eq:Sprior-m0, the Rényi entropy density is
#box(stroke: 0.5pt, inset: 6pt)[
$ s_alpha = lim_(N -> oo) S_alpha / N = -1 / (n gamma) thin extr_(q_0, q_1, hat(q)_d, hat(q)_0, hat(q)_1) phi_n. $ <eq:s-renyi>
]

== RS saddle-point equations

Stationarity of $phi_n$ in the conjugate variables $(hat(q)_d, hat(q)_0, hat(q)_1)$ involves only the prior piece. Differentiating @eq:Sprior-m0 and combining the resulting three relations yields the matrix-inversion identities $Q = (I - hat(Q))^(-1)$ on each RS subspace (cf. @app:spectrum):
#box(stroke: 0.5pt, inset: 6pt)[
$ cases(
1 - q_1 = 1 / A,
1 + n q_1 - (n+1) q_0 = 1 / B_0,
q_0 = hat(q)_0 / B_0^2,
) $ <eq:Q-inv-renyi>
] 
with 
$
A := 1 - hat(q)_d + hat(q)_1, quad B_0 := 1 - hat(q)_d - n hat(q)_1 + (n+1) hat(q)_0. 
$

Stationarity in $(q_0, q_1)$ couples the prior to the output term and gives the conjugate parameters in terms of the channel (the prior pieces in @eq:Sprior-m0 contribute $-(n+1)^2 q_0 hat(q)_0 \/ 2$ and $-n(n+1) q_1 hat(q)_1 \/ 2$, which are linear in $q_0, q_1$):
#box(stroke: 0.5pt, inset: 6pt)[
$ cases(
hat(q)_1 = (2 gamma) / (n (n+1)) thin partial_(q_1) Phi_n (q_0, q_1),
hat(q)_0 = -(2 gamma) / (n+1)^2 thin partial_(q_0) Phi_n (q_0, q_1).
)  $ <eq:RS-renyi-conj>
]

Equations @eq:Q-inv-renyi and @eq:RS-renyi-conj form a closed five-dimensional system. In the Bayes-optimal Shannon limit $n -> 0$, Nishimori symmetry collapses $(q_0, hat(q)_0) -> (0, 0)$, the second relation in @eq:Q-inv-renyi reduces to $1 + n q_1 = 1/B_0$, and the system contracts to @eq:RS-eqs after the linear-in-$n$ expansion of @app:RS-saddle. At finite $n$ the Nishimori symmetry is broken and all five parameters are generically nonzero.

== Final RS formula

Let $(q_0^*, q_1^*, hat(q)_d^*, hat(q)_0^*, hat(q)_1^*)$ denote a solution of @eq:Q-inv-renyi and @eq:RS-renyi-conj. Substituting @eq:Q-inv-renyi into @eq:Sprior-m0 — using $A^* = 1\/(1-q_1^*)$, $B_0^* = 1\/(1 + n q_1^* - (n+1) q_0^*)$, and $hat(q)_d^* = -q_1^* \/ (1 - q_1^*) + hat(q)_1^*$ — the prior contribution at the saddle reads
$ lr(cal(S)_"prior"^"RS" \/ m mid(|))_(m -> 0)^* =& thin (n+1) / 2 thin (q_1^*) / (1 - q_1^*) + n / 2 thin log(1 - q_1^*) \
  & + 1 / 2 thin log(1 + n q_1^* - (n+1) q_0^*) + (n+1)/2 thin (q_0^*) / (1 + n q_1^* - (n+1) q_0^*) \
  & - (n+1)(1 + n q_1^*)/2 thin hat(q)_1^* + (n+1)^2 / 2 thin q_0^* hat(q)_0^*. $ <eq:Sprior-renyi-star>
Together with $gamma thin Phi_n^*$ from the output piece, the Rényi entropy density is
#box(stroke: 0.5pt, inset: 8pt)[
$ s_alpha = -1 / (n gamma) [lr(cal(S)_"prior"^"RS" \/ m mid(|))_(m -> 0)^* + gamma thin Phi_n^*], quad alpha = n + 1, $ <eq:final-renyi>
] 
with @eq:Sprior-renyi-star supplying the prior part. The output integral
$ Phi_n^* = integral D u thin log integral D v integral dif x thin [tilde(cal(P))(x mid(|) sqrt(q_0^*) u + sqrt(q_1^* - q_0^*) v)]^(n+1) $
is the natural generalization of the conditional output entropy of @eq:dnPhi: as $n -> 0$ it tends to zero linearly with slope equal to minus the conditional entropy at common-cause variance $q_1^*$, recovering the first term of @eq:final-RS.

For *integer* $n >= 1$ the output integral $Phi_n$ reduces to a finite-dimensional Gaussian integral with $(n+1)$ explicit replica fields $w^a$, and @eq:final-renyi gives the *integer-order Rényi entropies* $H_2, H_3, dots$. The collision entropy $H_2$ ($n = 1$) is particularly tractable since
$ Phi_1 (q_0, q_1) = integral D u thin log integral D v integral dif x thin tilde(cal(P))(x | z(u,v))^2, $
which is purely a Gaussian double integral against the squared channel.

== Boundary cases

The two extreme limits $alpha -> oo$ (min-entropy) and $alpha -> 0$ (Hartley entropy) probe complementary aspects of $p_F$ — the height of the peak and the size of the support — and are most cleanly analyzed by going back to $Z_n (F)$ rather than through @eq:final-renyi.

=== Min-entropy limit ($alpha -> oo$, $n -> oo$)

The Rényi entropy concentrates on the most probable output:
$ H_oo (X) := -log sup_x p_F (x) = lim_(alpha -> oo) H_alpha (X). $
At the level of $Z_n$, Laplace asymptotics give
$ Z_n (F) tilde.op [sup_x p_F (x)]^(n+1) thin e^(O(log n)) quad (n -> oo), $
so $-n^(-1) log Z_n -> H_oo$ as expected. Inside the replica formula, the *pointwise* Laplace step at fixed latent $z$ is correct,
$ integral dif x thin [tilde(cal(P))(x|z)]^(n+1) = [sup_x tilde(cal(P))(x|z)]^(n+1) thin e^(O(log n)), $
but the inner $v$-integral inside $Phi_n = integral D u thin log integral D v thin (dots)$ is itself dominated by Laplace at $n -> oo$. Writing $g(u, v) := log sup_x tilde(cal(P))(x mid(|) z(u, v))$, the leading large-$n$ behaviour is
$ 1/(n+1) thin Phi_n (q_0^*, q_1^*) -> integral D u thin sup_v thin g(u, v) - 1/(2(n+1)) thin v_*^2 + dots, $
and the supremum over $v$ at fixed $u$ generically does *not* reduce to a Gaussian average over $z ~ cal(N)(0, q_1^oo)$. A correct $alpha -> oo$ treatment therefore requires a joint Laplace/large-deviation analysis over both $x$ and $v$ (and stabilization of the conjugate parameters $hat(q)_(d, 0, 1)^*$). Skipping the $v$-Laplace step would give the heuristic expression
$ s_oo^"heur." = EE_(z ~ cal(N)(0, q_1^oo)) [H_oo (tilde(cal(P))(dot mid(|) z))], $ <eq:s-min>
the expected conditional min-entropy of $tilde(cal(P))(dot|z)$ at the limiting overlap. This is the natural $alpha = oo$ analogue of @eq:final-RS, but it should be regarded as a *heuristic estimate* rather than a derivation: it is exact only in regimes where $sup_v g(u, v)$ collapses onto a Gaussian average over $v$ (e.g. when $g$ is a quadratic function of $z(u, v)$, or when the channel-induced fluctuations dominate over the Gaussian prior weight). The Rényi monotonicity $s_oo <= s$ is in any case preserved. When the channel is sufficiently informative one expects $q_1^oo -> 1$ (so $tilde(cal(P)) -> Pout$ and $z ~ cal(N)(0,1)$), recovering the natural form $s_oo = EE_(z ~ cal(N)(0,1)) [H_oo (Pout (dot|z))]$.

=== Hartley entropy ($alpha -> 0$, $n -> -1$)

The opposite extreme is the *Hartley entropy*
$ H_0 (X) := log abs("supp"(p_F)) = lim_(alpha -> 0^+) H_alpha (X), $
which measures the size of the achievable output set rather than its peakedness. Setting $n = -1$ in @eq:Zn gives $Z_(-1) (F) = abs("supp"(p_F))$ directly. For continuous channels with $p_F > 0$ on $RR$ (e.g. Gaussian $Pout$), $H_0 = +oo$ trivially; the meaningful regime is channels with restricted support — most naturally, *discrete-output* channels.

==== Canonical example: hard-threshold (sign) channel

Take $Pout (x mid(|) h) = delta_(x, "sign"(h))$ with $x in {-1, +1}$, so that
$ p_F (x) = integral D z thin product_(i=1)^N Theta(x_i (F z)_i), $
and
$ Z_(-1) (F) = abs({y in {plus.minus 1}^N : exists z "with" "sign"(F z) = y}) =: T(F) $
counts the *sign patterns* realizable by the $N$ random hyperplanes ${z : (F z)_i = 0}$ in $RR^D$.

By Cover's hyperplane-counting theorem (Cover, 1965; equivalently, Wendel's formula, 1962), for $F$ with rows in *general position* — which holds *almost surely* for $F$ with a continuous (in particular Gaussian) distribution — the count $T(F)$ is *deterministic* and equals
$ T(F) = T(N, D) := 2 sum_(k=0)^(D-1) binom(N-1, k) quad "a.s." $ <eq:cover-count>
No expectation or concentration step is needed: the statement that $T(F)$ depends on $F$ only through general position is a geometric fact, not a typicality fact. Using $sum_(k=0)^(D-1) binom(N-1, k) tilde.op binom(N-1, D-1) tilde.op e^(N h_2 (1\/gamma))$ for $D-1 < (N-1)/2$, one obtains the celebrated *Cover dichotomy*:
#box(stroke: 0.5pt, inset: 8pt)[
$ s_0 = lim_(N -> oo) (log T(N, D))/N = cases(
  log 2 & gamma <= 2 quad ("linearly-separable phase"),
  h_2 (1\/gamma) quad & gamma > 2 quad ("over-determined phase"),
) $ <eq:s-hartley>
]
where
$
h_2 (p) := -p log p - (1 - p) log(1 - p). 
$ 
Below the Cover threshold $gamma_c = 2$ all $2^N$ sign patterns are typically realized (so the support is generic and $s_0 = log 2$ saturates the trivial upper bound); above it the achievable fraction shrinks as $h_2 (1\/gamma)\/log 2$, and $s_0 -> 0$ as $gamma -> oo$. The same result is recovered from the $n -> -1$ saddle-point analysis of @eq:Zn — the *Gardner-volume* calculation in its earliest, replica-symmetric form — in agreement with the general Rényi formula @eq:s-renyi.

=== Summary of limits

Combining the boundary cases with the formulas of Section 2 and Section 3:
- $alpha = 1$ (Shannon): $s = EE_z [H(tilde(cal(P))(dot|z))] + "prior overlap term"$, see @eq:final-RS.
- $alpha = 2$ (collision): $s_2 = -gamma^(-1) phi_1^* / 1 = -phi_1^* \/ gamma$, with $Phi_1 (q_0, q_1) = integral D u thin log integral D v integral dif x thin tilde(cal(P))^2$.
- $alpha -> oo$ (min): $s_oo = EE_z [H_oo (tilde(cal(P))(dot|z))]$, see @eq:s-min.
- $alpha -> 0$ (Hartley): $s_0 = log abs("supp")$, channel-dependent (sign channel: @eq:s-hartley).

The chain $s_0 >= s >= s_2 >= s_oo$ encodes the ordering of moments of $p_F$.

= Worked example: noiseless sign channel <sec:sign>

== Setup

Take the deterministic sign channel
$ Pout (x mid(|) h) = delta_(x, "sign"(h)), quad x in {-1, +1}. $
This is the same channel that produced the Cover dichotomy at $alpha = 0$ in @eq:s-hartley; here we work it out across the full Rényi profile. Throughout this section let
$ Phi(t) := integral_(-oo)^t (e^(-s^2 \/ 2)) / sqrt(2 pi) thin dif s, quad phi(t) := Phi'(t) = e^(-t^2 \/ 2) / sqrt(2 pi), $
denote the standard normal CDF and density (not to be confused with the action $phi_n$ and the integral $Phi_n$ of Section 2 — both carry a subscript).

The effective channel @eq:Ptilde collapses to the *probit*
$ tilde(cal(P))(x mid(|) z) = Pr_(w ~ cal(N)(0,1)) ["sign"(z + sqrt(1 - q_1) thin w) = x] = Phi(x z \/ sqrt(1 - q_1)). $ <eq:sign-Ptilde>
A hard threshold convolved with a Gaussian of variance $1 - q_1$: as $q_1 -> 1$ the threshold sharpens onto the deterministic channel; as $q_1 -> 0$ the noise washes it out and $tilde(cal(P)) -> 1\/2$.

== Shannon entropy ($alpha = 1$)

=== Conditional output entropy

The output entropy at fixed $z$ is the binary entropy of @eq:sign-Ptilde:
$ H(tilde(cal(P))(dot|z)) = h_2(Phi(z \/ sqrt(1 - q_1))). $
Substituting in @eq:dnPhi and changing variable to $t = z \/ sqrt(1 - q_1)$ (which is $cal(N)(0, q_1\/(1 - q_1))$ when $z ~ cal(N)(0, q_1)$),
$ partial_n Phi_n |_(n=0) = -EE_(t ~ cal(N)(0, thin q_1 \/ (1 - q_1))) [h_2(Phi(t))]. $ <eq:sign-condH>
This is monotonically decreasing in $q_1$: at $q_1 = 0$ the latent $t$ is a point mass at $0$ and $h_2(Phi(0)) = log 2$; at $q_1 = 1$ the variance of $t$ diverges, $|t|$ is typically large, $Phi(t) in {0, 1}$ a.s., and the conditional entropy collapses to $0$.

=== Fisher information

From @eq:sign-Ptilde, $partial_z tilde(cal(P))(x|z) = (x \/ sqrt(1 - q)) thin phi(z\/sqrt(1 - q))$. Summing $(partial_z tilde(cal(P)))^2 \/ tilde(cal(P))$ over $x in {-1, +1}$ and using $1\/Phi + 1\/(1 - Phi) = 1\/(Phi (1 - Phi))$,
$ cal(I)(q) = 1/(1 - q) thin EE_(z ~ cal(N)(0, q)) [(phi^2(z\/sqrt(1 - q))) / (Phi(z\/sqrt(1 - q)) thin (1 - Phi(z\/sqrt(1 - q))))]. $ <eq:sign-fisher>
The change of variable $t = z\/sqrt(1 - q)$ rewrites this as a one-dimensional Gaussian quadrature against $cal(N)(0, q\/(1 - q))$:
$ cal(I)(q) = 1/(1 - q) thin EE_(t ~ cal(N)(0, thin q\/(1 - q))) [(phi^2(t)) / (Phi(t)(1 - Phi(t)))]. $ <eq:sign-fisher-t>

Two reference values:
- *$q = 0$:* $t = 0$ a.s., so $phi^2(0) \/ (Phi(0)(1 - Phi(0))) = (1\/(2 pi)) \/ (1\/4) = 2\/pi$, hence $cal(I)(0) = 2 \/ pi$.
- *$q -> 1$:* using the Mills tail $Phi(t)(1 - Phi(t)) tilde.op phi(t)\/|t|$ for $|t| -> oo$, the integrand grows like $|t| thin phi(t)$ in the tails. The bounded integral $C := integral_(RR) phi^2(t) \/ (Phi(t)(1 - Phi(t))) thin (dif t \/ sqrt(2 pi))$ converges, and a saddle estimate of @eq:sign-fisher-t gives $cal(I)(q) tilde.op C \/ sqrt(q (1 - q)) tilde.op C \/ sqrt(1 - q)$.

=== Saddle point and final formula

The Bayes-optimal saddle @eq:RS-eqs reads
$ q = (gamma thin cal(I)(q)) / (1 + gamma thin cal(I)(q)), quad hat(q) = gamma thin cal(I)(q), $ <eq:sign-saddle>
with $cal(I)(q)$ given by @eq:sign-fisher-t. The two extreme regimes are:
- *Small $gamma$:* $cal(I)(0) = 2\/pi$, so $hat(q)^* tilde.op 2 gamma \/ pi$ and $q^* tilde.op 2 gamma \/ pi -> 0$. The latent code is essentially invisible to the output and the entropy density tends to $log 2$.
- *Large $gamma$:* combining the saddle with $cal(I)(q) tilde.op C\/sqrt(1 - q)$ gives $hat(q)^* tilde.op gamma C \/ sqrt(1 - q^*)$, and $1 - q^* = 1\/(1 + hat(q)^*) tilde.op 1\/hat(q)^*$, hence $sqrt(1 - q^*) tilde.op (gamma C)^(-1)$, i.e.
  $ 1 - q^* tilde.op (gamma C)^(-2). $
  So $q^* -> 1$ slowly. The conditional-entropy term in @eq:final-RS vanishes (as $h_2(Phi(t))$ averaged against a Gaussian of diverging variance) and the prior bracket reduces to $-log(1 - q^*) - q^* tilde.op 2 log(gamma C) - 1$, so $s -> 0$ at rate $log(gamma) \/ gamma$.

The replica-symmetric Shannon entropy density is, from @eq:final-RS,
#box(stroke: 0.5pt, inset: 6pt)[
$ s = EE_(t ~ cal(N)(0, thin q^* \/ (1 - q^*))) [h_2(Phi(t))] + 1/(2 gamma) [-log(1 - q^*) - q^*], $ <eq:sign-shannon>
]
with $q^*$ solving @eq:sign-saddle.

The outer-1RSB analysis of the sign channel — Shannon-order reduction to the RS saddle @eq:sign-shannon, with the symmetry-induced collapse $q_0 = 0$ and a remarks on outer-replicon stability at finite Rényi order — is worked out in @app:1rsb-sign. We do *not* prove full absence of outer 1RSB at all $alpha$: parity symmetry forces $q_0 = 0$ on the RS branch but leaves the cross-group overlap $q_"s"$ to be checked by a replicon computation, which we do not carry out here. The only non-trivial RSB-like signature in the Rényi profile of this channel that we positively identify sits at $alpha = 0$, namely the Cover dichotomy @eq:s-hartley, which counts achievable output patterns rather than measuring posterior structure.

== Generic Rényi order ($q_0 = 0$ reduction) <sec:sign-renyi-q0>

Numerical solution of the five-dimensional system @eq:Q-inv-renyi–@eq:RS-renyi-conj shows that $q_0^* = 0$ for the sign channel across all tested values of $n$ and $gamma$. We exploit this observation — verified a posteriori at the end of this section — to reduce the saddle to a single scalar equation.

=== Simplified output integral

With $q_0 = 0$ the signal field $sqrt(q_0) u$ in @eq:hierarchical drops out: the output integral @eq:Phi loses its $u$-dependence and the outer $D u$ integral is trivial,
$ Phi_n (0, q_1) = log integral D v integral dif x thin [tilde(cal(P))(x mid(|) sqrt(q_1) thin v)]^(n+1). $
For the sign channel, inserting @eq:sign-Ptilde and changing variable to $t = v sqrt(q_1\/(1-q_1)) ~ cal(N)(0, rho)$ with $rho := q_1\/(1-q_1)$,
#box(stroke: 0.5pt, inset: 6pt)[
  $ Phi_n (0, q_1) = log EE_(t thin ~ cal(N)(0, rho)) [Phi(t)^(n+1) + (1-Phi(t))^(n+1)], quad rho := q_1 / (1 - q_1). $ <eq:sign-Phin-q0>
]

=== Reduced saddle-point equations

Setting $q_0 = 0$ in the third line of @eq:Q-inv-renyi immediately gives $hat(q)_0 = q_0 thin B_0^2 = 0$, confirming self-consistency. The first two lines then read
$
A = 1 / (1-q_1), quad B_0 = 1 / (1 + n q_1).
$
Using $A = 1 - hat(q)_d + hat(q)_1$ and $B_0 = 1 - hat(q)_d - n hat(q)_1$, subtracting gives
$
(n+1) hat(q)_1 = 1/(1-q_1) - 1/(1 + n q_1) = ((n+1) q_1)/((1-q_1)(1+n q_1)),
$
hence
$ hat(q)_1^* = q_1^* / ((1-q_1^*)(1+n q_1^*)). $ <eq:sign-q1hat>

Substituting into the first line of @eq:RS-renyi-conj (with $hat(q)_0 = 0$) yields a *single* self-consistency equation for $q_1$:
#box(stroke: 0.5pt, inset: 6pt)[
  $ q_1 / ((1-q_1)(1+n q_1)) = (2 gamma) / (n(n+1)) thin partial_(q_1) Phi_n (0, q_1). $ <eq:sign-saddle-renyi>
]
The five-dimensional system @eq:Q-inv-renyi–@eq:RS-renyi-conj thus reduces to a single one-dimensional root-finding problem.

=== Simplified prior and final formula

Inserting $q_0^* = 0$, $hat(q)_0^* = 0$, and @eq:sign-q1hat into @eq:Sprior-renyi-star, the $(n+1) q_1^*\/(2(1-q_1^*))$ terms cancel exactly and the prior collapses to
#box(stroke: 0.5pt, inset: 6pt)[
  $ lr(cal(S)_"prior"^"RS" \/ m mid(|))_(m -> 0)^* = n/2 thin log(1-q_1^*) + 1/2 thin log(1 + n q_1^*). $ <eq:sign-prior-simple>
]
Together with @eq:sign-Phin-q0 and the general Rényi formula @eq:final-renyi, the entropy density is
#box(stroke: 0.5pt, inset: 8pt)[
  $ s_alpha = - log(1-q_1^*) / (2 gamma) - log(1 + n q_1^*) / (2 n gamma) - 1/n thin log EE_(t thin ~ cal(N)(0, rho^*)) [Phi(t)^(n+1) + (1-Phi(t))^(n+1)], $ <eq:sign-renyi-main>
]
with $rho^* = q_1^*\/(1-q_1^*)$ and $q_1^*$ the positive root of @eq:sign-saddle-renyi.

_Shannon limit._ As $n -> 0$ ($alpha -> 1$): $log(1 + n q_1^*) \/ (2n gamma) -> q^*\/(2 gamma)$, and expanding the log-expectation at leading order in $n$ gives $-n thin EE_t [h_2(Phi(t))]$, so that $-1\/n$ times the log recovers $EE_t [h_2(Phi(t))]$. The formula @eq:sign-renyi-main then reduces to @eq:sign-shannon. The saddle @eq:sign-saddle-renyi similarly contracts to @eq:sign-saddle via $log(1+n q_1)/(n) -> q_1$ and the heat-equation identity $partial_(q_1) Phi_n(0, q_1)|_(n->0) = -(n\/2) thin cal(I)(q_1) + O(n^2)$.

== Collision entropy ($alpha = 2$)

For binary outputs,
$ sum_x tilde(cal(P))(x mid(|) z)^2 = Phi(xi)^2 + (1 - Phi(xi))^2 = 1 - 2 Phi(xi)(1 - Phi(xi)), quad xi := z \/ sqrt(1 - q_1). $
Inserting in $Phi_1$ from @eq:Phi at $n = 1$,
$ Phi_1 (q_0, q_1) = integral D u thin log integral D v thin [1 - 2 Phi(xi(u, v))(1 - Phi(xi(u, v)))], $
$ xi(u, v) := (sqrt(q_0) thin u + sqrt(q_1 - q_0) thin v) \/ sqrt(1 - q_1), $
a two-dimensional Gaussian quadrature, easily evaluated numerically. The collision-entropy saddle then comes from @eq:Q-inv-renyi and @eq:RS-renyi-conj at $n = 1$, and
$ s_2 = -1 / gamma thin lr(phi_1 mid(|))^* $
from @eq:final-renyi. Unlike the Shannon case, Nishimori symmetry is broken at $n = 1$, so all five RS parameters $(q_0^*, q_1^*, hat(q)_d^*, hat(q)_0^*, hat(q)_1^*)$ are generically nonzero. Under the $q_0 = 0$ assumption of @sec:sign-renyi-q0, which is observed to hold numerically also at $n = 1$, the formula @eq:sign-renyi-main specialises to
$ s_2 = -log(1-q_1^*) / (2 gamma) - log(1 + q_1^*) / (2 gamma) - log EE_(t thin ~ cal(N)(0, rho^*)) [1 - 2 Phi(t)(1-Phi(t))], $
with $q_1^*$ solving @eq:sign-saddle-renyi at $n = 1$, which simplifies to a single integral equation.

== Min-entropy ($alpha -> oo$)

For binary outputs and $z != 0$,
$ sup_x tilde(cal(P))(x mid(|) z) = max(Phi(z\/sqrt(1 - q_1)), Phi(-z\/sqrt(1 - q_1))) = Phi(|z|\/sqrt(1 - q_1)), $
so $H_oo (tilde(cal(P))(dot mid(|) z)) = -log Phi(|z|\/sqrt(1 - q_1))$. Inserting into the *heuristic* min-entropy formula @eq:s-min (subject to the caveat of Section 3.4.1 — the $v$-Laplace step is not justified in general), the channel's $z -> -z$ symmetry gives
$ s_oo^"heur." = -2 integral_0^oo (e^(-z^2 \/ (2 q_1^oo))) / sqrt(2 pi q_1^oo) thin log Phi(z \/ sqrt(1 - q_1^oo)) thin dif z. $ <eq:sign-min-heur>
At $q_1^oo -> 0$ the integrand reduces to $-log Phi(0) = log 2$ (uninformative limit); at $q_1^oo -> 1$ the integrand sharpens onto $z > 0$ where $Phi -> 1$, hence $s_oo^"heur." -> 0$. A rigorous min-entropy analysis for the sign channel would require a separate large-deviation treatment of the joint $(v, x)$ supremum at fixed $u$ inside @eq:Phi.

== Hartley entropy ($alpha -> 0$): Cover dichotomy

The $alpha -> 0$ result was already derived for this channel in Section 3.4; for completeness we recall
$ s_0 = cases(
  log 2 quad & gamma <= 2 quad ("linearly separable phase"),
  h_2 (1 \/ gamma) quad & gamma > 2 quad ("over-determined phase"),
) $
the *Cover dichotomy* of @eq:s-hartley. Below the threshold $gamma_c = 2$ all $2^N$ sign patterns are typically realized and $s_0 = log 2$ saturates; above it, the achievable fraction shrinks with the binary entropy of $1\/gamma$.

== Summary

The noiseless sign channel admits closed-form expressions (modulo at most a two-dimensional Gaussian quadrature) at every Rényi order considered, with the *only* non-trivial scalar input being the Bayes-optimal overlap $q^* = q^*(gamma)$ from @eq:sign-saddle entering the Shannon case. The Rényi profile satisfies $s_0 >= s >= s_2 >= s_oo$, with all four collapsing onto $log 2$ as $gamma -> 0$ (uninformative regime). For $gamma -> oo$ the orderings $s, s_2, s_oo -> 0$, while $s_0$ retains the explicit Cover form $h_2 (1\/gamma)$ that decays only logarithmically — a faithful diagnostic that, even when typical outputs concentrate, the *support* of $p_F$ remains exponentially rich. @fig:renyi-profile shows the full RS Rényi entropy profile across $alpha in [0.02, 10]$ obtained by numerically solving the saddle-point system, sandwiched between the Cover envelope at $alpha -> 0$ and the min-entropy limit at $alpha -> oo$.

#figure(
  image("plots/entropy_sign_renyi_all.pdf", width: 92%),
  caption: [
    Replica-symmetric Rényi entropy density $s_alpha(gamma)$ for the
    hidden-manifold sign channel as a function of the load $gamma = N\/D$,
    for $alpha in {0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1, 2, 5, 10}$
    (color-coded by $log alpha$, viridis from purple to yellow). The black
    curve is the Shannon entropy ($alpha = 1$, @eq:sign-shannon) and the
    gray dash-dot envelope is the Cover/Hartley limit $s_0 = h_2(1\/gamma)$
    for $gamma > 2$, $s_0 = log 2$ otherwise (@eq:s-hartley). Curves with
    $alpha != 1$ are produced by the $q_0 = 0$ reduction of
    @sec:sign-renyi-q0 (for $alpha in {0.4, 0.5, 0.9}$ the full
    five-parameter saddle is used; the two ansätze agree to numerical
    precision wherever both have been computed). Rényi monotonicity
    $partial_alpha s_alpha <= 0$ is satisfied across the entire grid; all
    curves collapse onto $log 2$ as $gamma -> 0$.
  ],
) <fig:renyi-profile>

#bibliography("bibliography.bib")

#show: arkheion-appendices

= Spectrum of $hat(Q)$ under the RS ansatz <app:spectrum>

Decompose $RR^M$ into three invariant subspaces of $hat(Q)$:

+ *Within-block antisymmetric:* $sum_a v^(alpha a) = 0$ for every $alpha$. Dimension $m n$. On such $v$,
  $ (hat(Q) v)^(alpha a) = hat(q)_d thin v^(alpha a) + hat(q)_1 sum_(b != a) v^(alpha b) + hat(q)_0 sum_(beta != alpha) sum_b v^(beta b) = (hat(q)_d - hat(q)_1) thin v^(alpha a). $
  Eigenvalue $lambda_1 = hat(q)_d - hat(q)_1$, multiplicity $m n$.

+ *Between-block antisymmetric:* $v^(alpha a) = c_alpha$ with $sum_alpha c_alpha = 0$. Dimension $m - 1$.
  $ (hat(Q) v)^(alpha a) = (hat(q)_d + n hat(q)_1) c_alpha + (n+1) hat(q)_0 sum_(beta != alpha) c_beta = [hat(q)_d + n hat(q)_1 - (n+1) hat(q)_0] c_alpha. $
  Eigenvalue $lambda_2 = hat(q)_d + n hat(q)_1 - (n+1) hat(q)_0$, multiplicity $m - 1$.

+ *Fully symmetric:* $v^(alpha a) equiv "const"$. Dimension $1$. Eigenvalue
  $ lambda_3 = hat(q)_d + n hat(q)_1 + (m-1)(n+1) hat(q)_0. $

Dimensions add to $m n + (m-1) + 1 = m(n+1) = M$, and @eq:logdet follows. Direct enumeration of $M^2$ pairs gives @eq:trace.

= The $m -> 0$ expansion of $log det (I - hat(Q))$ <app:m-zero>

With $A, B, C$ as in the main text, $log det(I - hat(Q)) = m n log A + (m-1) log B + log C$. Note $C - B = -m(n+1) hat(q)_0$, so for small $m$,
$ log C = log B + log(1 - m(n+1) hat(q)_0 \/ B) = log B - (m(n+1) hat(q)_0) / B + O(m^2). $
Hence
$ (m - 1) log B + log C = m log B - (m(n+1) hat(q)_0) / B + O(m^2), $
and
$ 1/m log det(I - hat(Q)) -> n log A + log B_0 - ((n+1) hat(q)_0) / B_0 quad "as" m -> 0, $
where $B_0 = B|_(m=0) = 1 - hat(q)_d - n hat(q)_1 + (n+1) hat(q)_0$. Combined with $1/m Tr(Q hat(Q)) -> (n+1) hat(q)_d + n(n+1) q_1 hat(q)_1 - (n+1)^2 q_0 hat(q)_0$, this gives @eq:Sprior-m0.

= The $n -> 0$ expansion of the prior term <app:n-zero-prior>

We differentiate @eq:Sprior-m0 in $n$ at fixed parameters. Set $L(n) = n log A + log B_0 - (n+1) hat(q)_0 / B_0$ and $T(n) = (n+1) hat(q)_d + n(n+1) q_1 hat(q)_1 - (n+1)^2 q_0 hat(q)_0$. Then
$ partial_n B_0 = -hat(q)_1 + hat(q)_0, $
$ partial_n [(n+1) hat(q)_0 \/ B_0] = hat(q)_0 / B_0 - ((n+1) hat(q)_0 (hat(q)_0 - hat(q)_1)) / B_0^2. $
At $n = 0$, $B_0 -> B := 1 - hat(q)_d + hat(q)_0$, hence
$ partial_n L |_(n=0) = log A - hat(q)_1 / B + (hat(q)_0 (hat(q)_0 - hat(q)_1)) / B^2. $
Likewise $partial_n T|_(n=0) = hat(q)_d + q_1 hat(q)_1 - 2 q_0 hat(q)_0$. Combining yields @eq:dnSprior.

= Order-$n$ saddle-point equations and Nishimori collapse <app:RS-saddle>

The free entropy $phi_n$ vanishes at $n = 0$ when evaluated at the saddle (since $log Z_0 (F) = 0$). Consequently the bare stationarity of $phi_n|_(n=0)$ in $(q_0, q_1, hat(q)_d, hat(q)_0, hat(q)_1)$ is degenerate, and the physical equations are extracted from the linear-in-$n$ piece.

The cleanest route — and the one we follow — is to *not* introduce a separate scaling ansatz at the level of @eq:Sprior-m0, but rather to work directly with the finite-$n$ Rényi saddle of @eq:Sprior-renyi-star. That object is by construction evaluated *at* the matrix-inversion identities @eq:Q-inv-renyi, so the conjugate parameters are no longer independent unknowns and the only scaling assumption we need is on the surviving overlaps $(q_0, q_1)$.

== Bayes-optimal scaling

For the *Bayes-optimal entropy* $S = EE_F H_F (X)$ — matched-prior teacher-student inference — Nishimori symmetry forces $q_0^* = 0$ at $n = 0$ (cross-block independence; see Section 2.5.1) and leaves a single surviving overlap $q_1^* = q^* > 0$. The correct scaling in the $n -> 0$ Shannon limit is therefore
$ q_1 = q + O(n), quad hat(q)_1 = hat(q) + O(n), quad q_0, hat(q)_0, hat(q)_d = O(n), $ <eq:correct-scaling>
where $q$ and $hat(q)$ denote the limiting Bayes-optimal values. The earlier ansatz $q_1 = q_0 + n thin r_1 + dots$ (with $q_0$ as the leading scale) is *incompatible* with $q_1^* = q^* > 0$ and is discarded. The matrix-inversion identity @eq:Q-inv-renyi at $n = 0$ then forces $hat(q)_d^* -> 0$ as well: from $1 - q = 1\/A$ with $A = 1 - hat(q)_d + hat(q)_1$,
$ 1 - hat(q)_d^* + hat(q)^* = 1 \/ (1 - q) = 1 + hat(q)^* "(using" q = hat(q)\/(1+hat(q)) "below)" quad ==> quad hat(q)_d^* = 0, $
consistent with @eq:correct-scaling.

== Order-$n$ expansion of the prior saddle

Inserting @eq:correct-scaling into @eq:Sprior-renyi-star and expanding at order $n$, the $O(1)$ piece vanishes (as it must, since $S_"prior"^"RS" \/ m |_(n=0, m -> 0) = 0$ at the saddle) and the surviving order-$n$ contribution is
$ lr(cal(S)_"prior"^"RS" \/ m mid(|))_(m -> 0)^* = n thin g_"prior" (q, hat(q)) + O(n^2), quad g_"prior" (q, hat(q)) := -1/2 thin q hat(q) + 1/2 thin hat(q) - 1/2 thin log(1 + hat(q)). $ <eq:gprior>
The $q_0, hat(q)_0, hat(q)_d$ contributions enter only at $O(n^2)$ — their leading $O(n)$ pieces cancel pairwise inside @eq:Sprior-renyi-star — so the surviving Bayes-optimal action depends only on the two scalars $(q, hat(q))$.

== Stationarity in $(q, hat(q))$

The order-$n$ piece of $phi_n$ at the Bayes-optimal saddle is
$ partial_n thin phi_n |_(n=0)^"Bayes-opt." = g_"prior" (q, hat(q)) + gamma thin partial_n thin Phi_n |_(n=0) = g_"prior" (q, hat(q)) - gamma thin F(q), $
with
$ F(q) := EE_(z ~ cal(N)(0, q)) [H(tilde(cal(P))(dot|z))] $
the conditional output entropy at common-cause variance $q$ (cf. @eq:dnPhi). Stationarity gives:
- *In $hat(q)$* (only $g_"prior"$ depends on $hat(q)$):
  $ partial_(hat(q)) g_"prior" = -q/2 + 1/2 - 1/(2(1 + hat(q))) = 0 quad ==> quad q = hat(q)/(1 + hat(q)). $
- *In $q$:* using $partial_q g_"prior" = -hat(q) \/ 2$ and the heat-equation identity $F'(q) = -(1\/2) thin cal(I)(q)$ (see below),
  $ -hat(q) / 2 - gamma thin F'(q) = -hat(q)/2 + gamma/2 thin cal(I)(q) = 0 quad ==> quad hat(q) = gamma thin cal(I)(q). $

Together these reproduce @eq:RS-eqs.

== Heat-equation derivation of $F'(q)$

Both the Gaussian density $phi_q (z) = (2 pi q)^(-1\/2) exp(-z^2 \/ (2 q))$ and the noise-augmented channel $tilde(cal(P))(x|z) = integral D w thin Pout (x | z + sqrt(1 - q) thin w)$ satisfy heat equations in $q$:
$ partial_q phi_q (z) = 1/2 partial_z^2 phi_q (z), quad partial_q tilde(cal(P))(x|z) = -1/2 partial_z^2 tilde(cal(P))(x|z). $ <eq:heat-eqs>
The opposite signs encode the fact that increasing $q$ broadens the prior and sharpens the channel (the residual variance is $1 - q$). Differentiating $F(q) = integral dif z thin phi_q (z) sum_x [-tilde(cal(P))(x|z) log tilde(cal(P))(x|z)]$ in $q$, integrating by parts twice in $z$ on the first contribution, and using $sum_x partial_z^2 tilde(cal(P)) = partial_z^2 sum_x tilde(cal(P)) = 0$, all "log" terms cancel and one is left with
$ F'(q) = -1/2 integral dif z thin phi_q (z) sum_x (partial_z tilde(cal(P))(x|z))^2 / tilde(cal(P))(x|z) = -1/2 thin cal(I)(q), $
with $cal(I)(q)$ the Fisher information of @eq:fisher. The minus sign is the source of the *positive* coupling $hat(q) = + gamma cal(I)(q)$ in the saddle: as $q$ increases the channel becomes sharper, the conditional entropy $F(q)$ decreases, and the missing entropy is transferred to the prior.

== Cross-block Nishimori identities

In the present RS parametrization the Bayes-optimal Shannon limit gives
$ q_0^* = 0, quad hat(q)_0^* = 0, quad hat(q)_d^* = 0, quad q_1^* = q^*, quad hat(q)_1^* = hat(q)^*. $
The vanishing of $q_0^*$ is an exact symmetry property of the Bayes-optimal teacher-student model (one-block marginal collapses to the bare prior at $n = 0$; cf. §2.5.1). The vanishing of the conjugates $hat(q)_0^*$ and $hat(q)_d^*$ is then a consequence of the matrix-inversion identities @eq:Q-inv-renyi at $n = 0$ together with the scaling @eq:correct-scaling, *not* of an identity of the form $hat(q)_0^* = hat(q)_1^*$.

The latter would arise in a different parametrization that introduces the teacher--student overlap $m$ and the student--student overlap $q$ as separate variables, in which case the standard Nishimori identity reads $hat(m) = hat(q)$. Here teacher and student replicas have already been symmetrized inside the $(n+1)$-block — there is no separate $m$-parameter — so $hat(m) = hat(q)$ does not appear in the present saddle equations. The $n$-expansion uses the collapse above to reduce the five-dimensional system @eq:Q-inv-renyi and @eq:RS-renyi-conj to the two-dimensional Bayes-optimal saddle @eq:RS-eqs.

The Fisher information $cal(I)(q)$ of @eq:fisher is the natural "signal-to-noise" measure controlling how much information $X$ carries about the latent code $Z$ at fixed overlap $q$.
This is the natural "signal-to-noise" measure controlling how much information $X$ carries about the latent code $Z$ at fixed overlap $q$.

= One-step replica-symmetry breaking <app:1rsb>

The action @eq:Smn carries two replica indices: the *internal* block index $a = 0, dots, n$ (the "molecule" associated with one copy of $Z_n$) and the *outer* disorder index $alpha = 1, dots, m$ (used to compute $EE_F log Z_n$). A Parisi instability of the quenched free entropy should first be looked for in the outer $alpha$-direction: the genuine Parisi level for $EE_F log Z_n$ is the one that hierarchically partitions the $alpha$-replicas. Internal breaking of the $(n+1)$-block is in principle possible at finite Rényi order, but it is a secondary check, not the primary RSB of the quenched problem. The appendix below sets up the *outer-1RSB* ansatz, derives its free entropy, comments on the Bayes-optimal/symmetric-channel reductions, and applies the framework to the noiseless sign channel of Section 4.

== The outer-1RSB ansatz

We keep the inner $(n+1)$-block replica-symmetric, with within-block off-diagonal overlap $q_"in"$, and partition the outer index $alpha = 1, dots, m$ into Parisi groups of size $mu$. Replicas in the same outer group share the cross-block overlap $q_"s"$; replicas in different outer groups share the cross-group overlap $q_0$. Explicitly, with $(alpha, a)$ a generic replica index and $g(alpha)$ the Parisi group containing $alpha$,
$ Q_((alpha, a),(beta, b)) = cases(
  1 quad & alpha = beta\, a = b,
  q_"in" & alpha = beta\, a != b,
  q_"s" & alpha != beta\, g(alpha) = g(beta),
  q_0 & alpha != beta\, g(alpha) != g(beta),
) $ <eq:outer1rsb-Q>
with the analogous structure on $hat(Q)$ in terms of $(hat(q)_d, hat(q)_"in", hat(q)_"s", hat(q)_0)$. The RS ansatz of §2.3 is recovered when $q_"s" = q_0$ (no outer breaking) or in the trivial Parisi limits $mu in {1, m}$. A genuine outer-RSB instability corresponds to
$ q_"s" > q_0, $
*not* to breaking the $(n+1)$-replica molecule.

== Hierarchical Gaussian decomposition

The covariance @eq:outer1rsb-Q admits the four-level Gaussian decomposition
$ h^(alpha a) = sqrt(q_0) thin u + sqrt(q_"s" - q_0) thin v_(g(alpha)) + sqrt(q_"in" - q_"s") thin r^alpha + sqrt(1 - q_"in") thin w^(alpha a), $ <eq:outer1rsb-hierarchy>
with $u, {v_g}, {r^alpha}, {w^(alpha a)}$ mutually independent standard Gaussians at successive levels of the hierarchy: $u$ is shared across all replicas, $v_g$ across all $alpha$ in the same outer Parisi group, $r^alpha$ across all $a$ in a single $(n+1)$-block, and $w^(alpha a)$ is independent at every leaf. The new layer relative to @eq:hierarchical is $v_g$.

Define the noise-augmented channel
$ tilde(P)_(q_"in") (x mid(|) z) := integral D w thin Pout (x mid(|) z + sqrt(1 - q_"in") thin w), $ <eq:Ptilde-outer1rsb>
and the *block kernel*
$ L_n (u, v) := integral D r integral dif x thin tilde(P)_(q_"in") (x mid(|) sqrt(q_0) u + sqrt(q_"s" - q_0) v + sqrt(q_"in" - q_"s") r)^(n+1) . $ <eq:Ln-outer1rsb>
The $r$-integral binds the $(n+1)$ inner replicas of one disorder block into a single observation $x$; the resulting $L_n$ depends on the outer common-causes $(u, v)$ that are seen by the entire block.

== Output free entropy

Outer 1RSB couples disorder blocks within the same Parisi group through the shared field $v_g$. Carrying out the integrations layer by layer (first $w$ inside each block, then $x$ to assemble $L_n$, then $v_g$ across the $mu$ blocks of one Parisi group, then $u$ across all Parisi groups), the disorder-averaged output partition function takes the nested form
$ Z_"out"^"out-1RSB" = integral D u thin [integral D v thin L_n (u, v)^mu]^(m\/mu). $
After the standard $m -> 0$ limit, the outer-1RSB output free entropy is
#box(stroke: 0.5pt, inset: 6pt)[
$ Phi_(n, mu)^"out-1RSB" (q_0, q_"s", q_"in") = 1 / mu integral D u thin log integral D v thin L_n (u, v)^mu . $ <eq:Phi-outer1rsb>
]
For $q_"s" = q_0$ the outer field $v$ drops out of $L_n$, the inner $D v$-integral is trivial, and @eq:Phi-outer1rsb reduces to the RS kernel @eq:Kn (with $q_1 = q_"in"$). The trivial Parisi limits $mu -> 1$ and $mu -> 0$ likewise recover the RS expression.

The prior contribution comes from diagonalizing $hat(Q)$ in its outer-1RSB invariant subspaces. The structure parallels @app:spectrum but with *four* eigenvalue sectors — the "within outer Parisi group, between $(n+1)$-blocks" sector being the new layer — and yields a four-parameter prior action $cal(S)_"prior"^"out-1RSB" (q_0, q_"s", q_"in", hat(q)_d, hat(q)_0, hat(q)_"s", hat(q)_"in")$, extremized together with @eq:Phi-outer1rsb over the seven scalars and the Parisi block size $mu$. Stationarity in $mu$ — the Parisi equation $partial_mu phi_(n, mu)^"out-1RSB" = 0$ — is the qualitatively new feature.

== Bayes-optimal Shannon limit

In the Shannon limit $n -> 0$, outer RS is automatic at the order needed for the entropy: $Z_0 (F) = 1$ pointwise, so the disorder replicas decouple at order $n$ (cf. §2.5.4) and outer-RSB corrections to the Shannon free entropy enter only at $O(n^2)$. Concretely, $L_0 (u, v) = 1$ identically and any $mu$-dependence in @eq:Phi-outer1rsb cancels at $O(n)$, so the leading Shannon piece of $Phi_(n, mu)^"out-1RSB"$ does not see the outer Parisi exponent $mu$ at all. The Shannon entropy of @eq:final-RS is therefore *unaffected* by outer 1RSB at the level of the quenched derivative.

Inner replica symmetry inside the $(n+1)$-block, on the other hand, is a Bayes-optimal ansatz: in the matched teacher-student model the Nishimori identity makes teacher and posterior samples exchangeable, equating the teacher--student and student--student overlap moments and supporting a single inner overlap $q_"in" = q^*$ at the saddle. This is the Bayes-optimal RS ansatz, not a theorem: exchangeability does not by itself prove RS — Parisi RSB ensembles are still exchangeable after averaging over the random hierarchy — so the RS formula should be read as exact under the standard hypothesis that the Nishimori saddle is locally stable against replicon and inner-1RSB perturbations. For the dense Gaussian HMM with simple log-concave or convex channels this is the expected situation; for more structured or hard channels the corresponding stability check should be performed.

== Finite Rényi order and symmetric channels

At finite Rényi order $n != 0$ the Bayes-optimal interpretation is no longer available — the measure is tilted by $p_F (x)^n$ — and the outer-1RSB ansatz @eq:Phi-outer1rsb is genuinely an ansatz to be tested. If the channel has the parity symmetry @eq:channel-symmetry, the one-block marginal @eq:rho-n is even in $z$ and the cross-group overlap satisfies
$ q_0 = 0 quad "on the RS branch, for every" n. $
Importantly, this is *not* sufficient to exclude outer 1RSB at finite $n$: a parity-symmetric problem can still develop an outer Parisi instability through
$ q_"s" > q_0 = 0, $
in which case @eq:Phi-outer1rsb has nontrivial $v$-dependence even though $q_0 = 0$. Symmetry kills the $u$-channel of the cross-block coupling but leaves the $v$-channel inside each Parisi group. A full check at finite Rényi order therefore requires either an explicit outer-replicon computation around the RS saddle, or the direct extremization of @eq:Phi-outer1rsb at $q_0 = 0$ with $q_"s"$ free.

== Dynamical 1RSB and the hard phase

Independent of the static structure, the action @eq:Phi-outer1rsb may admit a *secondary*, non-Nishimori saddle with $q_"s" > q_0$ and $mu in (0, 1)$. This dynamical saddle is not the global thermodynamic optimum, but it encodes physical structure of the Bayes-optimal posterior $p(z mid(|) x, F)$ that is invisible to the static thermodynamic free energy:

+ *Clustering of the posterior.* Above a *dynamical 1RSB threshold* $gamma_d$, channel-dependent, the posterior shatters into exponentially many metastable clusters. A typical cluster has weight $e^(-N Sigma)$, with the *complexity* $Sigma >= 0$ counting clusters per unit $N$.

+ *Information-computation gap.* Local algorithms (belief propagation, AMP, Glauber dynamics) get trapped inside a single cluster and fail to recover the planted $z^*$ even when the static RS overlap $q^*$ is positive. The *hard phase* is the regime $gamma_d < gamma < gamma_s$ where information-theoretic recovery is possible but no efficient algorithm achieves it; below $gamma_d$ the problem is "easy", above $gamma_s$ ("static spinodal") the planted cluster ceases to dominate and recovery is information-theoretically impossible.

+ *Slow MCMC mixing.* Posterior samplers based on local moves have mixing time exponential in $N$ above $gamma_d$, since transitions between clusters require crossing macroscopic free-energy barriers.

The complexity is extracted from the outer-1RSB action by the standard Monasson construction
$ Sigma (q_0, q_"s") = lr(partial_mu phi_(n, mu)^"out-1RSB" mid(|))_(n=0, thin mu -> 0^+), $
extremized over the cross-group sector at fixed inner overlap $q_"in" = q^*$ from the RS solution. The dynamical transition $gamma_d$ is the smallest $gamma$ at which $Sigma >= 0$ admits a nontrivial root with $q_"s" > q_0$. For richer settings — committee machines, sparse priors, low-rank matrix estimation — dynamical 1RSB is well-documented and underlies the algorithmic phase diagrams of the inference literature.

== Application to the noiseless sign channel <app:1rsb-sign>

We now specialize the framework to the noiseless sign channel of Section 4. The noise-augmented kernel @eq:Ptilde-outer1rsb is the probit with variance $1 - q_"in"$, $tilde(P)_(q_"in") (x mid(|) z) = Phi(x z \/ sqrt(1 - q_"in"))$. With parity symmetry, $q_0 = 0$ on the RS branch. Setting $q_0 = 0$, the block kernel @eq:Ln-outer1rsb evaluates to
$ L_n (u, v) thin |_(q_0 = 0) = sum_(x = plus.minus 1) integral D r thin Phi(x thin xi)^(n+1), quad xi := (sqrt(q_"s") thin v + sqrt(q_"in" - q_"s") thin r) \/ sqrt(1 - q_"in"), $
and the outer-1RSB output free entropy reduces to a $u$-independent integral,
$ Phi_(n, mu)^"out-1RSB" thin |_(q_0 = 0) = 1 / mu thin log integral D v thin (sum_(x = plus.minus 1) integral D r thin Phi(x thin xi)^(n+1))^mu. $

=== Reduction at the Bayes-optimal Shannon point

In the Shannon limit $n -> 0$ with $q_"in" = q^*$, the inner $r$-integral and the $x$-sum recombine because $L_0 = 1$ pointwise. Expanding to first order in $n$ at fixed $(q_"s", mu)$, the dependence on $mu$ and $q_"s"$ in @eq:Phi-outer1rsb cancels at $O(n)$ — consistent with the general Shannon argument of §2.5.4 — and the conditional-entropy formula @eq:sign-condH is recovered:
$ partial_n thin Phi_(n, mu)^"out-1RSB" thin |_(n = 0, thin q_"in" = q^*) = -EE_(t ~ cal(N)(0, thin q^*\/(1 - q^*))) [h_2 (Phi(t))]. $
At Shannon order, outer 1RSB therefore adds *no information* on top of the RS formula @eq:sign-shannon, regardless of the value of $(q_"s", mu)$. This does not, however, prove absence of dynamical 1RSB: the latter is a property of subleading saddles in the action, not of the leading Shannon derivative.

=== Remarks on stability at finite Rényi order

For the noiseless sign channel the posterior $p(z mid(|) x, F) prop p(z) thin product_i Theta(x_i (F z)_i)$ is a Gaussian measure restricted to an intersection of half-spaces — a log-concave measure on a convex set. Standard log-concavity arguments make full RS plausible at all $gamma > 0$ in the Shannon problem, in agreement with the empirical performance of AMP/belief-propagation attaining $q^*$. At finite Rényi order, however, the tilted measure $p_F (x)^(n+1)$ is not the original posterior, and absence of outer 1RSB requires a separate replicon/stability computation around the $q_0 = 0$ RS saddle — symmetry alone forces $q_0 = 0$ but leaves $q_"s"$ to be checked. We do not perform this stability analysis here; it is left as a natural follow-up to validate the RS Rényi profile of @fig:renyi-profile beyond a neighbourhood of $alpha = 1$.
