#set math.equation(numbering: "(1)")

= Conditional Hopfield model

== Setting

Consider a Hopfield model @hopfield1982neural with $N$ neurons and $P = alpha N$ patterns
$bold(xi)^mu in {plus.minus 1}^N$ ($mu = 1, dots, P$). The patterns are random
and uncorrelated, with i.i.d. entries drawn uniformly from ${plus.minus 1}$.
The Boltzmann distribution reads
$
  p(bold(x)) prop e^(-beta E(bold(x))), quad
  E(bold(x)) = -1/2 sum_(i, j) J_(i j) x_i x_j,
  quad J_(i j) = 1/N sum_(mu = 1)^P xi_i^mu xi_j^mu.
$
(The $i = j$ diagonal gives an additive constant $-alpha beta N slash 2$ that
does not affect the physics; keeping the full sum makes the Hamiltonian
manifestly $-beta E = (beta slash (2 N)) sum_mu A_mu^2$ with
$A_mu = sum_i xi_i^mu x_i$.)

We clamp a subset $U subset [N]$ of the spins to the first pattern, i.e.
$x_i = xi_i^1$ for all $i in U$. Write $|U| = (1 - t) N$, so that
$t = 1 - |U| slash N$ is the fraction of _free_ spins, and let
$F = [N] without U$ with $|F| = t N$. The set $U$ is drawn uniformly at random
among subsets of size $(1 - t) N$; by symmetry the replica calculation does not
depend on which particular subset has been chosen.

The Hamiltonian of the clampled system
takes the form:
$
  E_U(x_F) = -1/2 sum_(i, j in F) J_(i j) x_i x_j - sum_(i in F) h^U_i x_i + "const",
$
where the effective field on free sites is
$
  h^U_i = sum_(j in U) J_(i j) xi_j^1
    = (1 - t) xi_i^1 + 1/N sum_(mu >= 2) xi_i^mu sum_(j in U) xi_j^1 xi_j^mu.
$

We are interested in the retrieval properties of the first pattern as a function
of $(alpha, t)$: how much information about $bold(xi)^1$ do the free spins on
$F$ carry when a fraction $1 - t$ of them is already pinned to $bold(xi)^1$?

== Replica computation

The disorder-averaged free energy per spin is
$
  -beta f = lim_(N -> oo) 1/N overline(ln Z), quad
  Z = sum_(bold(x) in {plus.minus 1}^N \ x_i = xi_i^1 forall i in U) e^(-beta E(bold(x))),
$
with overline denoting the quenched average over the patterns. We evaluate
$overline(ln Z)$ by the replica trick,
$
  overline(ln Z) = lim_(n -> 0) (overline(Z^n) - 1) / n,
$
introducing $n$ copies of the system with replica index $a = 1, dots, n$ and
evaluating $overline(Z^n)$ for integer $n$ before analytically continuing.

=== Gauge transformation

It is convenient to remove pattern 1 from the disorder by performing the gauge
transformation
$
  sigma_i^a = xi_i^1 x_i^a,
$
which maps $x_i^a in {plus.minus 1}$ to $sigma_i^a in {plus.minus 1}$ with the
same support. In the new variables the clamped spins become $sigma_i = 1$ for
all $i in U$ (since $xi_i^1 dot xi_i^1 = 1$), while on the free sites $i in F$
the variables $sigma_i^a$ remain free binary spins. Defining the rotated noise
patterns
$
  tilde(xi)_i^mu = xi_i^1 xi_i^mu quad (mu >= 2),
$
one has $tilde(xi)_i^mu in {plus.minus 1}$ still i.i.d. uniform by symmetry of
the distribution of $bold(xi)^mu$. The Hebbian energy becomes
$
  -beta E = beta / (2 N) sum_(mu = 1)^P A_mu^2,
$
with
$
  A_1(bold(sigma)) &= sum_(i = 1)^N sigma_i
    = (1 - t) N + sum_(i in F) sigma_i,\
  A_mu(bold(sigma)) &= sum_(i in U) tilde(xi)_i^mu + sum_(i in F) tilde(xi)_i^mu sigma_i
    quad (mu >= 2).
$
Pattern 1 contributes a deterministic "condensed" term proportional to the
total magnetisation $(1 slash N) sum_i sigma_i = (1 - t) + t m$ (free-sector
magnetisation rescaled by $t$, plus the clamped contribution $(1 - t)$). The
remaining $alpha N - 1$ patterns contribute an i.i.d. Gaussian-like noise that
will be handled by replica/HS tricks.

=== Order parameters

On the free sector $F$ we introduce the intrareplica magnetisation and the
Edwards--Anderson overlap
$
  m^a = 1 / (t N) sum_(i in F) sigma_i^a, quad
  q^(a b) = 1 / (t N) sum_(i in F) sigma_i^a sigma_i^b quad (a != b).
$
The corresponding "full" quantities that include the clamped block are
$
  M^a = 1 / N sum_i sigma_i^a = (1 - t) + t m^a,
$
$
  hat(Q)^(a b) = 1 / N sum_i sigma_i^a sigma_i^b =
    cases(
      1 - t + t q^(a b) quad & a != b,
      1                 quad & a = b,
    )
$
where $hat(Q)^(a a) = 1$ follows from $sigma_i^2 = 1$. If we adopt the
convention $q^(a a) = 1$ the two cases collapse into the single expression
$hat(Q)^(a b) = (1 - t) + t q^(a b)$ for all $a, b$.

=== Condensed pattern
 
The contribution of pattern 1 to the replicated action is deterministic and
reads
$
  beta / (2 N) sum_a A_1(bold(sigma)^a)^2 = (beta N) / 2 sum_a (M^a)^2.
$
This is the "signal" term: it favours configurations with $M^a$ aligned to
the clamped block, i.e. $m^a approx 1$.

=== Noise patterns (#box($mu >= 2$))

For each noise pattern $mu >= 2$ and replica $a$ decouple $A_mu^a$ by a
Hubbard--Stratonovich transformation,
$
  exp(beta / (2 N) (A_mu^a)^2) =
    integral (d z_mu^a) / sqrt(2 pi) exp(-1/2 (z_mu^a)^2 + sqrt(beta slash N) z_mu^a A_mu^a).
$
Collecting replicas and expanding $A_mu^a$ explicitly, the argument of the
exponential becomes
$
  sqrt(beta slash N) sum_a z_mu^a A_mu^a =
    sqrt(beta slash N) [
      (sum_a z_mu^a) sum_(i in U) tilde(xi)_i^mu +
      sum_(i in F) tilde(xi)_i^mu (sum_a z_mu^a sigma_i^a)
    ].
$
Each $tilde(xi)_i^mu$ is i.i.d. $plus.minus 1$, and averaging factorises over
sites:
$
  overline(exp(c tilde(xi))) = cosh(c) = 1 + c^2 / 2 + O(c^4)
  approx exp(c^2 / 2) quad "for" quad c = O(1 slash sqrt(N)).
$
With $c_i = sqrt(beta slash N) times (dots)_i$ of order $1 slash sqrt(N)$, this
is exact in the large-$N$ limit up to subleading corrections. Collecting
over sites,
$
  overline(exp(sqrt(beta slash N) sum_i tilde(xi)_i^mu (dots)_i))
  approx exp(beta / (2 N) sum_i (dots)_i^2).
$
Using $(1 slash N) sum_(i in F) sigma_i^a sigma_i^b = t q^(a b)$ and
$sum_(i in U) 1 = (1 - t) N$, the squared term expands to
$
  sum_(i in F) (sum_a z_mu^a sigma_i^a)^2 + (1 - t) N (sum_a z_mu^a)^2
  = N sum_(a b) z_mu^a z_mu^b hat(Q)^(a b),
$
the $(1 - t) N$ term from the clamped block contributing the same overlap
correction to every replica pair. The disorder-averaged HS integrand for each
$mu >= 2$ thus reads
$
  integral product_a (d z_mu^a) / sqrt(2 pi)
  exp(-1/2 sum_(a b) (delta_(a b) - beta hat(Q)^(a b)) z_mu^a z_mu^b)
  = det(I - beta hat(Q))^(-1 slash 2).
$
Raising to the power $P = alpha N$ and taking the log gives
$
  product_(mu >= 2) integral dots = exp(-(alpha N) / 2 "Tr" ln(I - beta hat(Q))).
$

=== Free sector trace

After noise averaging, the effective action decouples across free sites up to
the global order parameters $m^a, q^(a b)$. Introducing conjugate fields
$hat(m)^a, hat(q)^(a b)$ to enforce the definitions of $m^a$ and $q^(a b)$
(e.g. via an integral representation of the delta function), the free-site
trace is
$
  Z_"site" =
    sum_({sigma^a})
    exp(sum_a hat(m)^a sigma^a + sum_(a != b) hat(q)^(a b) sigma^a sigma^b),
$
and the action contains the terms
$
  t N sum_a hat(m)^a m^a + t N sum_(a != b) hat(q)^(a b) q^(a b) - t N ln Z_"site".
$
At the saddle point $hat(m)^a$ and $hat(q)^(a b)$ will be fixed by extremisation
in $m^a, q^(a b)$.

=== RS ansatz

In the Replica Symmetric (RS) ansatz we set $m^a = m$, $q^(a b) = q$ for
$a != b$, $hat(m)^a = hat(m)$, $hat(q)^(a b) = hat(q)$. Then
$
  hat(Q)^(a b) = cases(1 quad & a = b, tilde(q) quad & a != b),
  quad tilde(q) := (1 - t) + t q.
$
As an $n times n$ matrix $hat(Q) = (1 - tilde(q)) I + tilde(q) bold(1) bold(1)^T$,
so its eigenvalues are
$
  lambda_+ &= 1 + (n - 1) tilde(q) quad "(multiplicity 1, eigenvector " bold(1) ")",\
  lambda_- &= 1 - tilde(q) quad "(multiplicity" n - 1 ", orthogonal complement)".
$
Hence
$
  "Tr" ln(I - beta hat(Q)) = ln(1 - beta lambda_+) + (n - 1) ln(1 - beta lambda_-).
$
Expanding in small $n$ with $D := 1 - beta(1 - tilde(q))$, so that
$1 - beta lambda_+ = D - n beta tilde(q)$ and $1 - beta lambda_- = D$:
$
  "Tr" ln(I - beta hat(Q))
  &= ln(D - n beta tilde(q)) + (n - 1) ln D \
  &= n ln D - (n beta tilde(q)) / D + O(n^2),
$
so that
$
  lim_(n -> 0) 1/n "Tr"[ln(I - beta hat(Q))] =
  ln(1 - beta(1 - tilde(q))) - (beta tilde(q)) / (1 - beta(1 - tilde(q))).
$
For the free-site trace, one uses the HS identity
$exp(hat(q) sum_(a != b) sigma^a sigma^b) = integral D z exp(sqrt(2 hat(q)) z sum_a sigma^a - n hat(q))$
and performs the sum over ${plus.minus 1}$ at each site, giving
$
  (1 / n) ln Z_"site" arrow.r -hat(q) + integral D z ln[2 cosh(hat(m) + sqrt(2 hat(q)) z)] quad (n -> 0).
$

Extremising in $hat(m), hat(q)$ (and rescaling so that
$hat(m) = beta M$, $2 hat(q) = beta^2 alpha r$; equivalently one introduces
$r$ as an additional order parameter conjugate to the noise-overlap so that
the noise kernel enters the site trace as an effective Gaussian field
$beta sqrt(alpha r) z$), one ends up with the compact RS free energy
$
  -beta f_"RS" =& beta / 2 M^2 - (alpha beta^2 t) / 2 r (1 - q) \
  &+ alpha / 2 [(beta tilde(q)) / (1 - beta(1 - tilde(q))) - ln(1 - beta(1 - tilde(q)))] \
  &+ t integral D z ln[2 cosh(beta(M + sqrt(alpha r) z))],
$ <eq:free-energy>
with $M = (1 - t) + t m$, $tilde(q) = (1 - t) + t q$, and
$D z = e^(-z^2 slash 2) d z slash sqrt(2 pi)$ the standard Gaussian measure.

=== Saddle-point equations

Stationarity of @eq:free-energy with respect to $m$, $q$, $r$ yields
$
  m &= integral D z med tanh(beta(M + sqrt(alpha r) z)), \
  q &= integral D z med tanh^2(beta(M + sqrt(alpha r) z)), \
  r &= tilde(q) / (1 - beta(1 - tilde(q)))^2.
$ <eq:saddle>
The first equation is the self-consistent magnetisation in an effective
field $beta M$ with Gaussian noise of variance $beta^2 alpha r$; it is the
direct generalisation of the Curie--Weiss identity. The second identifies
the free-sector Edwards--Anderson parameter as the second moment of the same
local field. The third encodes the feedback of the free-sector overlap on
the noise variance, promoted by the clamped block ($tilde(q) >= q$ when $t < 1$).

_Limits._ At $t = 1$ we have $M = m$, $tilde(q) = q$, and @eq:saddle reduces to
the classical Amit--Gutfreund--Sompolinsky (AGS) RS equations
@amitSpinGlassModels1985 @amitStatisticalMechanicsNeural1987. At $t = 0$ the
free sector is empty, $M equiv 1$ and $tilde(q) equiv 1$, the integrals collapse,
and the retrieval of pattern 1 is trivially exact.

=== Zero-temperature analysis

The zero-temperature limit is obtained by taking $beta -> oo$ while keeping
$
  C_0 := beta(1 - tilde(q)) = beta t (1 - q)
$
finite. This is the standard scaling for Hopfield-like models: deep in a
retrieval minimum the spins are almost always polarised, so $q -> 1$, but
$1 - q$ vanishes as $1 slash beta$ times a finite factor controlled by the
noise. In this limit $tanh(beta x) -> "sign"(x)$ pointwise, so
$
  m &-> integral D z med "sign"(M + sqrt(alpha r) z)
     = "erf"(M slash sqrt(2 alpha r)), \
  1 - q &-> integral D z med "sech"^2(beta(M + sqrt(alpha r) z)).
$
The $"sech"^2$ integrand concentrates on a thin region of width
$O(1 slash beta)$ around the zero of its argument. Using
$integral "sech"^2(beta x) d x = 2 slash beta$ and approximating
$"sech"^2(beta(M + sqrt(alpha r) z))$ as $(2 slash beta) delta(M + sqrt(alpha r) z)$,
$
  1 - q &approx (2 slash beta) (1 slash sqrt(alpha r)) (D z)_(z = -M slash sqrt(alpha r))
     = (sqrt(2 slash pi)) / (beta sqrt(alpha r)) e^(-M^2 slash (2 alpha r)).
$
Defining $y = M slash sqrt(2 alpha r)$ gives
$
  m = "erf"(y), quad
  C_0 = (t sqrt(2 slash pi) e^(-y^2)) / sqrt(alpha r).
$
From $r = tilde(q) slash D^2 -> 1 slash (1 - C_0)^2$ we get
$sqrt(alpha r) = sqrt(alpha) slash (1 - C_0)$, and substituting in the expression
for $C_0$,
$
  C_0 = u(1 - C_0), quad u := t sqrt(2 slash (pi alpha)) e^(-y^2)
  quad => quad C_0 = u / (1 + u).
$
Finally, using $sqrt(alpha r) = sqrt(alpha)(1 + u)$ and rearranging the
self-consistency $M = (1 - t) + t m = (1 - t) + t "erf"(y)$ as
$y sqrt(2 alpha)(1 + u) = M$, and noting that
$y sqrt(2 alpha) u = t (2 y slash sqrt(pi)) e^(-y^2)$, we arrive at the single
scalar equation
$
  y sqrt(2 alpha) = (1 - t) + t [underbrace("erf"(y) - (2 y) / sqrt(pi) e^(-y^2), g(y))].
$ <eq:T0>
The bias $(1 - t)$ on the right-hand side is the contribution of the clamped
block: it acts as a staggered "external field" on the free spins, whose
strength grows as $t$ decreases.

_Existence of a solution._ Since $g(0) = 0$, $g(oo) = 1$ and
$g'(y) = (4 y^2 slash sqrt(pi)) e^(-y^2) >= 0$, the right-hand side of @eq:T0
ranges over $[1 - t, 1)$ as $y$ ranges over $[0, oo)$. The left-hand side,
$y sqrt(2 alpha)$, is unbounded above. Hence a solution $y_* > 0$ always exists
when $t > 0$: the $y = 0$ (unmagnetised) fixed point of the classical Hopfield
model is destroyed by the bias of the clamped block.

=== Retrieval spinodal

The retrieval branch — the largest root of @eq:T0 — can annihilate with an
intermediate unstable root in a tangent bifurcation. At the bifurcation both
@eq:T0 and its $y$-derivative vanish:
$
  y sqrt(2 alpha_c) = (1 - t) + t g(y), quad sqrt(2 alpha_c) = t g'(y).
$
Eliminating $sqrt(2 alpha_c)$ by substituting the second into $y$ times the
first gives
$
  t [y g'(y) - g(y)] = 1 - t,
$
i.e.
$
  h(y_*) = (1 - t) / t, quad
  h(y) := y g'(y) - g(y) = (2 y) / sqrt(pi) (2 y^2 + 1) e^(-y^2) - "erf"(y).
$ <eq:spinodal>
The function $h$ satisfies $h(0) = 0$, $h(oo) = -1$, and
$h'(y) = y g''(y) = (8 y^2 (1 - y^2) slash sqrt(pi)) e^(-y^2)$, so it is
unimodal with a single maximum at $y = 1$:
$
  h_"max" = h(1) = 6 / (e sqrt(pi)) - "erf"(1) approx 0.4027.
$
Equation @eq:spinodal admits a solution (hence a spinodal bifurcation exists)
iff
$
  (1 - t) / t <= h_"max" quad <==> quad t >= t^* = 1 / (1 + h_"max") approx 0.7129.
$
Given $y^* >= 1$ on the decreasing branch of $h$, the spinodal capacity is
$
  alpha_c (t) = t^2 / 2 [g'(y^*)]^2 = (8 t^2 (y^*)^4) / pi e^(-2 (y^*)^2).
$ <eq:alphac>
At the cusp $y^* = 1$, $alpha_c (t^*) = 8 (t^*)^2 slash (pi e^2) approx 0.1753$,
which is the maximal retrieval capacity reachable by clamping.

=== Phase diagram ($T = 0$)

The $T = 0$ diagram in the $(alpha, t)$ plane has three regions:

#figure(
  table(
    columns: 3,
    align: (center, center, center),
    table.header[$t$][$y^*$][$alpha_c (t)$],
    [$1.00$],             [$1.513$], [$0.1379$],
    [$0.95$],             [$1.47$],  [$0.1417$],
    [$0.85$],             [$1.37$],  [$0.1515$],
    [$t^* approx 0.713$], [$1.00$],  [$0.1753$],
    [$t < t^*$],          [---],     [spinodal absent],
  ),
  caption: [Retrieval spinodal $alpha_c (t)$ at $T = 0$ from @eq:alphac.],
)

- *Retrieval region* ($t >= t^*$ and $alpha < alpha_c (t)$, or $t < t^*$):
  the large-$y$ root of @eq:T0 gives a high-$m$ solution that is locally stable.
  $alpha_c (t)$ is a monotone decreasing function of $t$ on $[t^*, 1]$ going
  from the maximal capacity $approx 0.1753$ at $t^*$ down to the classical
  AGS value $approx 0.138$ at $t = 1$.
- *First-order loss of retrieval* ($t in [t^*, 1]$, $alpha > alpha_c (t)$):
  the retrieval branch has annihilated with the intermediate unstable fixed
  point, leaving only a low-$m$ solution. The jump in $m$ across this line is
  the remnant of the classical AGS first-order transition.
- *No spinodal* ($t < t^*$): @eq:T0 has a unique root at any $alpha > 0$,
  and $m(alpha; t)$ is a smooth, monotone-decreasing function of $alpha$ that
  stays strictly positive. Clamping more than a fraction
  $1 - t^* approx 0.287$ of the spins is enough to suppress the first-order
  retrieval transition altogether: the retrieval attractor persists
  continuously up to arbitrarily large $alpha$.

#figure(
  image("plots/hopfield_T0_phase_diagram.png", width: 90%),
  caption: [Retrieval magnetisation $m$ at $T = 0$ in the $(alpha, t)$ plane,
    from the largest-root solution of @eq:T0. The red dashed curve is the
    spinodal $alpha_c (t)$ from @eq:spinodal. The orange dot marks the cusp
    $(alpha_c^*, t^*) approx (0.175, 0.713)$ at which the spinodal terminates;
    for $t < t^*$ the first-order transition is absent and $m$ varies smoothly.],
) <fig:hopfield-T0-pd>

#figure(
  image("plots/hopfield_T0_m_vs_t.png", width: 90%),
  caption: [Retrieval magnetisation $m$ versus $t$ at $T = 0$ for several
    values of $alpha$. Solid curves show the high-$m$ (retrieval) branch;
    dashed segments of the same colour show the low-$m$ (uninformed) branch,
    visible only where the two roots are distinct. For
    $alpha <= alpha_c (t^*) approx 0.175$ the retrieval branch terminates in a
    first-order jump at some $t_c (alpha)$; for $alpha > alpha_c (t^*)$ the
    jump is absent and $m(t)$ is smooth for all $t in (0, 1]$.
    All curves approach $m = "erf"(1 slash sqrt(2 alpha))$ as $t -> 0$.],
) <fig:hopfield-T0-mvst>

=== MCMC validation at $T = 0.01$

The two figures below compare the $T = 0$ theory curves with MCMC runs at
$beta = 100$ ($T = 0.01$), a temperature low enough that finite-$T$ corrections
are negligible. Each dot is the average magnetisation over 10 independent chains
after 400 sweeps on a system of $N = 20000$ spins.

#figure(
  image("plots/hopfield_T001_m_vs_t_mcmc_pattern.png", width: 90%),
  caption: [MCMC at $T = 0.01$ ($beta = 100$, $N = 20000$, 10 independent chains,
    400 sweeps each), chains initialised on pattern 1. Solid curves show the
    high-$m$ $T = 0$ theory branch; dashed curves of the same colour show the
    low-$m$ branch where it exists. The pattern-started chains track the
    retrieval branch cleanly across all $(alpha, t)$, confirming the spinodal
    boundary $alpha_c (t)$ from @eq:alphac.],
) <fig:hopfield-T001-pattern>

#figure(
  image("plots/hopfield_T001_m_vs_t_mcmc_random.png", width: 90%),
  caption: [MCMC at $T = 0.01$ ($beta = 100$, $N = 20000$, 10 independent chains,
    400 sweeps each), chains initialised uniformly at random. Solid and dashed
    curves as in @fig:hopfield-T001-pattern. Random-started chains converge to
    the low-$m$ branch and track the dashed theory curves well. For
    $t >= t^*$ and $alpha < alpha_c (t)$ they exhibit a sharp jump onto the
    retrieval branch as $t$ decreases, confirming the first-order nature of
    the transition predicted by @eq:spinodal.],
) <fig:hopfield-T001-random>

=== Remarks

- *Clamping as an external field.* The clamped block enters the effective
  local field on free sites through $M = (1 - t) + t m$, i.e. it contributes a
  constant bias $beta (1 - t)$ irrespective of $m$. This single observation
  explains both (i) the upward shift of $alpha_c$ as $t$ decreases (more
  clamping = stronger field = larger retrieval basin), and (ii) the eventual
  disappearance of the first-order transition below $t^*$ (the field becomes
  large enough to keep the retrieval fixed point isolated from any competing
  attractor at all $alpha$).

- *Global vs local transition.* Within RS, $alpha_c (t)$ is the local
  stability (spinodal) boundary of the retrieval branch. A first-order
  thermodynamic transition, defined by equality of free energies between the
  retrieval and spin-glass branches, lies at a slightly smaller $alpha_"th" (t)$.
  In the standard Hopfield model the gap $alpha_c - alpha_"th"$ is already
  very small ($alpha_c approx 0.138$ vs. $alpha_"th" approx 0.137$) and we
  expect the same to hold here.

- *Replica-symmetry breaking.* The de Almeida--Thouless (AT) instability
  @deAlmeidaThouless1978 of RS is controlled by the replicon eigenvalue of
  the free-energy Hessian.
  With the substitution $q -> tilde(q)$ in the noise kernel, the AT condition
  takes the form
  $
    alpha t^2 integral D z med "sech"^4(beta(M + sqrt(alpha r) z)) = (1 - beta(1 - tilde(q)))^2.
  $
  Its $T = 0$ limit should determine a narrow RSB pocket near the spinodal,
  qualitatively similar to the one in the classical Hopfield phase diagram.

#bibliography("bibliography.bib", style: "ieee")
