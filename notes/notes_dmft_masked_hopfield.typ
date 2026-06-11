#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/drafting:0.2.2": inline-note, margin-note
#import "@preview/mannot:0.3.0": markrect

#show figure.caption: set align(left)
#show figure.caption: set text(style: "italic")

#show: arkheion.with(
  title: [DMFT Masked Hopfield],
  authors: (
    (name: "Carlo Lucibello", email: "carlo.lucibello@unibocconi.it", affiliation: [Bocconi University, Milan]),
    (name: "Filippo Elgorni", email: "filippo.elgorni@unibocconi.it", affiliation: [Bocconi University, Milan]),
  ),
  date: datetime.today().display("[day] [month repr:Long] [year]"),
  abstract: [#align(left)[]],
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

= Autoregressive Hopfield Setting

Consider $x_i = plus.minus 1$, $i=1... L$, and
$M$ patterns $xi^mu ~ "Unif"({-1,+1}^L)$. Define
$W_(i j) = 1/sqrt(L) sum_mu xi^mu_i xi^mu_j$.

The probability of a trajectory $x$ in a autoregressive generation is given by
$
  p_W (x) = product_i p_W (x_i | x_(<i)) = product_i sigma(beta thin x_i sum_(j < i) W_(i j)/sqrt(L) x_j)
$
where $beta > 0$ is the inverse temperature controlling the sharpness of the conditional, and $sigma(z) = 1/(1 + e^(-z))$ is the logistic.
The generating functional of Martin-Siggia-Rose is given by
$
  EE_W Z_W (psi) = EE_W sum_x p_W (x) e^(sum_i psi_i x_i)
$
Where the expectation over $W$ is intended as the expectation over the patterns $xi^mu$.
We can write it as

$
  EE_W Z_W (psi) &= EE_W integral product_i (dif h_i dif hat(h)_i)/(2 pi) sum_x e^(sum_i psi_i x_i -i sum_i hat(h)_i h_i + i sum_i hat(h)_i x_i sum_(j < i) W_(i j)/sqrt(L) x_j) product_i sigma(beta h_i)\
  &= EE_(\{xi^mu\}_(mu=1)^(M)) integral product_i (dif h_i dif hat(h)_i)/(2 pi) sum_x e^(sum_i psi_i x_i -i sum_i hat(h)_i h_i + i 1/L sum_(mu, i) hat(h)_i x_i sum_(j < i) xi^mu_i xi^mu_j x_j) product_i sigma(beta h_i)
$ <eq:Z-msr>

== Partial-overlap order parameters

The pattern indices appear inside a triangular sum that is not a perfect square, so the Hubbard--Stratonovich trick of the symmetric Hopfield case (see appendix) cannot be applied directly. Following the MSR/DMFT logic we instead introduce, for each pattern $mu = 1, dots, M$ and each site $i$, the partial overlap

$ m^mu_i := 1/L sum_(j < i) xi^mu_j x_j, $ <eq:partial-overlap>

via a Fourier representation of the delta function:

$
  1 = integral product_(mu, i) (L dif m^mu_i dif hat(m)^mu_i)/(2 pi) thin e^(i L sum_(mu, i) hat(m)^mu_i (m^mu_i - 1/L sum_(j < i) xi^mu_j x_j)).
$

The pattern-dependent piece of the action then reads
$
  i/L sum_(mu, i) hat(h)_i thin x_i thin xi^mu_i sum_(j < i) xi^mu_j x_j = i sum_(mu, i) hat(h)_i thin x_i thin xi^mu_i thin m^mu_i,
$
and rearranging the auxiliary term by swapping the order of summation,
$
  sum_i hat(m)^mu_i sum_(j < i) xi^mu_j x_j = sum_j xi^mu_j x_j hat(M)^mu_j, quad hat(M)^mu_i := sum_(k > i) hat(m)^mu_k.
$ <eq:Mhat>

Each $xi^mu_i in {plus.minus 1}$ now enters linearly, so the disorder average factorises across $(mu, i)$:
$
  EE_(xi^mu_i) thin e^(i xi^mu_i thin x_i (hat(h)_i m^mu_i - hat(M)^mu_i))
  = cos(x_i (hat(h)_i m^mu_i - hat(M)^mu_i))
  = cos(hat(h)_i m^mu_i - hat(M)^mu_i),
$
where the last equality uses $x_i = plus.minus 1$ and the parity of $cos$. Collecting all factors,

$
  EE_W Z_W (psi) &= integral [dif h thin dif hat(h)] [dif m thin dif hat(m)] sum_x e^(sum_i psi_i x_i - i sum_i hat(h)_i h_i + i L sum_(mu, i) hat(m)^mu_i m^mu_i) product_i sigma(beta h_i) product_(mu, i) cos(hat(h)_i m^mu_i - hat(M)^mu_i)\
  &= integral [dif h thin dif hat(h)] [dif m thin dif hat(m)] e^(sum_i log 2 cosh(psi_i) + sum_i log sigma(beta h_i) - i sum_i hat(h)_i h_i + sum_(mu, i)log cos(hat(h)_i m^mu_i - hat(M)^mu_i) + i L sum_(mu, i) hat(m)^mu_i m^mu_i)\
$ <eq:Z-orderparams>

with shorthand $[dif h thin dif hat(h)] = product_i (dif h_i dif hat(h)_i)/(2 pi)$ and $[dif m thin dif hat(m)] = product_(mu,i) (L dif m^mu_i dif hat(m)^mu_i)/(2 pi)$.

== Warmup: paramagnetic regime

We now make the *DMFT closure*: assume that, at the relevant saddle, the cosine arguments
$ theta_i^mu := hat(h)_i thin m^mu_i - hat(M)^mu_i $
are small, so that

$ log cos(theta_i^mu) = -1/2 (theta_i^mu)^2 + sum_(k >= 2) kappa_(2k) (theta_i^mu) / (2k)!, $

and the higher cumulants $kappa_(2k)$ are neglected. This is a *closure assumption*, not an exact statement: while the second-cumulant calculation below is exact for finite-dimensional marginals of the disorder average, the full $L$-site generating functional generally retains non-Gaussian contributions that the present treatment discards. Note moreover that the bound $m^mu_i = O(1/sqrt(L))$ alone does not control $hat(M)^mu_i = sum_(k > i) hat(m)^mu_k$, which is a sum of $L - i$ small quantities; the small-cosine expansion is therefore an assumption on $theta_i^mu$ itself, to be verified a posteriori at the saddle.

Under this closure,

$
  log product_(mu, i) cos(theta_i^mu (m_i^mu, hat(m)^mu_i))) approx -1/2 sum_(mu, i) [hat(h)_i^2 (m^mu_i)^2 - 2 hat(h)_i m^mu_i hat(M)^mu_i + (hat(M)^mu_i)^2].
$ <eq:quadratic-action>

The pattern indices factorise; for each $mu$ the Gaussian integrals over $m^mu_i, hat(m)^mu_i$ can be done explicitly.
$
  &integral (dif m_i^mu)/sqrt(2pi) e^(-1/2 [hat(h)_i^2 (m^mu_i)^2 - 2 hat(h)_i m^mu_i hat(M)^mu_i + (hat(M)^mu_i)^2] + i L hat(m)^mu_i m^mu_i)=\
  &=e^(-1/2 (hat(M)^mu_i)^2) integral (dif m_i^mu)/sqrt(2pi) e^(-1/2 hat(h)_i^2 (m^mu_i)^2 + m^mu_i (hat(h)_i hat(M)^mu_i + i L hat(m)^mu_i))=\
  &= 1/hat(h)_i e^(-1/2 [(hat(M)^mu_i)^2-(hat(h)_i hat(M)^mu_i + i L hat(m)^mu_i)^2/(hat(h)_i^2)])\
  &= 1/hat(h)_i e^(1/2 [
    ( i 2 hat(h)_i hat(M)^mu_i L hat(m)^mu_i -L^2 (hat(m)^mu_i)^2)/(hat(h)_i^2)])\
$

so the next integral is:

$
  & integral product_i (dif hat(m)^mu_i)/sqrt(2pi) e^(1/2 sum_i [
    ( i 2 hat(h)_i L hat(m)^mu_i sum_(k > i) hat(m)^mu_k -L^2 (hat(m)^mu_i)^2)/(hat(h)_i^2)]) =\
  & integral product_i (dif hat(m)^mu_i)/sqrt(2pi) e^(i L/hat(h)_i hat(m)_i^mu sum_(i, k > i) hat(m)^mu_k -1/2 sum_i (L/hat(h)_i)^2 (hat(m)^mu_i)^2) =\
  & integral product_i (dif hat(m)^mu_i)/sqrt(2pi) e^(-1/2 sum_(i j) hat(m)_i^mu A_(i j) hat(m)_j^mu )=1 /(sqrt(det A)) \
$

where the symmetric matrix $A$ has entries $A_(i i) = (L/hat(h)_i)^2$  and $A_(i j) = - i L/hat(h)_min(i, j)$:
$
  A=mat(
    L^2 / hat(h)_1^2, - i L/hat(h)_1, - i L/hat(h)_1, dots, - i L/hat(h)_1;
    - i L/hat(h)_1, L^2 / hat(h)_2^2, - i L/hat(h)_2, dots, - i L/hat(h)_2;
    - i L/hat(h)_1, - i L/hat(h)_2, L^2 / hat(h)_3^2, dots, - i L/hat(h)_3;
    dots.v, dots.v, dots.v, dots.down, dots.v;
  ).
$
We can rewrite $A=D(I+K)D$ where $D$ is a diagonal matrix whose entries are $D_(i i) = L/hat(h)_i$, and $K$ is a matrix with entries $K_(i j) = - i hat(h)_max(i, j) / L$ and $0$ on the diagonal. Then $det A = det D^2 det(I + K) = L^(2 L) / (product_i hat(h)_i^2) det(I + K)$. Indeed $A_(i j) = D_(i i)(delta_(i j) + K_(i j))D_(j j)$: for $i=j$ we get $A_(i i) = D_(i i)^2 = L^2/hat(h)_i^2$, while for $i != j$ we get $A_(i j) = D_(i i) K_(i j) D_(j j) =L/hat(h)_i (- i hat(h)_max(i, j) / L)L/hat(h)_j=-i L/hat(h)_min(i, j)$ as required.

Putting everything together:

$
  &integral [dif m dif hat(m)] e^(sum_(mu, i)log cos(hat(h)_i m^mu_i - hat(M)^mu_i) + i L sum_(mu, i) hat(m)^mu_i m^mu_i)=product_(mu) 1/( sqrt(det (I+K)))= e^(-M/2 log det (I+K)) \
$

Notice that elements of $K$ are $O(1/L)$ but they can still contribute to the determinant at leading order because $K$ is dense. We expand $log det (I+K) = Tr K - 1/2 Tr K^2 + 1/3 Tr K^3 + ...$. The first term is $Tr K = sum_i K_(i i) = 0$ since $K$ has zero diagonal. For the second term, we compute $Tr K^2 = sum_(i, k) K_(i k) K_(k i) = -1/L^2 sum_i sum_(k != i) hat(h)_max(i, k)^2 = -1/L^2[ sum_i sum_(k < i) hat(h)_max(i, k)^2+ sum_i sum_(k > i) hat(h)_max(i, k)^2] = -1/L^2[ sum_i sum_(k < i) hat(h)_i^2+ sum_i sum_(k > i) hat(h)_k^2]= -1/L^2[ sum_i sum_(k < i) hat(h)_i^2+ sum_i sum_(k > i) hat(h)_k^2] = -2/L^2 sum_i (i-1)hat(h)_i^2$ hence:


$
  EE_W Z_W (psi) &= integral [dif h thin dif hat(h)] e^(sum_i log 2 cosh(psi_i) + sum_i log sigma(beta h_i) - i sum_i hat(h)_i h_i-alpha/2 sum_i (i-1)/L hat(h)_i^2) \
$

We can factorise $i$ and integrate $hat(h)_i$:

$
  EE_W Z_W (psi) &= product_i integral (dif h_i)/sqrt(2pi) thin 1/sqrt(alpha (i-1)/L) e^(log 2 cosh(psi_i) + log sigma(beta h_i) - 1/2(h_i^2)/(alpha (i-1)/L)) \
  &= product_i integral D h thin 2cosh(psi_i) sigma(beta sqrt(alpha thin ((i-1)\/L)) thin h_i) \
  &= integral cal(D) h thin exp(L integral_0^1 dif t log 2 cosh(psi(t)) +log sigma(beta sqrt(alpha thin t) thin h(t)))
$<eq:Gaussian-decoupling>

Where in the last step we made explicit the continuous limit where we introduced the time $t=i\/L$.

However if we were to include all therms, we would have to compute the determinant of $I + K$ exactly, which is not trivial. The recursive structure of $A$ makes it however doable: define $c_i = -i hat(h)_i / L$. Then:

$
  (I+K)_L = mat(
    (I+K)_(L-1), c_L bold(1);
    c_L bold(1)^top, 1
  ).
$

We can use Shur's formula:

$
  det(I+K)_L & =det(I+K)_(L-1) (1 -c_L^2 bold(1)^top (I+K)_(L-1)^(-1)bold(1)).
$

Defining $q_L = bold(1)^top (I+K)_(L)^(-1)bold(1)$  one gets that $det(I+K)_L = product_(i=2)^L (1 - c_i^2 q_(i-1))$. So once we have all $L$ therms $q_L$, we have an expression for the determinant.This can be done again recursively: we can compute $(I+K)_L^(-1)$ via inversion of a block matrix:

$
  (I+K)_L^(-1) = mat(
    (I+K)^(-1)_(L-1) + (c_L^2 (I+K)^(-1)_(L-1) bold(1)bold(1)^top (I+K)^(-1)_(L-1))/(1-c_L^2 q_(L-1)), -(c_L (I+K)^(-1)_(L-1) bold(1))/(1-c_L^2 q_(L-1));
    -(c_L bold(1)^top (I+K)^(-1)_(L-1))/(1-c_L^2 q_(L-1)), 1/(1-c_L^2 q_(L-1))
  )
$


and

$
  q_L =bold(1)^top (I+K)_(L)^(-1)bold(1) &= bold(1)^top mat(
    (I+K)^(-1)_(L-1) bold(1) + (c_L^2 (I+K)^(-1)_(L-1) bold(1)bold(1)^top (I+K)^(-1)_(L-1)bold(1))/(1-c_L^2 q_(L-1)) -(c_L (I+K)^(-1)_(L-1) bold(1) )/(1-c_L^2 q_(L-1));
    -(c_L bold(1)^top (I+K)^(-1)_(L-1) bold(1))/(1-c_L^2 q_(L-1))+ bold(1)/(1-c_L^2 q_(L-1))
  )\
  &= q_(L-1) + (c_L^2 q_(L-1)^2)/(1-c_L^2 q_(L-1)) - (c_L q_(L-1))/(1-c_L^2 q_(L-1)) - (c_L q_(L-1))/(1-c_L^2 q_(L-1)) + 1/(1-c_L^2 q_(L-1))\
  &= q_(L-1) + (1- c_L q_(L-1))^2/(1-c_L^2 q_(L-1)).
$
Substituting the definition of $c_i$ we get:

$
  log det (I+K) = sum_(i=2)^L log (1 + hat(h)_i^2 / L^2 q_(i-1))
$

with of course $q_1 = 1$ and

$
  q_L - q_(L-1) = (1+ i/L hat(h)_L q_(L-1))^2/(1+ hat(h)_L^2/L^2 q_(L-1)).
$


We now go to the continuous limit introducing the time $t=i\/L$. It must be that $q_L$ must scale like $L$: indeed, putting $K=0$, we have $q_L =bold(1)^top I bold(1)= sum_i i = L$. Parametrise $q_L = L r(t)$, then the recursion becomes:

$
  L (r(t + 1\/L) - r(t)) = (1+ i hat(h)(t) r(t))^2/(1+ (hat(h)(t)^2)/L r(t))
$

the second therm in the denominator vanishes, and we recognise the right hand side as a derivative:

$
  (d r) / (d t) = (1+ i hat(h)(t) r(t))^2
$

with initial condition $r(0) = 0$. In the continuous limit, the determinant becomes

$
  log det (I+K) = integral_0^1 dif t thin hat(h)(t)^2 r(t).
$

The full expression for the generating functional in the continuous limit is then

$
  EE_W Z_W (psi) &= integral product_i (dif h_i thin dif hat(h)_i)/(2pi) e^(sum_i log 2 cosh(psi_i) + sum_i log sigma(beta h_i) - i sum_i hat(h)_i h_i-M/2 log det (I+K)) \
  &= integral cal(D)h cal(D)hat(h) thin e^(L S[h, hat(h)])
$

with $S[h, hat(h)] = integral_0^1 dif t thin s(t)$ and

$
  s = log 2 cosh(psi) + log sigma(beta h) - i hat(h) h-alpha/2 thin hat(h)^2 r + lambda(dot(r) - (1+ i hat(h) r)^2)
$

where we enforced the definition of $r(t)$ via a Lagrange multiplier $lambda(t)$ in the action.

However, both the gaussian closure and the exact dynamical eqations are trivially  useless in absence of a sigal to be retrieved. We therefore need to look for solutions with a macroscopic overlap with one of the patterns, which will be the retrieved one. This is the subject of the next section.

== Condensed regime

To capture pattern retrieval we single out one pattern, say $xi^1$, and look for solutions with macroscopic overlap

$ m_i := m^(mu=1)_i = 1/L sum_(j < i) xi^(mu=1)_j x_j = O(1), $ <eq:cond-overlap>

while the remaining $M - 1$ patterns stay in the paramagnetic regime $m^mu_i = O(1/sqrt(L))$ for $mu > 1$. Here we don't have to average on $xi^1$ since the retrieval performance will be read off by conditioning on a specific value of $xi^1$. For this reason, we introduce the spin variablee $s_i=x_i xi_i^(mu=1)$.

$
  EE_W Z_W (psi|xi^(mu=1)) &= EE_(\{xi^mu\}_(mu>1)^(M)) integral [dif h dif hat(h)] sum_x e^(sum_i psi_i x_i -i sum_i hat(h)_i h_i + i 1/L sum_(i) hat(h)_i x_i sum_(j < i) xi^(mu=1)_i xi^(mu=1)_j x_j) times\
  &times e^(i 1/L sum_(mu>1, i) hat(h)_i x_i sum_(j < i) xi^mu_i xi^mu_j x_j) product_i sigma(beta h_i)=\
  &= EE_(\{xi^mu\}_(mu>1)^(M)) integral [dif h dif hat(h)] sum_s e^(sum_i psi_i xi^(mu=1)_i s_i -i sum_i hat(h)_i h_i + i 1/L sum_(i) hat(h)_i s_i sum_(j < i) s_j) times\
  &times e^(i 1/L sum_(mu>1, i) hat(h)_i s_i xi_i^(mu=1)xi^mu_i sum_(j < i) xi^mu_j xi^(mu=1)_j s_j) product_i sigma(beta h_i)
$

One can define new patterns $tilde(xi)^mu_i := xi^(mu=1)_i xi^mu_i$ which are again iid Rademacker $plus.minus 1$, so the disorder average factorises across patterns as before and wee get the same result (with a vanishing difference due to $M-1$). Notice that it goes out of the su over $\{s\}$ due to the cosine trick of before. We can also harmlessly absorb $xi^(mu=1)$ into $psi$ as it is always taken in its 0 limit when used. For the macroscopic magnetiation therm, we enforce $m_(i+1) = m_i + 1/L s_i$:

$
  EE_W Z_W (psi|xi^(mu=1))
  &= integral [dif h dif hat(h)] product_i (L dif m_i dif hat(m)_i)/(2pi) sum_s e^(sum_i psi_i s_i + i sum_(i) hat(h)_i s_i m_i - i sum_i hat(m)_i s_i) times\
  &times e^(i L sum_i (m_(i+1)-m_i)-i sum_i hat(h)_i h_i -(M-1)/2 log det (I+K[hat(h)])+sum_i log sigma(beta h_i)) =\
  &= integral [dif h dif hat(h)] product_i (L dif m_i dif hat(m)_i)/(2pi) e^(sum_i log 2 cosh (psi_i + i hat(h)_i m_i - i hat(m)_i)) times\
  &times e^(i L sum_i (m_(i+1)-m_i)-i sum_i hat(h)_i h_i -(M-1)/2 log det (I+K[hat(h)])+sum_i log sigma(beta h_i)).
$

In the continuous limit this is:

$
  EE_W Z_W (psi|xi^(mu=1))
  &= integral cal(D) h cal(D) hat(h) cal(D) m cal(D) hat(m) cal(D) lambda cal(D) r thin e^(L S[h, hat(h), m, hat(m), lambda, r])
$

With the action $S=integral_0^1 dif t s(t)$ and

$
  s = & log sigma(beta h) + log 2 cosh (psi + i hat(h) m-i hat(m))+ \
      & + i hat(m) dot(m)- i hat(h) h -alpha/2 hat(h)^2 r + lambda (dot(r) - (1+ i hat(h) r)^2).
$

It would be tempting to do a functional saddle point on the whole functional, but one must not forget that the field $h$ is a fluctuating quantity even at $L->infinity$: as we saw in the pedagogical paramagnetic closure it was gaussian with variance $alpha t = O(1)$. We must therefore integrate the $h$ and $hat(h)$ fields first.

$
  s = & log sigma(beta h) + log 2 cosh (psi + i hat(h) m-i hat(m))+ \
      & + i hat(m) dot(m)- i hat(h) h -alpha/2 hat(h)^2 r + lambda (dot(r)-1) +lambda hat(h)^2 r^2 - 2i lambda hat(h) r= \
      & = -1/2 (alpha r - 2 lambda r^2) hat(h)^2 - (i h +2i lambda r) hat(h) + log 2 cosh (psi + i hat(h) m-i hat(m))+ \
      & + i hat(m) dot(m) + lambda (dot(r)-1) + log sigma(beta h). \
$

Due to the exponential nature of the integral, the gaussian integrals can be performed at fixed time $t$ and then put back together in the continuous limit. The first one is

$
  &integral (dif hat(h))/sqrt(2pi) thin e^(-1/2 (alpha r - 2 lambda r^2) hat(h)^2 - (2i lambda r + i h) hat(h) + log 2 cosh (psi + i hat(h) m-i hat(m)))=\
  & = integral (dif hat(h))/sqrt(2pi) thin e^(-1/2 (alpha r - 2 lambda r^2) hat(h)^2 -(2i lambda r+i h ) hat(h)) [e^(psi + i hat(h) m-i hat(m))+e^(-(psi + i hat(h) m-i hat(m)))]\
$


We have a sum of the two integrals of the form
$
  & integral (dif hat(h))/sqrt(2pi) thin e^(-1/2 (alpha r - 2 lambda r^2) hat(h)^2 +(2i lambda r-i h ) hat(h)) [e^(plus.minus (psi + i hat(h) m-i hat(m)))]=\
  &=1/sqrt(alpha r - 2 lambda r^2) e^( thin plus.minus (psi - i hat(m)) + (2 lambda r + h plus.minus m)^2/(2 (alpha r - 2 lambda r^2))).\
$

Then:

$
  & 1/sqrt(alpha r - 2 lambda r^2)integral (dif h)/sqrt(2pi) thin sigma(beta h) e^( (2 lambda r + h plus.minus m)^2/(2 (alpha r - 2 lambda r^2))) = \
  &= integral D z sigma(beta (plus.minus m + 2 lambda r + sqrt(alpha r - 2 lambda r^2) z))
$

After doing the last integral, we put everything back in the continuous limit and obtain the generating functional action (from now on we put $psi=0$ because I am pretty sure it is useless)

$
  s &= i hat(m) dot(m) + lambda (dot(r)-1) + log sum_(p=plus.minus 1) e^(- i p hat(m)) integral D z sigma(beta(p m + 2 lambda r + sqrt(alpha r - 2 lambda r^2) z)) \
  &=i hat(m) dot(m) + lambda (dot(r)-1) + log [ G_+(m, hat(m), r, lambda)+ G_-(m, hat(m), r, lambda)]. \
$

The symmetry of $sigma$ lets Euler Lagrange equations for $hat(m)$ and $lambda$ give the symmetric $hat(m)=0$ and $lambda=0$ as solutions. Indeed, with $G_plus.minus$ defined as above, one has:

$
          G_+(m, hat(m)=0, r, lambda=0) & = 1-G_-(m, hat(m)=0, r, lambda=0) \
  partial_m G(m, hat(m)=0, r, lambda=0) & = -partial_m G(m, hat(m)=0, r, lambda=0) \
  partial_r G(m, hat(m)=0, r, lambda=0) & = -partial_r G(m, hat(m)=0, r, lambda=0) \
$

Then the Euler-Lagrange equations on $hat(m)$ and $dot(lambda)$ give:

$
  i dot(hat(m)) = (partial_m G_+(m, hat(m), r, lambda)+ partial_m G_-(m, hat(m), r, lambda)) / (G_+(m, hat(m), r, lambda)+ G_-(m, hat(m), r, lambda)) = 0
$

and

$
  dot(lambda) = (partial_r G_+(m, hat(m), r, lambda)+ partial_r G_-(m, hat(m), r, lambda)) / (G_+(m, hat(m), r, lambda)+ G_-(m, hat(m), r, lambda)) = 0
$

which are satisfied by $hat(m)=0$ and $lambda=0$. Instead, differentiating with respect to $hat(m)$ and inserting the symmetric soltion we get the familiar

$
  dot(m) = 1-2 integral D z sigma(beta(m + sqrt(alpha r) z))
$

We just have to find the equation for $r$: varying with respect to $lambda$ we get:

$
  dot(r) = 1 - (partial_lambda G_+ (m, hat(m), r, lambda) + partial_lambda G_- (m, hat(m), r, lambda))/(G_+ (m, hat(m), r, lambda) + G_- (m, hat(m), r, lambda))
$

since

$
  partial_lambda G_(plus.minus) (m, hat(m), r, lambda) =& beta e^(minus.plus i hat(m)) integral D z sigma'(beta(plus.minus m + 2 lambda r + sqrt(alpha r - 2 lambda r^2) z)) [2r - r^2/sqrt(alpha r-2 lambda r^2)z]\
$

we have

$
  &partial_lambda G_+ (m, hat(m)=0, r, lambda=0)+partial_lambda G_- (m, hat(m)=0, r, lambda=0)= \
  &=beta integral D z [sigma'(beta(m + sqrt(alpha r) z))+sigma'(beta(- m + sqrt(alpha r) z))] [2r cancel(- r^2/sqrt(alpha r)z)]\
  &=4 beta r integral D z sigma'(beta(m + sqrt(alpha r) z))
$
where we used the symmetry of $sigma$ to get rid of the second term and double the first one. The complete equations are then:

$
  dot(m) & = 2 integral D z sigma(beta(m + sqrt(v) z))-1 \
  dot(v) & = alpha -4 beta v integral D z sigma'(beta(m + sqrt(v) z)).
$

Where we defined $v=alpha r$. We discover that the equation of our previous work is obtained by neglecting this last susceptivity term $4 beta v integral D z sigma'(beta(m + sqrt(v) z))$.
Specialising for the to algorithms "fair" ($beta = 1$) and "greedy" ($beta = infinity$) we get:

$
  dot(m) & = 2 integral D z sigma(m + sqrt(v) z) -1 \
  dot(v) & = alpha -4 v integral D z sigma'(m + sqrt(v) z)
$

and

$
  dot(m) & = 2 Phi(m /sqrt(v)) -1 \
  dot(v) & = alpha - 4 sqrt(v) phi(m /sqrt(v)).
$



= Autoregressive network with learned weights

Suppose now the weights have been obtained by minimising the following loss:

$
  cal(L)_(i) (\{w_(i j)\}_(j=1)^L, Xi) &= - 1/M sum_(mu=1)^M EE_(t) 1/t EE_(m^t) thin II(m^(t)_i=1) log sigma(1 / sqrt(L) xi^mu_i sum_(j) (w_(j) (1-m^t_j)xi_(j)^(mu))).
$

We call the data $Xi = {xi^mu}_(mu=1)^M$ and the MSR generating functional is then

$
  EE_W Z_W (psi) =& EE_W integral [dif h dif hat(h)] sum_x e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i x_i sum_(j < i) W_(i j)/sqrt(L) x_j + sum_i log sigma(h_i)) \
  =& EE_Xi P(W | Xi) Z_W(psi)\
  =& EE_Xi 1/(Z_Xi (beta)) integral product_(i j) dif w_(i j) thin e^(-beta M sum_i cal(L)_i (\{w_(i j)\}_(j=1)^L, Xi) - 1/2 beta sum_(i j) lambda ||w_(i j)||^2 ) times \
  &times integral [dif h dif hat(h)] sum_x e^( -i sum_i hat(h)_i h_i + i sum_i hat(h)_i x_i sum_(j < i) w_(i j)/sqrt(L) x_j + sum_i log sigma(h_i))
$

with
$
  Z_Xi (beta)= EE_Xi integral product_(i j) dif w_(i j) thin e^(-beta M sum_i cal(L)_i (\{w_(i j)\}_(j=1)^L, Xi) - 1/2 beta sum_(i j) lambda ||w_(i j)||^2 ).
$

We use $Z_Xi(beta)^(-1) = lim_(n->0) Z_Xi(beta)^(n-1)$ to write:
$
  EE_W Z_W (psi)
  =& lim_(n->0)EE_Xi integral product_a product_(i j) dif w_(i j)^a thin e^(-beta M sum_i cal(L)_i (\{w_(i j)^a\}_(j=1)^L, Xi) - 1/2 beta sum_(i j) lambda ||w^a_(i j)||^2 ) times \
  &times integral [dif h dif hat(h)] sum_x e^( -i sum_i hat(h)_i h_i + i sum_i hat(h)_i x_i sum_(j < i) w_(i j)^(a=1)/sqrt(L) x_j + sum_i log sigma(h_i))\
  =& sum_x integral [dif h dif hat(h)] e^(-i sum_i hat(h)_i h_i + sum_i log sigma(h_i)) times \
  &times lim_(n->0)EE_Xi integral product_a product_(i j) dif w_(i j)^a thin e^(-beta M sum_i cal(L)_i (\{w_(i j)^a\}_(j=1)^L, Xi) - 1/2 beta sum_(i j) lambda (w^a_(i j))^2 + i sum_i hat(h)_i x_i sum_(j < i) w_(i j)^(a=1)/sqrt(L) x_j)\
  =& sum_x integral [dif h dif hat(h)] e^(-i sum_i hat(h)_i h_i + sum_i log sigma(h_i)) times \
  &times lim_(n->0) EE_Xi integral product_a product_(i j) dif w_(i j)^a thin e^(-beta M sum_i cal(L)_i (\{w_(i j)^a\}_(j=1)^L, Xi) - 1/2 beta sum_(i j) lambda (w^a_(i j))^2 + sum_(i j) w_(i j)^(a=1) u_(i j))\
$

where $u_(i j) = i 1/sqrt(L) hat(h)_i x_i x_j II_(j<i)$.
// So far we have imposed only diffusion to be autoregressive. If also learning happens autoregressively:

// $
//   cal(L)_(i) (\{w_(i j)\}_(j=1)^L, Xi) &= - 1/M sum_(mu=1)^M log sigma(1 / sqrt(L) xi^mu_i sum_(j<i) (w_(i j) xi_(j)^(mu))).
// $

// so we are interested in the exponent:

// $
//   &-beta M sum_i cal(L)_i (\{w_(i j)^a\}_(j=1)^L, Xi) - 1/2 beta sum_(i j) lambda (w^a_(i j))^2 + sum_(i j) w_(i j)^(a=1) u_(i j)=\
//   & = beta sum_i sum_mu log sigma(1 / sqrt(L) sum_(j) w^a_(i j) m_(i j)^mu) - 1/2 beta sum_(i j) lambda (w^a_(i j))^2 + sum_(i j) w_(i j)^(a=1) u_(i j)
// $

// where now $m_(i j)^mu = xi_i^(mu) xi_j^(mu) II_(j<i)$.
Using the same CLT arguments as befor we have that average we are trying to perform is
$
  &EE_{mu_i^a}_(a=1)^M integral product_(a i j) dif w_(i j)^a thin e^(beta sum_(a i mu) integral_0^1 dif t EE_(z_a) log sigma(xi_i^mu z_a) - 1/2 beta sum_(i j) lambda (w^a_(i j))^2 + sum_(i j) w_(i j)^(a=1) u_(i j))\
$

where here the difference with before we must define the overlaps
$
  q^(a b)_(i j)= 1/L sum_k w_(i k)^a w_(j k)^b
$

== Orthogonal Gaussian Ansatz

$
  EE_W Z_W (psi) =& sum_x integral [dif h dif hat(h)] e^(-i sum_i hat(h)_i h_i + sum_i log sigma(h_i)) EE_Xi product_i lr(chevron.l e^(sum_j i/sqrt(L) hat(h)_i x_i x_j bold(1)_(j < i) dot W_(i k))chevron.r)\
$

can be expanded in cumulants, keep only the first

A first simple approximation is to use our knowledge of $W$ from the replica computation.
Assue $W_(i j)$ elements are Gaussian with row variance $1/L sum_j W_(i j)^2 = q$. Then we select a pattern $xi^(mu=1)$ and define $s_i=xi_i^(mu=1) x_i$:

$
  EE_W Z_W (psi|xi^(mu=1)) =& EE_W integral [dif h dif hat(h)] sum_x e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i x_i sum_(j < i) W_(i j)/sqrt(L) x_j + sum_i log sigma(h_i)) \
  =&EE_W integral [dif h dif hat(h)] sum_s e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i s_i sum_(j < i) J_(i j)/sqrt(L) s_j + sum_i log sigma(h_i)) \
$

where we defined $J_(i j) =xi_i^(mu=1) W_(i j) xi_j^(mu=1)$. The important observation is that $J_(i j)$ are again Gaussian with row variance $q$. However, from the replica computationwe now know the row mean $1/sqrt(L) sum_j J_(i j)$ is a random variable that converges in distribution to $mu(x)$ with $x~cal(N)(0,1)$. For each row $i$ we will have some different $mu_i$ drawn from this distribution, we thus write: $J_(i j) = mu_i/sqrt(L) + sqrt(q) z_(i j)$, where the $z_(i j)$ are gaussian with the extra condition that $sum_j z_(i j)=0$. Notice that this enforces the guassian vectors to live in the $L-1$ space orthogonal to $bold(1)$, and the covariance matrix of each row of $z_i = (z_(i 1), z_(i, 2), dots)$ is $Sigma_i = II - 1/L bold(1)bold(1)^top$.

$
  EE_W Z_W (psi|xi^(mu=1))
  =&EE_W integral [dif h dif hat(h)] sum_s e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i s_i sum_(j < i) J_(i j)/sqrt(L) s_j + sum_i log sigma(h_i)) \
  =&EE_(mu, z) integral [dif h dif hat(h)] sum_s e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i s_i sum_(j < i) (mu_i/sqrt(L) + sqrt(q) z_(i j))/sqrt(L) s_j + sum_i log sigma(h_i)) \
  =&EE_(mu, z) integral [dif h dif hat(h)] sum_s e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i s_i mu_i sum_(j < i) s_j/L + i sum_i hat(h)_i s_i sqrt(q) sum_(j < i) z_(i j)/sqrt(L) s_j + sum_i log sigma(h_i)) \
  =&EE_(mu, z) integral [dif h dif hat(h)] sum_s e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i s_i mu_i m_i + i sum_i hat(h)_i s_i sqrt(q) sum_(j < i) z_(i j)/sqrt(L) s_j + sum_i log sigma(h_i)) \
$

We do the first Gaussian integral on the $z_(i j)$ factorizing by row $i$:
$
  & integral product_j D z_(i j) thin e^(i sum_i hat(h)_i s_i sqrt(q) sum_(j < i) z_(i j)/sqrt(L) s_j) = e^(1/2 a^top Sigma_i a)
$
were $a_j = i hat(h)_i sqrt(q/L) s_i s_j bold(1)_(j<i)$. We have

$
  a^top Sigma_i a = a^top a - 1/L (a^top bold(1))^2 = - q hat(h)_i^2 [(i-1)/L- (1/L sum_(j < i) s_j)^2]
$

We now enforce $m_(i+1) = m_i + 1/L s_i$: notice that it also imposes $1/L sum_(j<i) s_j=m_i$ recursively.

$
  EE_W Z_W &(psi|xi^(mu=1))=\
  =&EE_mu integral [dif m dif hat(m)] [dif h dif hat(h)] sum_s e^(-i sum_i hat(h)_i h_i + i sum_i hat(h)_i s_i mu_i m_i -1/2 q sum_i [(i-1)/L-m_i^2] hat(h)_i^2 + sum_i log sigma(h_i) + i L sum_i hat(m)_i (m_(i+1) - m_i - 1/L s_i)) \
  =&EE_mu integral [dif m dif hat(m)] [dif h dif hat(h)] sum_s product_i e^(-i hat(h)_i h_i + i hat(h)_i s_i mu_i m_i -1/2 q [(i-1)/L-m_i^2] hat(h)_i^2 + log sigma(h_i) + i L hat(m)_i (m_(i+1) - m_i - 1/L s_i)) \
  =&EE_mu integral [dif m dif hat(m)] [dif h dif hat(h)] product_i e^(-i hat(h)_i h_i -1/2 q [(i-1)/L-m_i^2] hat(h)_i^2 + log sigma(h_i) + i L hat(m)_i (m_(i+1) - m_i )) sum_(s_i=plus.minus 1) e^( i hat(h)_i s_i mu_i m_i- i hat(m)_i s_i ) \
  =&EE_mu integral [dif m dif hat(m)] [dif h dif hat(h)] product_i e^(-i hat(h)_i h_i -1/2 q [(i-1)/L-m_i^2] hat(h)_i^2 + log sigma(h_i) + i L hat(m)_i (m_(i+1) - m_i ) + log 2 cosh i (hat(h)_i mu_i m_i - hat(m)_i))\
$

performing the integrals in $h$ and $hat(h)$ as before we get, in the continuous limit:
$
  EE_W Z_W & (psi|xi^(mu=1))=EE_mu integral cal(D) m cal(D) hat(m) thin e^(L S[m, hat(m)])
$

where $s$ such that $S[m, hat(m)] = integral_0^1 dif t thin s(t)$ is

$
  s = i hat(m) dot(m) + log sum_(p=plus.minus 1) e^(- i p hat(m)) integral D z D x thin sigma(beta(p mu(x) m + sqrt(q (t-m^2)) z))
$

Clearly the fixed points solution is, as we found in the other metod:

$
  dot(m) = 2 integral D z D x thin sigma(beta(mu(x) m + sqrt(q (t-m^2)) z))-1.
$
== Susceptibility Term Ansatz

$
  dot(m) & = 2 integral D z sigma(beta(m + sqrt(v) z))-1 \
  dot(v) & = alpha -4 beta v integral D z sigma'(beta(m + sqrt(v) z)).
$

if $dot(m)=(d m)/(d t)$, if we interpret



== Hebbianity of learned weights

We notice that this description completely misses the susceptibility therm of the Hebbian case. It is then natural to start look at to what order the learned matrix $W$ is close to the Hebbian one.

Define $H_(i j) = 1/sqrt(L) sum_mu xi^mu_i xi^mu_j$. We look at the rows $H_i = (H_(i 1), ..., H_(i L))$ and $W_i = (W_(i 1), ..., W_(i L))$ as vectors in $L$-dimensional space, and assume each row is independent from each other. We can then decompose:
$
  W_i = (W_i dot H_i) /( ||H_i||^2) H_i + R_i.
$

We already know $1/L||H_i||^2 = 1/L^2 sum_j sum_(mu, nu) xi^mu_i xi^nu_j xi^nu_i xi^mu_j = alpha$. Then the projection of $W_i$ on $H_i$ is
$
  1/L W_i dot H_i = 1/L sum_j W_(i j) (sum_mu 1/sqrt(L) xi^mu_i xi^mu_j) = alpha /M sum_mu (1/sqrt(L) sum_j W_(i j) xi^mu_i xi^mu_j)
$

By the usual properties of the row mean, each element $(1/sqrt(L) sum_j W_(i j) xi^mu_i xi^mu_j)$ of the sum over the pattens is again some $mu ~ mu(x)$ with $x~cal(N)(0,1)$. The empirical average over all infinite patterns makes this therm converge to $EE mu(x)$. Therfore, the projection of $W_i$ on $H_i$ is equal for all rows and it is
$
  overline(mu) = (W_i dot H_i) /( ||H_i||^2) = (alpha EE mu(x) ) / alpha = integral D x thin mu(x).
$

The cosine similarity between $W_i$ and $H_i$ is then
$
  cos(theta_i) = (W_i dot H_i) / (||W_i|| ||H_i||) = overline(mu) sqrt(alpha / q).
$
As we see from plots, it gets indeed very close to 1 for both regularisation values.
#figure(
  grid(
    columns: 2,
    image("plots/cosine_sim_lambda_001.png"), image("plots/cosine_sim_lambda_0.png"),
  ),
  caption: [value of $gamma$ for the two regularisation values, and cosine similarity between $W_i$ and $H_i$ for each row.],
)


== Hebbian Componets


The easiest way to try and recover a susceptibility therm is by decmposing $W$ over Hebbian directions.  Let's move from $W_(i j)$ to $J_(i j)$. It will be useful to define $h_(i j)^nu=1/sqrt(L) xi^nu_i xi^nu_j$, so that $H_(i j) = sum_nu h_(i j)^nu$. We decompose each row $W_i$ as:
$
  W_i = sum_nu a^nu_i h_i^nu.
$

(I don't know to what extent this is true. To be investigated)

Then

$
  J_(i j) = sum_nu a^nu_i xi_i^(nu=1) h_(i j)^nu xi_j^(nu=1) =a_i^(nu=1)/sqrt(L) + 1/sqrt(L) sum_(nu>1) a_i^nu tilde(xi)^nu_i tilde(xi)^nu_j.
$

Define the naive projection
$
  mu_i^nu = W_i dot h_i^nu = 1/sqrt(L) sum_j W_(i j) xi^nu_i xi^nu_j.
$
As we know, $mu_i^nu$ is iid fromo $mu(x)$ with $x~cal(N)(0,1)$. Then, $mu_i^nu = sum_eta G^((i))_(nu eta) a_i^eta$ where

$
  G^((i))_(nu eta) = 1/L sum_j xi^nu_i xi^nu_j xi^eta_i xi^eta_j
$
is the gram matrix. The derivation of the susceptibility term is for the most part unchanged. However, we must replace as follows:

$
  -M/2 log det (I+K) -> -1/2 sum_nu log det (I+K_nu)
$

where $K_nu$ depends on the pattern $nu$ as follows:
$
  log det (I+K_nu) = sum_(i=2)^L log (1 + ((a_i^nu)^2 hat(h)_i^2) / L^2 q_(i-1))
$

and

$
  q_L - q_(L-1) = (1+ i/L a_L^nu hat(h)_L q_(L-1))^2/(1+ ((a_L^nu)^2 hat(h)_L^2)/L^2 q_(L-1)).
$



#show: arkheion-appendices

= Standard Hopfield Setting
In Standard Hopfield I would have an annealed partition function. Setting $y_i^mu := x_i thin xi_i^mu$, $(y_i^mu)^2 = 1$, so

$
  sum_i x_i sum_(j < i) xi_i^mu xi_j^mu x_j = sum_(i > j) y_i^mu y_j^mu = 1/2 [(sum_i y_i^mu)^2 - sum_i (y_i^mu)^2] = 1/2 [(sum_i x_i xi_i^mu)^2 - L].
$

Therefore

$
  EE_xi sum_x e^(1/L sum_mu sum_i x_i sum_(j < i) xi^mu_i xi^mu_j x_j) &= e^(-M / 2) thin EE_xi sum_x e^(1/(2 L) sum_mu (sum_i x_i thin xi^mu_i)^2)\
  &= e^(-M / 2) thin EE_xi integral product_mu dif m_mu thin e^(-L / 2 sum_mu m_mu^2) sum_x e^(sum_mu m_mu sum_i x_i thin xi^mu_i),
$

where the trajectory-independent prefactor $e^(-M/2) = e^(-alpha L / 2)$ comes from the diagonal $sum_i (y_i^mu)^2 = L$ piece. This factor cancels against an identical contribution coming from the diagonal $i = j$ piece of the standard $sum_(i, j) W_(i j) x_i x_j$ Hopfield energy, and is therefore commonly dropped together with $x$-independent normalisation constants. Keeping it explicit is required if one wants to compare the autoregressive and symmetric annealed free energies directly.


== Orthogonal Hebbian-Gaussian Ansatz


We write $J_(i j)$ as follows:
$
  J_(i j) = mu_i/sqrt(L) + gamma/sqrt(L) sum_(mu>1) tilde(xi)^mu_i tilde(xi)^mu_j + sqrt(q - gamma^2 alpha) z_(i j)
$

the integrals over the random variables can be performed as before, independently, and we get the following equation:

$
  dot(m) & = 2 integral D x D z thin sigma(beta(mu(x) m + sqrt(v+(q-gamma^2 alpha)(t-m^2)) z))-1 \
  dot(v) & = alpha gamma^2 -4 beta gamma v integral D x D z thin sigma'(beta(mu(x) m + sqrt(v+(q-gamma^2 alpha)(t-m^2)) z)).
$
or, specialising for the "fair" and "greedy" algorithms:
$
  dot(m) & = 2 integral D x D z thin sigma(mu(x) m + sqrt(v+(q-gamma^2 alpha)(t-m^2)) z)-1 \
  dot(v) & = alpha gamma^2 -4 gamma v integral D x D z thin sigma'(mu(x) m + sqrt(v+(q-gamma^2 alpha)(t-m^2)) z).
$
and
$
  dot(m) & = 2 integral D x thin Phi((mu(x) m )/ sqrt(v+(q-gamma^2 alpha)(t-m^2)))-1 \
  dot(v) & = alpha gamma^2 -4 gamma v/sqrt(v+(q-gamma^2 alpha)(t-m^2)) integral D x thin phi((mu(x) m )/ sqrt(v+(q-gamma^2 alpha)(t-m^2))).
$

This gives very wrong results.
