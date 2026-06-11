#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/drafting:0.2.2": inline-note, margin-note
#import "@preview/mannot:0.3.0": markrect

#show figure.caption: set align(left)
#show figure.caption: set text(style: "italic")

// #import "@preview/equate:0.3.2": equate

//// ALTERNATIVE ARTICLE TEMPLATES
// #import "@preview/rubber-article:0.5.0": article, appendix, maketitle
// #import "@preview/starter-journal-article:0.4.0": article, author-meta, appendix
// #import "@preview/elsearticle:0.4.2": *
//
#show: arkheion.with(
  title: [Masked Diffusion Models in high-dimension
    with shallow score functions],
  // authors: (
  //   (name: "Carlo Lucibello", email: "carlo.lucibello@unibocconi.it",
  //    affiliation: [Bocconi University, Milan]),
  // ),
  date: datetime.today().display("[day] [month repr:Long] [year]"),
  abstract: [#align(left)[We study minimal models of learned generative masked diffusion.]],
)


// #let note(content) = inline-note(content, fill: gray.lighten(80%), stroke: gray, par-break: false)
#let note(content) = highlight([NOTE: ] + content, fill: gray.lighten(80%))

#let citecolor = rgb("#93430e")
#show cite: set text(fill: citecolor)
#show link: set text(fill: blue)
#show link: underline
#show ref: set text(fill: blue)

// #show: equate.with(breakable: true, sub-numbering: true)
// #set math.equation(numbering: "(1.1)")

#let cM = $cal(M)$
#let ovx = $overline(x)$
#let tx = $tilde(x)$

#outline(depth: 2, indent: n => n * 1em)

= Introduction

Masked diffusion models @sahooSimpleEffectiveMasked2024, have emerged as a powerful class of generative models for sequences of discrete tokens, with applications in natural language processing, protein design and other domains. They can be seen as a generalization of autoregressive models, where the model is trained to predict a token given any subset of the other tokens, rather than only the past tokens as in standard autoregressive models. In this sense, they can be seen as any-order autoregressive models @ouYourAbsorbingDiscrete2025.

Here we study a simple model where the score function is a single linear layer. In the continuous diffusion case, this has been studied in Refs. @biroliGenerativeDiffusionVery2023@mergerGeneralizationDynamicsLinear2025. A linear model can only model Gaussian distributions, therefore multimodal distributions
cannot be expressed and also the model cannot memorize the training set (no memorization phenomenon). However, in the discrete case, even a linear model can express multimodal distributions and can also memorize (in some sense) the training set, as we shall see.

We investigate the phenomenon of memorization @bonnaireWhyDiffusionModels2025
@georgeDenoisingScoreMatching2025, and the one of speciation @biroliDynamicalRegimesDiffusion2024.

In a last application, we will consider a more complicated score model, the random feature score model, where the score function is a two-layer neural network with random and frozen first layer  and trainable second layer.

As for the data models we will consider: (i) uniform data on hypercube ${-1,+1}^L$, (ii) data from a hidden manifold.


We also explore the use of masked diffusion models as an associative memory under the erasure channel. Maybe the Gripon-Berrou model @gripon2011sparse can be a useful reference for this.

= Preliminaries

== Masked Diffusion Models

We denote with $m$ the mask token. We denote with $x$ a sequence of token of lenght $L$.  We assume that the tokens in $x$ are discrete and belong to a vocabulary of size $V$.

We call $cM$ a set of indices in ${1, dots, L}$ corresponding to the positions of the tokens where we want to impose a mask. We denote with $x_cM$ the subsequence of $x$ corresponding to the indices in $cM$. We denote with $overline(cM)$ the complementary set of indices to  $cM$.

The masked diffusion model is a probabilistic model over sequences of tokens defined by a model $p_theta (x_i|x_cM)$ that takes as input a masked sequence $x_cM$  and outputs probabilities over the tokens in the vocabulary for each masked position $i in overline(cM)$.

Masked diffusion models can be see as any-order autoregressive networks, since the model single variable marginals given any subset of variables, and not only
the conditionals given the past variables as in standard autoregressive models.


For a given clean (i.e. containing no mask tokens) sequence $x^0$ from the data distribution $p_"data"$ and for a given $0<= t <=1$, we denote with $x^t$ the sequence obtained by replacing each token $x_i^0$ with the mask token $m$ independently with probability $t$. The masked objective can then be defined as @ouYourAbsorbingDiscrete2025

$
  cal(L)(theta) = -EE_(t~U(0,1)) 1/t thin EE_(x^0,x^t) sum_(i thin : thin x^t_i=m) log p_theta (x^0_i|x^t).
$ <eq:loss>

Once the model is trained, sampling can be done using @algo:sample.



// #equate(
//   loss(theta) = EE_(x ~ p_data) EE_(cM ~ C) [- log p_theta(x_overline(cM) | x_cM)],
//   label: "loss",
// )

#figure(
  algo(
    title: "Algorithm 1 - Sample",
    parameters: ($"tokens per step" k$, $"length" L$),
  )[
    $x = (m)_(i=1)^L$ #comment[initialize with mask]\
    $cM = emptyset$\
    while $|cM| < L$ do #i\
    $tilde(k) = min(k, L - |cM|)$\
    sample distinct $i_1, dots, i_tilde(k) in overline(cM)$\
    sample $x_i ~ p_theta (x_i|x)$ for $i in {i_1, dots, i_tilde(k)}$ \
    $cM = cM union {i_1, dots, i_tilde(k)}$ #d\
    return $x$
  ],
  kind: "algo",
  supplement: "Algorithm",
) <algo:sample>


== Hidden Manifold Model

TODO @geraceGeneralisationErrorLearning2021

= The Linear Score Model

We assume a binary alphabet of Ising spins, $x_i in {-1, +1}$, and for convenience represent the mask as an auxiliary variable $m_i$, such that $m_i=1$ for a masked position and $m_i=0$ otherwise.
Therefore for a clean sequence $x$, the masked sequence $x^t$ is represented as the tuple $(x, m^t)$, where $m^t_i$ are i.i.d. Bernoulli variables with parameter $t$.

We use a linear model as a score function.
The model outputs the logits for the categorical distribution over the non-mask tokens in correspondence of a masked position $i$ as

$
  p_theta (x_i=1|x^t) = sigma(1 / sqrt(L) sum_(j=1)^L (W_(i j) (1-m_j^t) x_j + V_(i j) m_j^t))
$

where $sigma(z) = (1 + exp(-z))^(-1)$. The bias is not needed since that role can be fulfilled by $W_(i i 0)$.

We can study this problem as statistical physics problem with energy function given by the loss in @eq:loss.


We consider $M$ training examples, $x^mu$, $mu in {1, dots, M}$, sampled from a data distribution $p_"data"$. The simple model we consider can be factorized over output positions. Therefore we can optimize the $L$ neurons independently. For a given output $i$, the empirical loss over the training set reads
$
  cal(L)_(i) (w,v) &= - 1/M sum_(mu=1)^M EE_(t~U(0,1)) 1/t EE_(m^t) thin II(m^(t)_i=1) log sigma(1 / sqrt(L) x^mu_i sum_(j) (w_(j) (1-m^t_j)x_(j)^(mu)+v_(j) m^t_j)),\
  cal(L)(theta) &= sum_i cal(L)_(i) (W_i,V_i).
$
Here we denoted with $w$ the weights of a single neuron, i.e. $w_(j) = W_(i j)$.

We define the partition function
$
  Z_i (beta) = integral dif w dif v thin e^(-beta M cal(L)_i (w,v) - 1/2 beta lambda ||w||^2 - 1/2 beta lambda ||v||^2).
$ <eq:partfunc>
We denote with $chevron.l thin dot thin chevron.r_(beta,i)$ the average with respect to the corresponding Boltzmann distribution. The optimization is than obtained as
$
  hat(W)_i,hat(V)_i = "argmin"_(w,v) thin cal(L)_i (w,v) + 1/2 lambda ||w||^2 = lim_(beta -> oo) thin chevron.l w chevron.r_(beta,i),thin chevron.l v chevron.r_(beta,i).
$

The problem is convex, therefore the minimum is unique.
As metrics, we will consider the test loss

$
  cal(L)^"test" (theta) = - EE_(t~U(0,1)) 1/t EE_(x^0 ~ p_"data") EE_(x^t|x^0) sum_(i : x^t_i=m) log p_theta (x^0_i|x^t)
$
besides the train loss. We will also consider time-sliced train and test accuracy defined as
$
  cal(E)^"train"_(i,t) & = 1/M sum_(mu=1)^M EE_(x^(mu,t)) thin
                         II(x^(mu,t)_i=m) thin II (x^mu_i = "argmax"_c thin p_theta (x^0_i=c|x^(mu,t))), \
   cal(E)^"test"_(i,t) & = EE_(x^0~p_"data") EE_(x^t|x^0) thin
                         II(x^(t)_i=m) thin II (x^0_i = "argmax"_c thin p_theta (x^0_i=c|x^(t))).
$


We are interested in the high-dimensional limit where $L, M -> oo$ with $alpha = M/L$ finite. We will use the replica method to compute the typical value of the partition function over the training set sampled from a given data distribution $p_"data"$.


= Case I: Uniform data and linear score

Let's consider the simplest data distribution, the one which is uniform on the hypercube ${-1, +1}^L$, that is $p_"data" (x) = 2^(-L)$. Although trivial, when in presence of a finite training set ${x^mu}_(mu=1)^M$, this case is not trivial, since the model can memorize the training set.


As positions are statistically equivalent, we can focus on $i=1$. Considering the partition function in $Z_1 (beta)$ of @eq:partfunc, our goal is to compute the asymptotic free entropy

$
  phi(beta) = lim_(L -> oo) EE_X 1/L log Z_1 (beta),
$
where in the limit we keep the ratio $alpha = M/L$ fixed.


== Replica Calculation

=== Replicated Free Entropy

We introduce $n$ integers replicas and we compute

The replicated partition function reads
$
  EE_X Z_1^n (beta) =& EE_X integral product_a dif w^a dif v^a thin e^(-beta sum_(a=1)^n M cal(L)_(1) (w^a, v^a) - 1/2 beta lambda sum_(a=1)^n ||w^a||^2 - 1/2 beta lambda sum_(a=1)^n ||v^a||^2)\
  =& integral product_a dif w^a dif v^a thin
  (EE_x thin e^(beta integral_(0)^1 dif t thin 1/t
    EE_(m^t) sum_(a=1)^n II(m^(t)_1=1) log sigma
    (1 / sqrt(L) x_1 sum_(j) (w^a_(j) (1-m^t_j)x_(j)+v^a_(j) m^t_j))))^M\
  &e^(- 1/2 beta lambda sum_(a=1)^n ||w^a||^2 - 1/2 beta lambda sum_(a=1)^n ||v^a||^2).
$
where we used the fact that the data are i.i.d..

Let's focus on the term

$
  A & = EE_x thin e^(beta integral_(0)^1 dif t thin 1/t
      EE_(m^t) sum_(a=1)^n II(m^(t)_1=1) log sigma
      (1 / sqrt(L) x_1 sum_(j) (w^a_(j) (1-m^t_j)x_(j)+v^a_(j) m^t_j))) \
    & = EE_x thin e^(beta integral_(0)^1 dif t thin EE_(m_(\/1)^t) sum_(a=1)^n log sigma
      (1 / sqrt(L) x_1 sum_(j) (w^a_(j) (1-m^t_j)x_(j)+v^a_(j) m^t_j)))
$
where $EE_(m_(\/1)^t)$ is the expectation with respect to the mask vector $m^t$ conditioned on $m^t_1=1$.

Now we claim that for fixed $t, x, w^a,v^a$, the variable $z^a = 1 / sqrt(L) sum_(j) (w^a_(j) (1-m^t_j)x_(j)+v^a_(j) m^t_j)$ is asymptotically Gaussian by the central limit theorem.

The  mean is given by

$
  EE_(m^t_(\/1)) z_a & = EE_(m^t_(\/1)) 1 / sqrt(L) sum_(j) (w^a_(j) (1-m^t_j)x_(j)+v^a_(j) m^t_j) \
                     & = (1-t) / sqrt(L) sum_(j) w^a_(j) x_(j)+t /sqrt(L) sum_(j) v^a_(j) + O(1 /sqrt(L)) \
                     & =: (1-t) mu_a + t mu^v_a + O(1 /sqrt(L))
$
where in the last line we defined
$
  mu_a = 1 / sqrt(L) sum_(j) w^a_(j) x_(j), quad mu^v_a = 1 / sqrt(L) sum_(j) v^a_(j).
$
.

The covariance is given by
$
  "Var"_(m^t_(\/1)) z_a & = EE_(m^t_(\/1)) (1 / sqrt(L) sum_(j) (w^a_(j) (1-m^t_j)x_(j)+v^a_(j) m^t_j))^2 - mu_a^2 \
                        & = (t(1-t))/L sum_(j=1)^L (v_j^a - w_j^a x_j)^2 \
                        & -> t(1-t)(q_(a a) + q^v_(a a))
$
where in the last line we compute the large $L$ limit, using the definitions

$
  q_(a b) = 1/L sum_(j) w^a_(j) w^b_(j), quad q^v_(a b) = 1/L sum_(j) v^a_(j) v^b_(j).
$
Therefore we can write the exponent as
$
  beta integral_(0)^1 dif t thin sum_(a=1)^n EE_(z_a ~ cal(N)((1-t) mu_a + t mu^v_a, t(1-t)(q_(a a) + q^v_(a a)))) log sigma(x_1 z_a).
$

The quantities $mu^v_a$ and $q_(a b)$ and $q^v_(a b)$ are not data dependent, and can be enforced with Dirac's delta outside $A$. The quantities $mu_a$ instead depend on the data $x$, and we need to compute their joint distribution at fixed $w^a$s. Again by a CLT argument, we have that ${mu^w_a}_a$ are asymptotically jointly Gaussian with moments
$      EE_x_(\/1) thin mu_a & = EE_x_(\/1) 1 / sqrt(L) sum_(j !=1) w^a_(j) x_(j) = 0 \
EE_x_(\/1) thin mu_a mu_b & = EE_x_(\/1) 1 / L sum_(j,k !=1) w^a_(j) w^b_(k) x_(j) x_(k) =q_(a b). $.
The last expectation is over $x$ at fixed $x_1$. We have therefore obtained

$
  A = 1/2 sum_(x_1)EE_({mu_a}_a ~ cal(N)(0, q)) thin e^( beta integral_0^1 dif t sum_(a=1)^n EE_(z_a ~ cal(N)((1-t) mu_a + t mu^v_a, t(1-t)(q_(a a) + q^v_(a a)))) log sigma(x_1 z_a)).
$


Enforcing the definition of the overlaps with delta functions in integral representation, we can write the replicated partition function as

$
  EE_X Z_1^n (beta) = integral product_(a<=b) dif q_(a b) dif hat(q)_(a b) thin e^(n L phi_(beta) (q, hat(q))).
$


It can be shown that for the data we consider [Make this statement more precise!] we have $m^v_a=0$ and $q^v_(a b) = 0$. That is, the $v$ weights shrink to zero (even for $lambda=0$). With this simplification, we write the replicated parition function as
$
  phi_(beta) (q, hat(q)) = & - 1/(2 n) sum_(a,b) q_(a b) hat(q)_(a b) + G_S (hat(q))+ alpha G_E (q)
$ <eq:phi_before_rs>
with the entropic term
$
  G_S (hat(q)) = 1/(n L) log integral product_a dif w^a thin e^( - 1/2 beta lambda sum_a ||w^a||^2 + 1/2 sum_(a,b) hat(q)_(a b) w^a dot w^b)
$

and energetic term
$
       G_E (q) & = 1 / n log EE_({mu_a}_a ~ cal(N)(0, q)) thin e^( - beta sum_a ell (mu_a,q_(a a))) \
  ell (mu,q_d) & =-EE_t thin EE_(z ~ cal(N)((1-t) mu, t(1-t) q_d))log sigma(z).
$
In the definition of $G_E$ we also used the fact that $x_1$ is uncorrelated with $mu_a$ and can be set to $x_1=1$ for symmetry.

=== Replica Symmetric Ansatz

Now we apply the replica symmetric ansatz on the free entropy of @eq:phi_before_rs. We use:

$
  q_(a a) & = q + delta q \
  q_(a b) & = q
$

$
  hat(q)_(a a) & = hat(q) + delta hat(q) \
  hat(q)_(a b) & = hat(q).
$

Let's start with $G_S$:

$
  G_S &= 1/(n L) log integral product_a dif w^a thin e^( - 1/2 beta lambda sum_a sum_i (w^a_i)^2 + 1/2 [sum_i sum_a delta hat(q) (w_i^a)^2 + sum_i sum_(a b)hat(q) thin w_i^a w_i^b]) = \
  &=1/(n L) log integral product_a dif w^a thin e^( - 1/2 beta lambda sum_a sum_i (w^a_i)^2 + 1/2 [sum_i sum_a delta hat(q) (w_i^a)^2 + sum_i (sum_a sqrt(hat(q)) thin w_i^a)^2]) = \
  &=1/(n L) log integral product_(a i) dif w^a_i product_i thin e^( - 1/2 beta lambda sum_a (w^a_i)^2 + 1/2 [ sum_a delta hat(q) (w_i^a)^2 + (sum_a sqrt(hat(q)) thin w_i^a)^2]) = \
  &=1/(n L) log integral product_i product_a dif w^a_i product_i integral D_z thin e^( - 1/2 beta lambda sum_a (w^a_i)^2 + 1/2 [ sum_a delta hat(q) (w_i^a)^2 + 2 sqrt(hat(q))z sum_a w_i^a]) = \
  &=1/(n L) log (integral D_z (integral dif w thin e^( - 1/2[ beta lambda w^2 -delta hat(q) w^2 - 2 sqrt(hat(q)) w z]))^n )^L = \
  &=integral D_z log integral dif w thin e^( - 1/2 (beta lambda -delta hat(q)) w^2 + sqrt(hat(q)) w z) = \
  &= integral D_z thin (hat(q) z^2)/(2(beta lambda -delta hat(q))) = hat(q)/(2(beta lambda -delta hat(q)))
$

Now we take care of $G_E$. Since we have that the covariance matrix of each $mu$ is given by $q$, we'll also need its inverse. One has:

$
  q_(a a)^(-1) & = (delta q+(n-1)q)/(delta q (delta q + n q)) \
  q_(a b)^(-1) & = (-q)/(delta q (delta q + n q)).
$

We also need its determinant. $det Sigma = det q = delta q^(n-1) (delta q + n q)$.
We get

$
  G_E & = 1 / n log integral product_a (dif mu_a) / sqrt(2 pi) thin e^( - 1/(2)[sum_a mu_a^2 1/(delta q) -sum_(a b) mu_a mu_b q/(delta q (delta q + n q))]-1/2 log(delta q^(n-1)(delta q + n q)) -beta sum_a ell (mu_a,q+ delta q)) \
  &= 1 / n log integral product_a (dif mu_a) / sqrt(2 pi) thin e^( - 1/(2)[sum_a mu_a^2 1/(delta q) -(sum_(a) mu_a)^2 q/(delta q (delta q + n q))] -1/2 log(delta q^(n-1)(delta q + n q)) -beta sum_a ell (mu_a,q+ delta q))\
$

apply an Hubbard–Stratonovich transformation to linearize the quadratic term in the exponent, we get:

$
  G_E & = 1 / n log integral product_a (dif mu_a) / sqrt(2 pi)integral D x thin e^( - 1/(2 delta q)sum_a mu_a^2+ sqrt(q/(delta q (delta q + n q)))x sum_a mu_a -1/2 log(delta q^(n-1)(delta q + n q)) -beta sum_a ell (mu_a,q+ delta q))=\
  & = 1 / n log integral D x ( integral (dif mu) / sqrt(2 pi) thin e^( - 1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q)))x mu -beta ell (mu_a,q+ delta q)))^n e^(-1/2 log(delta q^(n-1)(delta q + n q)))=\
  &= integral D x log integral (dif mu) / sqrt(2 pi) thin e^( - 1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q)))x mu -beta ell (mu,q+ delta q)) -1/(2n) log(delta q^(n-1) (delta q + n q)) =\
  &= integral D x log integral (dif mu) / sqrt(2 pi) thin e^( - 1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q)))x mu -beta ell (mu,q+ delta q)) -1/2 log(delta q) -1/(2n) log(1+n q/(delta q))=\
  &= integral D x log integral (dif mu) / sqrt(2 pi) thin e^( - 1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q)))x mu -beta ell (mu,q+ delta q)) -1/2 log(delta q) -1/2 q/(delta q).
$

Numerically we will use:

$
  ell (mu,q) & =-EE_t thin EE_(z ~ cal(N)((1-t) mu, t(1-t) q))log sigma(z)= \
             & =-EE_t EE_(y~cal(N)(0,1))log(sigma(sqrt(t(1-t)q) y+(1-t) mu))
$

Finally we take Care of the first therm in $phi_beta$. In the small $n$ limit:

$
  phi_beta & = - 1/(2n) n(q hat(q)+q delta hat(q) + hat(q) delta q + delta q delta hat(q))-1/(2n)n(n-1)q hat(q) + G_S (hat(q))+ alpha G_E (q)= \
  & = - 1/2 (q delta hat(q) + hat(q) delta q + delta q delta hat(q)) + G_S (hat(q))+ alpha G_E (q).
$

In the zero temperature limit, we apply the scalings:
$
       delta q & -> (delta q)/beta \
        hat(q) & -> beta^2 hat(q) \
  delta hat(q) & -> beta delta hat(q).
$
We rewrite the free entropy as
$
  phi= lim_(beta->infinity) 1/beta phi_beta.
$

The first therm in $phi$ becomes

$
  - 1/(2beta) (beta q delta hat(q) + beta^2 hat(q) (delta q)/beta + (delta q)/beta beta delta hat(q)) -> -1/2 q delta hat(q) -1/2 hat(q) delta q.
$

We are then left with

$
  phi & = -1/2 q delta hat(q) -1/2 hat(q) delta q +tilde(G)_S + alpha tilde(G)_E
$
$
  tilde(G)_S & = lim_(beta->infinity) 1/beta G_S =1/beta [beta^2 hat(q)/(2beta (lambda -delta hat(q)))]= \
             & = hat(q)/(2 (lambda -delta hat(q))).
$

$
  tilde(G)_E& = lim_(beta->infinity) 1/beta G_E =1/beta [integral D x log integral d mu thin e^( - beta/(2 delta q) mu^2 + beta sqrt(q)/(delta q)x mu -beta ell (mu,q+ (delta q)/beta)) -1/2 log((delta q)/beta) -1/2 beta q/(delta q)]=\
  &= integral D x max_mu [ - 1/(2 delta q) mu^2 + sqrt(q)/(delta q)x mu - ell (mu,q)] -1/2 q/(delta q).
$

Where $mu_*(x) = arg max [...]$ has to be intended as a function of $x$. Putting them together, we get

$
  phi =alpha tilde(G)_E (q, delta q) -1/2 (q delta hat(q) + hat(q) delta q -hat(q)/ (lambda -delta hat(q))).
$

Imposing stationarity for $hat(q)$ and $delta hat(q)$ leads to:

$
        (partial phi) / (partial hat(q)) = 0 & -> delta hat(q) = lambda -1/(delta q) \
  (partial phi) / (partial delta hat(q)) = 0 & -> hat(q) = -q/(delta q^2).
$

Plugging back these expressions in $phi$, we get
$
  phi = alpha tilde(G)_E (q, delta q) +1/2 q/(delta q) -lambda/2 q.
$

The stationarity on the remainig order parameters leads to the saddle point equations:

$
  (partial phi) / (partial q) = 0 & -> alpha partial_q tilde(G)_E +1/2 q/(delta q) -lambda/2=0 -> delta q = (lambda-2 alpha partial_q tilde(G)_E)^(-1) \
  (partial phi) / (partial delta hat(q)) = 0 & -> alpha partial_(delta q) tilde(G)_E - 1/2 q/(delta q^2)=0 -> q = 2 alpha delta q^2 partial_(delta q) tilde(G)_E.
$

The derivatives of $tilde(G)_E$ can be computed as
$
          partial_q tilde(G)_E & = integral D_x [(x mu_*)/(2 delta q sqrt(q))- partial_q ell (mu_*,q)] -1/(2 delta q) \
  partial_(delta q) tilde(G)_E & = integral D_x [(mu_*^2)/(2 delta q^2)- (x mu_* sqrt(q))/(delta q^2)] +1/2 q/(delta q^2) \
$

where, for our choice of activatio function $sigma$, we have
$ partial_q ell (mu,q) = EE_t EE_(y ~ cal(N)(0,1)) sigma(sqrt(t(1-t)q)z+(1-t)mu)z sqrt(t(1-t))/ (2 sqrt(q)). $

=== Accuracy as an Associative Memory

We now show how to compute train metrics for our model
as a function of the order parameters at saddle point.

We want to compute the time-sliced test accuracy defined as

$
  cal(E)_(1,t) & = EE_(x^0~p_"data") EE_(x^t|x^0) thin
  II(x^(t)_1=m) thin II (x^0_1 = "argmax"_c thin p_theta (x^0_1=c|x^(t)))=\
  & = EE_X lr(chevron.l EE_(m_t) Theta(x_i^(mu=1)sum_j w_j/sqrt(L) x_j^(mu=1)) chevron.r) = EE_X chevron.l epsilon_(1,t)(w, x^(mu=1)) chevron.r.
$

Where we discarded the $v$ therm as we argued it is ininfluntial. The $chevron.l thin dot thin chevron.r$ has to be intended in the $beta -> infinity$ limit.

$
  cal(E)_(1,t) & =lim_(beta, L -> infinity) EE_X 1/(Z_1(beta)) integral dif w thin e^(- 1/2 beta lambda ||w||^2) thick product_(mu>1)^M e^(-beta cal(L)_(1) (w, x^mu)) thick e^(-beta cal(L)_(1) (w, x^(mu=1))) epsilon_(1,t)(w, x^(mu=1)) \
  & = lim_(beta, L -> infinity) 1/(Z_1(beta)) integral dif w thin e^(- 1/2 beta lambda ||w||^2) thick [ EE_x e^(-beta cal(L)_(1) (w, x))]^(M-1) thick EE_(x^(mu=1)) e^(-beta cal(L)_(1) (w, x^(mu=1))) epsilon_(1,t)(w, x^(mu=1)).
$

We write $Z_1(beta)^(-1) = lim_(n->0) Z_1(beta)^(n-1)$

$
  cal(E)_(1,t) & =lim_(beta, L -> infinity) lim_(n->0) integral dif w e^(- 1/2 beta lambda sum_a ||w^a||^2) [EE_x e^(-beta sum_a cal(L)_(1) (w^a, x))]^(M-1) EE_(x^(mu=1)) e^(-beta sum_a cal(L)_(1) (w^a, x^(mu=1))) epsilon_(1,t)(w, x^(mu=1)) \
  & = lim_(beta, L -> infinity) lim_(n->0) integral dif w e^(- 1/2 beta lambda sum_a ||w^a||^2) [EE_x e^(-beta M sum_a cal(L)_(1) (w^a, x))] (EE_(x^(mu=1)) e^(-beta sum_a cal(L)_(1) (w^a, x^(mu=1))) epsilon_(1,t)(w, x^(mu=1)))/(EE_(x^(mu=1)) e^(-beta sum_a cal(L)_(1) (w^a, x^(mu=1)))) \
  & = lim_(beta, L -> infinity) lim_(n->0) integral dif w e^(- 1/2 beta lambda sum_a ||w^a||^2) [EE_x e^(-beta M sum_a cal(L)_(1) (w^a, x))] tilde(cal(E))_(1,t)(w, x^(mu=1)).
$

Let's focus on the new term   $tilde(cal(E))_(1,t)(w, x)$:

$
  tilde(cal(E))_(1,t)(w, x) & = (EE_(x) e^(-beta sum_a cal(L)_(1) (w^a, x)) EE_(m_t) Theta(x_i sum_j w_j/sqrt(L) x_j))/(EE_(x) e^(-beta sum_a cal(L)_(1) (w^a, x)))
$

where following the same CLT arguments as before, we can write

$
  tilde(cal(E))_(1,t)(q, x) & = (EE_({mu_a}_a ~ cal(N)(0, q)) thin e^(-beta sum_a ell(mu_a, q_(a a))) EE_(z_a ~ cal(N)((1-t) mu_a, t(1-t)q_(a a) )) Theta(x_i z_a))/(EE_({mu_a}_a ~ cal(N)(0, q)) thin e^(-beta sum_a ell(mu_a, q_(a a)))) \
  & = (EE_({mu_a}_a ~ cal(N)(0, q)) thin e^(-beta sum_a ell(mu_a, q_(a a))) EE_(z ~ cal(N)((1-t) mu, t(1-t)q_d )) Theta(z))/(EE_({mu_a}_a ~ cal(N)(0, q)) thin e^(-beta sum_a ell(mu_a, q_(a a)))) \
$

where we used the symmetry to set $x_i=1$. Moreover we have:

$
  EE_(z ~ cal(N)((1-t) mu, t(1-t)q )) Theta(z) = 1-H(mu/sqrt(q) sqrt((1-t)/t)) = Phi(mu/sqrt(q) sqrt((1-t)/t)).
$

We can plug in the RS ansatz. The numerator is:
$
  & integral product_a (dif mu_a)/sqrt(2 pi) e^(1/2[sum_a mu_a^2 1/(delta q)-sum_(a b)mu_a mu_b q/(delta q (delta q +n q))]-1/2 log [delta q^(n-1)(delta q +n q)]-beta sum_a ell(mu_a, q+delta q)) Phi(mu_(a=1)/sqrt(q) sqrt((1-t)/t))= \
  = & sqrt(delta q^(n-1)(delta q + n q)) integral D x [ integral (dif mu)/sqrt(2 pi) e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q))]^n Phi(mu_(a=1)/sqrt(q) sqrt((1-t)/t)).
$

The denominator is the same without the $Phi$ term. Therefore we have
$
  tilde(cal(E))_(1,t)(q, delta q) & = (integral D_x integral dif mu
  e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q)) Phi(mu/sqrt(q) sqrt((1-t)/t)))/(integral D_x integral dif mu e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q))).
$

Notice that we wrote everything as a function of $q$ and $delta q$ and used the RS ansatz, the whole equation being:

$
  cal(E)_(1,t) & = lim_(beta, L -> infinity) lim_(n->0) integral dif q dif delta q e^(-n beta L phi(q, delta q)) tilde(cal(E))_(1,t)(q, delta q, x^(mu=1)).
$


We expect the $tilde(cal(E))_(1,t)$ therm to be $cal(O)(1)$ and to not be able to shift the SP. So in the limits we will just have $cal(E)_(1,t) & = tilde(cal(E))_(1,t)(q^*, delta q^*)$. We just need to compute the high temperature limit of the latter.

$
  tilde(cal(E))_(1,t)(q^*, delta q^*) & = lim_(beta -> infinity) (integral D_x integral dif mu
  e^(beta [ - 1/(2 delta q^*) mu^2 + sqrt(q^*)/(delta q^*)x mu - ell (mu,q^*)]) Phi(mu/sqrt(q^*) sqrt((1-t)/t)))/(integral D_x integral dif mu e^( beta[- 1/(2 delta q^*) mu^2 + sqrt(q^*)/(delta q^*)x mu - ell (mu,q^*)])) =\
  & = integral D_x Phi((mu_*(x)) /sqrt(q^*) sqrt((1-t)/t))
$

where as before $mu^*(x)$ is the argmax of the exponent.

// TODO add figure
// #figure(
//   image("../julia-code/SP/plots/varepsilon_vs_alpha.png"),
//   caption: [Accuracy $cal(E)_t$ as a function of $alpha$ for $lambda=0.001$],
// ) <fig:accuracy>


=== Time-integrated Accuracy approximation

Having obtained an expression for the time-sliced accuracy, $cal(E)_t$, we woul like to obtain an approximated expression for the the overlap of the generated sample $hat(x)^mu$ when starting the generation from a partially masked training example $x^mu$ with masking fraction $t$, that is $x^(mu,t)$. 

Assume that at time $t$ the we have a fraction $t$ of masked positions, 
and the unmasked positions could be either correct or incorrect with respect to the original training example $x^mu$. We denote with $c_t$ the fraction of correctly unmasked positions, and with $e_t$ the fraction of incorrectly unmasked positions, so that the total masking fraction is $t = 1 - c_t - e_t$.

Assume we can compute $cal(E)_t (c_t, e_t)$, that is the probability that the model correctly predicts a position given that at time $t$ the fraction of correctly unmasked positions is $c_t$ and the fraction of incorrectly unmasked positions is $e_t$.
We can compute this quantity by adapting the previous calculation for $cal(E)_(t)$.


For convenience, let's call $tilde(t) = 1-t$.
Neglecting correlations developed during the generation process, we can write the following approximate equations for the evolution of $c_t$ and $e_t$ during the generation process in the high-dimensional limit:
$
  (dif c_t) / (d tilde(t)) & = cal(E)_t (c_t, e_t) \
  (d e_t) / (d tilde(t)) & =1 - cal(E)_t (c_t, e_t)
$

// TODO change name of accuracy cal(E), because that name is appriopriate for error rather than accuracy

This is to be integrate from a certain time $t$ down to 0, with initial conditions $c_t = 1 - t$, $e_t = 0$.

The final overlap when starting from a configuration with masking fraction $t$ is then given by $m^("fin")(t) = c_0 - e_0$.

TODO We plot $m^"fin"$ vs $m^"in" = 1 - t$ as approximated by the ODE vs the empirical results from simulations in Figure ...


== Experiments

We train the model using pytorch and pytorch lightning. Each model is trained for at least 5000 epochs, using batch sizes of 512 and the AdamW optimizer.


=== U-Turn Memorization

In this experiment, for each example $x^0$ in the training set and a given time $t$, we mask a certain fraction ($t$ in expectation) of the positions according to the forward process to obtain $x^t$. We then use $x^t$ as the starting point for the reverse process in @algo:sample, where we unmask one position at a time to produce a final configuration $hat(x)^0$. Averaging over all examples, we estimate the U-turn overlap defined as

$
  q^t_U = EE 1 / L sum_i hat(x)^0_i thin x^0_i.
$

This is plotted in @fig:uturn_linear_uniform for different values of $alpha$ as a function of $t$. While for a completely unbiased process
we would have $q^t_U = 1-t$, we see that the we have a memorization effect, since the process recovers a larger part of the original configuration, also for large $t$.

#figure(
  grid(
    columns: 2,
    image("plots/uturn_linear_uniform_alpha0.1_l2reg0.0.png"),
    image("plots/uturn_linear_uniform_alpha0.5_l2reg0.0.png"),
  ),
  caption: [U-Turn memorization for uniform data and all times. On the $x$-axis, we have the initial time for the reverse process. This corresponds
    to an initial overlap $1-t$. If there is no memorization at all, $1-t$ would also be the final overlap (diagonal line).
    Since all datapoints are above the diagonal, there is some memorization. (Left)  $alpha=0.1$, (Right)  $alpha=0.5$.],
) <fig:uturn_linear_uniform>

=== Full Memorization

We now check instead if the model generates samples close to the training point when starting from a configuration with all positions masked, that is $t=1$.

In this experiment, for each generated sample $hat(x)$, we compute the overlap with all training examples, and we consider the top 3 overlaps. We then average over all generated samples.

As shown in @fig:fullmem_linear_uniform, we see that the model is not able to fully memorize the training set, since the top overlap is significantly smaller than 1 and
dicreases with $L$. Nonetheless, the top overlap is significantly larger than what would be obtained by random guessing, which is of order $sqrt(2log(M)/L)$. Therefore, at small and medimum data size, the model partially memorizes the training set when $alpha$ is small enough.

#figure(
  grid(
    columns: 2,
    image("plots/top3_overlaps_linear_uniform_alpha0.1_l2reg0.0.png"),
    image("plots/top3_overlaps_linear_uniform_alpha0.5_l2reg0.0.png"),
  ),
  caption: [U-Turn memorization for uniform data and all times. On the $x$-axis, we have the initial time for the reverse process. This corresponds
    to an initial overlap $1-t$. If there is no memorization at all, $1-t$ would also be the final overlap (diagonal line). Data size ],
) <fig:fullmem_linear_uniform>



= Case II: Hidden Manifold Data and Linear Score

We consider the case in which the data is generated from a $D$-dimensional manifold, through a random projection matrix $F in RR^(L times D)$, and a $"sign"$  activation:
$
  F_(i j) & ~ cal(N)(0, 1), \
     z^mu & ~ cal(N)(0, I_D), \
     x^mu & = "sign"((F z^mu) / sqrt(D)).
$


The computation of $A$ now involve computing the joint distribution of $mu_a = 1/sqrt(L) sum_(j != 1) w^a_(j) x_(j)$, and the first data component preactivation $h = 1/sqrt(D) sum_k F_(1 k) z_k$ at given $F$. This is equivalent to the supervised random feature case, assuming the teacher model $theta^*$ is $theta^*_k =F_(1 k)$. See Ref. @geraceGeneralisationErrorLearning2021.

$
  EE_(x|F) thin mu_a &= 1/sqrt(L) EE_z sum_(j != 1) w^a_(j) "sign"(sum_k (F_(j k) z_k)/ sqrt(D)) = 0\
  EE_(x|F) h &= EE_z sum_k (F_(1 k) z_k)/ sqrt(D) = 0\
  EE_(x|F) h^2 &= 1\
  EE_(x|F) h mu_a &= 1/sqrt(L) EE_z sum_(j != 1) w^a_(j) sum_k (F_(1 k) z_k )/ sqrt(D) "sign"(sum_k' (F_(j k') z_k')/ sqrt(D))\
  &approx kappa_1 1/sqrt(L) 1 / D sum_(j!=1,k) w^a_(j) F_(1 k) F_(j k)\
  EE_(x|F) mu_a mu_b &= 1/L EE_z sum_(j,j' != 1) w^a_(j) w^b_(j') "sign"(sum_k (F_(j k) z_k)/ sqrt(D)) "sign"(sum_k' (F_(j' k') z_k')/ sqrt(D))\
  &approx kappa^2_* q_(a b) + kappa_1^2 q^s_(a b)
$
with $kappa_1 = EE_(u~cal(N)(0,1)) u "sign"(u) = sqrt(2/pi)$, and $kappa^2_star = EE_(u_~ cal(N)(0, 1)) "sign"^2(u)- kappa_1^2 = 1 - 2 / pi$.


= Beyond the Linear Model: Random Feature Score Model

== Experiments

=== U-Turn Memorization


#figure(
  image("plots/uturn_allmodels_uniform_L1024_alpha0.1_l2reg0.0.png"),
  caption: [U-Turn memorization. On the $x$-axis, we have the initial time for the reverse process. This corresponds
    to an initial overlap $1-t$. If there is no memorization at all, $1-t$ would also be the final overlap (diagonal line).
    Since all datapoints are above the diagonal, there is some memorization. ],
) <fig:uturn_allmodels_uniform>


= Experiments on Real Architectures and Datasets

We perform U-Turn experiments on real architectures.


#bibliography("bibliography.bib")


#show: arkheion-appendices


= Appendix Case I: Uniform Data and Linear Score

== Other U-Turn Plots

In @fig-app:uturn_linear_uniform  we see that finite size effects are small.

#figure(
  grid(
    columns: 2,
    image("plots/uturn_linear_uniform_alpha0.1_l2reg0.0.png"),
    image("plots/uturn_linear_uniform_alpha0.5_l2reg0.0.png"),
  ),
  caption: [U-Turn memorization for uniform data and all times. On the $x$-axis, we have the initial time for the reverse process. This corresponds
    to an initial overlap $1-t$. If there is no memorization at all, $1-t$ would also be the final overlap (diagonal line).
    Since all datapoints are above the diagonal, there is some memorization. (Left)  $alpha=0.1$, (Right)  $alpha=0.5$.],
) <fig-app:uturn_linear_uniform>
