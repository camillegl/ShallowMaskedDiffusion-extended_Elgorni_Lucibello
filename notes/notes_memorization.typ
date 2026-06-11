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
  title: [Masked Diffusion as Associative Memory],
  authors: (
    (name: "Filippo Elgorni", email: "filippo.elgorni@phd.unibocconi.it", affiliation: [Bocconi University, Milan]),
    (name: "Carlo Lucibello", email: "carlo.lucibello@unibocconi.it", affiliation: [Bocconi University, Milan]),
  ),
  date: datetime.today().display("[day] [month repr:Long] [year]"),
  abstract: [#align(left)[
    Masked diffusion language models (MDLMs) have recently emerged as a flexible class of generative models for discrete data, extending autoregressive approaches by enabling any order generation and multiple token unmasking. As in continuous diffusion probabilistic models, a central question is to understand under which conditions these models memorize their training data rather than generalizing beyond it.

    We first provide empirical evidence that memorization phenomena are indeed present in practical MDLMs trained on real datasets. Motivated by this observation, we introduce a minimal model of a learned masked diffusion system that is analytically tractable using tools from statistical physics, where the denoising model is given by a single dense layer, and the synthetic data is made of uniformly distributed binary sequences. In the high-dimensional proportional limit, we show that this model naturally exhibits behavior analogous to associative memories, acting as a retrieval system under partial observation.

    We characterize the memorization behavior as a function of the model load, deriving predictions for reconstruction performance and retrieval dynamics. These theoretical predictions are validated through experiments on both synthetic and real datasets.
  ]],
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
@georgeDenoisingScoreMatching2025. The data models we will consider: uniform data on the hypercube ${-1,+1}^L$,

We explore the use of masked diffusion models as an associative memory under the erasure channel.
This is the first work to provide a precize high-dimensional characterization of the memorization capacity of a diffusion model.

= Related Works

*Linear scores for generative diffusion* The study of linear score models for continuous diffusion has been done in

@biroliGenerativeDiffusionVery2023@mergerGeneralizationDynamicsLinear2025.

*Diffusion as Associative Memory*

@ambrogioniSearchDispersedMemories2023a
@bonnaireWhyDiffusionModels2025@georgeDenoisingScoreMatching2025@achilliMemorizationGeneralizationGenerative2025@phamMemorizationGeneralizationEmergence2025


*Error Correcting Codes* Maybe the Gripon-Berrou model @gripon2011sparse can be a useful reference for this.

*Hopfield Networks*

= Preliminaries

== Memorization in Masked Diffusion Models
We trained the masked diffusion model of @schiff2025simple on progressively smaller sbsets of the cifar10 @krizhevsky2009learning dataset, and observed phenomena compatible with memoriztion. We worked with images instead of text to be abel to work with a reduced vocabulary. Indeed here we have a 1-hot encoding of 256 variables + 1 mask for each of the 32x32x3 pixel of each image.

We find that, for smaller datasets, the model is able to reconstruct the training images with high fidelity, even when starting from a heavily masked version of the image (95% of the tokens masked).

#figure(
  grid(
    columns: 1,
    image("plots/mdlm.png"),
  ),
  caption: [U-turn/reconstruction experiment starting from an image where 95% of the tokens are masked. In order to generate realistic images in the generalisation regime, we used classifier free guidance with $gamma=2.0$. The number of steps is same as the size of an image. Each model has been trained for 200000 epochs. ],
)
Moreover, even when new images are generated from scratch, we find that the models trained on smaller datasets generate samples close (or in some case nearly identical) to those of the original dataset in the L2 sense. We can compute the frequency of memorised images as the fraction of generated images that are closer to their nearest neighbour than their second nearest neighbour by a given threshold $kappa$ as proposed in @yoon2023diffusion: $x$ is considered memorised if, given $x^(mu_1)$ and $x^(mu_2)$ its nearest and second nearest neighbour in the training set, one has

$
  (||x-x^(mu_1)||^2)/(||x-x^(mu_2)||^2) < kappa.
$

As prior papers in the literature, we choose $kappa=1\/3$.Being in a categorical setting we can also work with the Hamming distance and compute the fraction of generated images that have at least a portion of their tokens equal to those of an image from the training data. As expected, for both metrics we see that the fraction of memorised images increases as the training set size decreases.

#figure(
  grid(
    columns: 1,
    image("plots/f_mem_by_dataset_size.png", width: 75%),
  ),
  caption: [Frequency of memorised images as a function of the training set size. The red line corresponds to the L2 criterion with $kappa=1\/3$. The blue line corresponds to the Hamming distance criterion, where an image is considered memorised if it has at least the percentage reffered in the legend of its tokens equal to those of an image in the training set. For each model we generated 1000 images from scratch and used cfg on one of the 10 categories at random with $gamma=1.0$. The models used are the same of above.],
)

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


== The Linear Score Model

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


= Replica Calculation

Let's consider the simplest data distribution, the one which is uniform on the hypercube ${-1, +1}^L$, that is $p_"data" (x) = 2^(-L)$. Although trivial, when in presence of a finite training set ${x^mu}_(mu=1)^M$, this case is not trivial, since the model can memorize the training set.


As positions are statistically equivalent, we can focus on $i=1$. Considering the partition function in $Z_1 (beta)$ of @eq:partfunc, our goal is to compute the asymptotic free entropy

$
  phi(beta) = lim_(L -> oo) EE_X 1/L log Z_1 (beta),
$
where in the limit we keep the ratio $alpha = M/L$ fixed.

== Replicated Free Entropy

We introduce $n$ integers replicas and we compute

$
  EE_X log Z_1 (beta) = lim_(n -> 0) (EE_X Z_1^n (beta) - 1)/n.
$

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
$
       EE_x_(\/1) thin mu_a & = EE_x_(\/1) 1 / sqrt(L) sum_(j !=1) w^a_(j) x_(j) = 0 \
  EE_x_(\/1) thin mu_a mu_b & = EE_x_(\/1) 1 / L sum_(j,k !=1) w^a_(j) w^b_(k) x_(j) x_(k) =q_(a b).
$
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

== Replica Symmetric Ansatz

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
  &=1/(n L) log integral product_i product_a dif w^a_i product_i integral D z thin e^( - 1/2 beta lambda sum_a (w^a_i)^2 + 1/2 [ sum_a delta hat(q) (w_i^a)^2 + 2 sqrt(hat(q))z sum_a w_i^a]) = \
  &=1/(n L) log (integral D z (integral dif w thin e^( - 1/2[ beta lambda w^2 -delta hat(q) w^2 - 2 sqrt(hat(q)) w z]))^n )^L = \
  &=integral D z log integral dif w thin e^( - 1/2 (beta lambda -delta hat(q)) w^2 + sqrt(hat(q)) w z) = \
  &= integral D z thin (hat(q) z^2)/(2(beta lambda -delta hat(q))) = hat(q)/(2(beta lambda -delta hat(q)))
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
  &= integral D x log integral (dif mu) / sqrt(2 pi) thin e^( - 1/(2 delta q) mu^2 + sqrt(q/( (delta q )^2))x mu -beta ell (mu,q+ delta q)) -1/2 log(delta q) -1/2 q/(delta q).
$

Numerically we will use:

$
  ell (mu,q) & =-EE_t thin EE_(z ~ cal(N)((1-t) mu, t(1-t) q))log sigma(z)= \
             & =-EE_t EE_(y~cal(N)(0,1))log(sigma(sqrt(t(1-t)q) y+(1-t) mu))
$

Finally we take care of the first term in $phi_beta$. In the small $n$ limit:

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

Where $mu_*(x) = arg max [...]$ has to be intended as a function of $x$. Numerically this can be done by implicitly solving

$
  partial_mu [ - 1/(2 delta q) mu^2 + sqrt(q)/(delta q)x mu - ell (mu,q)] =0 -> mu_* /(delta q) = sqrt(q)/(delta q) x - partial_mu ell (mu_*,q)
$

where $partial_mu ell(mu, q) = EE_t EE_y [sigma(y sqrt(t(1-t)q)+(1-t)mu)(1-t)]-1\/2$. Putting them together, we get

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
$<eq:SP>

The derivatives of $tilde(G)_E$ can be computed as
$
          partial_q tilde(G)_E & = integral D x [(x mu_*)/(2 delta q sqrt(q))- partial_q ell (mu_*,q)] -1/(2 delta q) \
  partial_(delta q) tilde(G)_E & = 1/2 integral D_x ((mu_*(x))/(delta q) -x sqrt(q)/(delta q))^2 \
$

where, for our choice of activation function $sigma$, we have
$ partial_q ell (mu,q) = EE_t EE_(y ~ cal(N)(0,1)) sigma(sqrt(t(1-t)q)z+(1-t)mu)z sqrt(t(1-t))/ (2 sqrt(q)). $

The Train Loss satisfies $L phi = -M cal(L)_"train" - 1/2 lambda ||w||^2$ from which $cal(L)_"train" = -(lambda q\/2 + phi)\/alpha$


#figure(
  grid(
    columns: 2,
    image("plots/SP/replica_lambdas_0p01_to_0p01_alpha_0p08_to_13p0_q_and_loss.png"),
    image("plots/SP/replica_lambdas_0p0_to_0p0_alpha_0p03_to_10p0_q_and_loss.png"),
  ),
  caption: [Results of the saddle point equations compared with their experimetal values for $lambda=0.01$ (left) and $lambda=0$ (right). ],
)

== Accuracy as an Associative Memory

We now show how to compute some metrics for our model
as a function of the order parameters at saddle point. We focus on the time-sliced test accuracy $cal(E)_t$  defined as the probability, given a datum $mu=1$, to correctly unmask the next token having already predicted a fraction $1-t$ of its tokens correctly. This quantity already describes a scenario in which the leftover fraction $t$ of tokens are unmasked in one-shot.
Indeeed, if one defines the magnetization $m_t$ as the differece between the proportion of correctly unmasked tokens and the proportion of incorrectly unmasked tokens with respect to a datum $mu=1$, one can use $cal(E)_t$ to get the final magnetisation: to the initial $1-t$ correct tokens one must add $t$ of them with probability $cal(E)_t$ and subtract $t$ of them with probability $1-cal(E)_t$. Therefore the final magnetization is given by
$
  m_f (m_i=1-t) & = 1-t + t cal(E)_t -t(1-cal(E)_t) = 1 - 2t(1-cal(E)_t)
$

=== Time Sliced Test Accuracy

Notice that this object will depend on the sampling algorithm we use. If we sample each next token via a Bernoulli distribution, in what will from now on defined as a "Fair" smpling scheme, we will compute the time-sliced test accuracy defined as

$
  cal(E)_t &= EE_X lr(chevron.l EE_(m^t) thin II(x_i^(mu=1) = hat(x) ~ "Bernoulli"(sigma(sum_j w_j/sqrt(L) x_j^(mu=1)(1-m_j^t)))) chevron.r).
$

We can also consider a "Greedy" sampling scheme, where we sample each next token by just taking the sign of the prectivation. In what follows we will focus on the "Fair" sampling scheme, but since the derivtion is valid for any logistic-like activation function  symmetric arond 0, it can be easily adapted to the "Greedy" scheme by replcing the $sigma$ with the $Theta$ function at the end of each calculation.

We discarded the $v$ therm as we argued it is ininfluntial. The $chevron.l thin dot thin chevron.r$ has to be intended in the $beta -> infinity$ limit.
In theory $cal(E)_t$ also depeds on the component $i$ (which we have put to $1$), but due to the symmetry of the system it is the same for all components.

Let's write

$
  cal(E)_t = EE_X chevron.l epsilon_t (w, x^(mu=1)) chevron.r.
$
Then
$
  cal(E)_t & =lim_(beta, L -> infinity) EE_X 1/(Z_1(beta)) integral dif w thin e^(- 1/2 beta lambda ||w||^2) thick product_(mu>1)^M e^(-beta cal(L)_(1) (w, x^mu)) thick e^(-beta cal(L)_(1) (w, x^(mu=1))) epsilon_t (w, x^(mu=1)) \
  & = lim_(beta, L -> infinity) 1/(Z_1(beta)) integral dif w thin e^(- 1/2 beta lambda ||w||^2) thick [ EE_x e^(-beta cal(L)_(1) (w, x))]^(M-1) thick EE_(x^(mu=1)) e^(-beta cal(L)_(1) (w, x^(mu=1))) epsilon_t (w, x^(mu=1)).
$

We write $Z_1(beta)^(-1) = lim_(n->0) Z_1(beta)^(n-1)$

$
  cal(E)_t & =lim_(beta, L -> infinity) lim_(n->0) integral dif w thin e^(- 1/2 beta lambda sum_a ||w^a||^2) [EE_x e^(-beta sum_a cal(L)_(1) (w^a, x))]^(M-1) EE_(x^(mu=1)) e^(-beta sum_a cal(L)_(1) (w^a, x^(mu=1))) epsilon_t (w^(a=1), x^(mu=1)) \
  & = lim_(beta, L -> infinity) lim_(n->0) integral dif w thin e^(- 1/2 beta lambda sum_a ||w^a||^2) [EE_x e^(-beta sum_a cal(L)_(1) (w^a, x))]^M (EE_(x^(mu=1)) e^(-beta sum_a cal(L)_(1) (w^a, x^(mu=1))) epsilon_t (w^(a=1), x^(mu=1)))/(EE_(x^(mu=1)) e^(-beta sum_a cal(L)_(1) (w^a, x^(mu=1)))) \
  & = lim_(beta, L -> infinity) lim_(n->0) integral dif w thin e^(- 1/2 beta lambda sum_a ||w^a||^2) [EE_x e^(-beta sum_a cal(L)_(1) (w^a, x))]^M tilde(cal(E))_t (w^(a=1)).
$

Let's focus on the term   $tilde(cal(E))_t (w^(a=1))$. Conditioned to $w$, $x$ and $m^t$ the preactivation will be taken to be gaussian as before, $sum_j w_j/sqrt(L) x_j^(mu=1)(1-m_j^t) = z_(a=1) ~ cal(N)((1-t)mu, t(1-t)q_d)$. Therefore we can write, following the same CLT arguments as before:

$
  tilde(cal(E))_t (w^(a=1)) & = (EE_(x) thin e^(-beta sum_a cal(L)_(1) (w^a, x))EE_(z_(a=1)|x) [sigma(z_(a=1))Theta(x_1)+(1-sigma(z_(a=1)))Theta(-x_1)])/(EE_(x) e^(-beta sum_a cal(L)_(1) (w^a, x))) \
  & = (1/2 sum_x_1 EE_(\{mu\}_a~cal(N)(0,q)) thin e^(-beta sum_a ell(mu_a, q_(a a), x_1)) EE_(z_(a=1)) [sigma(z_(a=1))Theta(x_1)+(1-sigma(z_(a=1)))Theta(-x_1)])/(EE_(\{mu\}_a~cal(N)(0,q)) e^(-beta sum_a ell(mu_a, q_(a a)))) \
  & = (1/2 sum_x_1 EE_(\{mu\}_a~cal(N)(0,q)) thin e^(-beta sum_a ell(mu_a, q_(a a), x_1)) EE_(z_(a=1)) [sigma(z_(a=1))Theta(x_1)+sigma(-z_(a=1))Theta(-x_1)])/(EE_(\{mu\}_a~cal(N)(0,q)) e^(-beta sum_a ell(mu_a, q_(a a)))) \
  & = (EE_(\{mu\}_a~cal(N)(0,q)) thin e^(-beta sum_a ell(mu_a, q_(a a))) EE_(z_(a=1)) sigma(z_(a=1)))/(EE_(\{mu\}_a~cal(N)(0,q)) e^(-beta sum_a ell(mu_a, q_(a a)))) \
$

Were we used the symmetry around $x_1$. We can plug in the RS ansatz. The numerator is:
$
  & integral product_a (dif mu_a)/sqrt(2 pi) e^(1/2[sum_a mu_a^2 1/(delta q)-sum_(a b)mu_a mu_b q/(delta q (delta q +n q))]-1/2 log [delta q^(n-1)(delta q +n q)]-beta sum_a ell(mu_a, q+delta q)) EE_z_(a=1) sigma(z_(a=1))= \
  = & sqrt(delta q^(n-1)(delta q + n q)) integral D x [ integral (dif mu)/sqrt(2 pi) e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q))]^(n-1)\
  & integral (dif mu_1)/sqrt(2 pi) e^( -1/(2 delta q) mu_1^2 + sqrt(q/(delta q (delta q + n q))) x mu_1 -beta ell(mu_1, q+ delta q)) EE_z_(a=1 | mu_1) sigma(z_(a=1))\
  approx & sqrt(delta q^(n-1)(delta q + n q)) integral D x (integral (dif mu_1)/sqrt(2 pi) e^( -1/(2 delta q) mu_1^2 + sqrt(q/(delta q (delta q + n q))) x mu_1 -beta ell(mu_1, q+ delta q)) EE_z_(a=1 | mu_1) sigma(z_(a=1)))/(integral (dif mu)/sqrt(2 pi) e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q)))
$

The denominator is the same without the $EE_z_(a=1)$ term. Therefore we have
$
  tilde(cal(E))_t (q, delta q) & = integral D x ( integral dif mu thin
  e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q)) EE_y sigma(sqrt(t(1-t)q)y + (1-t)mu))/(integral dif mu thin e^( -1/(2 delta q) mu^2 + sqrt(q/(delta q (delta q + n q))) x mu -beta ell(mu, q+ delta q))).
$

Notice that we wrote everything as a function of $q$ and $delta q$ and used the RS ansatz, the whole equation being:

$
  cal(E)_t & = lim_(beta, L -> infinity) lim_(n->0) integral dif q dif delta q thin e^(-n beta L phi(q, delta q)) tilde(cal(E))_t (q, delta q).
$


We expect the $tilde(cal(E))_t$ therm to be $cal(O)(1)$ and to not be able to shift the SP. So in the limits we will just have $cal(E)_t & = tilde(cal(E))_t (q_*, delta q_*)$. We just need to compute the low temperature limit of the latter.

$
  cal(E)_t& = lim_(beta -> infinity) integral D x thin (integral dif mu thin
  e^(beta [ - 1/(2 delta q_*) mu^2 + sqrt(q_*)/(delta q_*)x mu - ell (mu,q_*)]) EE_y sigma(sqrt(t(1-t)q)y + (1-t)mu))/(integral dif mu thin e^( beta[- 1/(2 delta q_*) mu^2 + sqrt(q_*)/(delta q_*)x mu - ell (mu,q_*)])) =\
  & = integral D x thin D y thin sigma(sqrt(t(1-t)q_*)y + (1-t)mu^*(x))
$<eq:accuracy_t>

where as before $mu^*(x)$ is the argmax of the exponent.

=== Greedy Algorithm

The greedy sampling scheme is recovered by replacing $sigma$ with $Theta$. The  accuracy is then simplified as follows:

$
  EE_y Theta(sqrt(t(1-t)q)y + (1-t)mu(x)) = 1-H(mu/sqrt(q) sqrt((1-t)/t)) = Phi(mu/sqrt(q) sqrt((1-t)/t)).
$

so finally:
$
  cal(E)_t & = integral D x thin Phi((mu^*(x))/sqrt(q_*) sqrt((1-t)/t)).
$

#figure(
  grid(
    columns: 2,
    image("plots/one-shot/lambda001/oneshot_accuracy_0p01_alg_greedy_x_0p0_3p1_nt_11.png"),
    image("plots/one-shot/lambda0/oneshot_accuracy_0p0_alg_greedy_x_0p0_3p1_nt_11.png"),

    image("plots/one-shot/lambda001/m_i_vs_m_f_λ=0.01_alg_greedy_alpha_0p05_8p0.png"),
    image("plots/one-shot/lambda0/m_i_vs_m_f_λ=0.0_alg_greedy_alpha_0p05_8p0.png"),
  ),
  caption: [Final magnetization $m_f$ for one-shot unmasking in the greedy algorithm using the saddle points of before compared to the experimental values. $lambda=0.01$ (left) and $lambda=0$ (right). ],
)

In @app:accuracy_t0 we develop an upper bound for this expression of the greedy accuracy at $t=0$.

=== Fair Algorithm
While numerically $lambda$ had little impact on the greedy sampling scheme, sing the fair sampling scheme the strongest memorization is achieved for $lambda -> 0$. Indeed at $alpha=0$ we have that$q_*=0$ and thus $hat(mu)_* = mu_*(alpha=0)$ does not depend on $x$. In this case:

$
  cal(E)_t & = sigma((1-t)hat(mu)_*)= 1/(1+e^(-(1-t)hat(mu)_*)) approx 1 - e^(-(1-t)hat(mu)_*).
$

It can be shown $hat(mu)_* -> infinity$ as $lambda->0$. At $alpha=0$ we have (@app:accuracy_t0) $delta q = 1\/lambda$. Then the saddle point equation for $hat(mu)_*$ reads:

$
  hat(mu)_* & = -delta q_* partial_mu ell(hat(mu)_*, 0) = 1/lambda [1/2- integral_0^1 dif t thin sigma((1-t)hat(mu)_*)(1-t)].
$

The argument in square brackets (let's call it $F(hat(mu)_*)$) is a decreasing function of $hat(mu)_*$ that maps $[-infinity, +infinity]$ to $[0, 1\/2]$. Thus, the solution to $lambda mu_* = F(mu_*)$ diverges for $lambda -> 0$.

#figure(
  grid(
    columns: 2,
    image("plots/one-shot/lambda001/oneshot_accuracy_0p01_alg_fair_x_0p0_3p1_nt_11.png"),
    image("plots/one-shot/lambda0/oneshot_accuracy_0p0_alg_fair_x_0p0_3p1_nt_11.png"),

    image("plots/one-shot/lambda001/m_i_vs_m_f_λ=0.01_alg_fair_alpha_0p05_8p0.png"),
    image("plots/one-shot/lambda0/m_i_vs_m_f_λ=0.0_alg_fair_alpha_0p05_8p0.png"),
  ),
  caption: [Final magnetization $m_f$ for one-shot unmasking in the fair algorithm using the saddle points of before compared to the experimental values. $lambda=0.01$ (left) and $lambda=0$ (right). ],
)


=== Probability of reconstructing a full pattern

The probability of making a correct prediction at a certain $t$ is  $cal(E)_t$ , so that, in $L$ total steps, the probability of perfectly reconstructing all coordinates of some original $x^mu$ in any order is $P(hat(x)=x^mu) = cal(E)_(1) cal(E)_(1-1\/L) thin cal(E)_(1-2\/L) dots thin cal(E)_(1\/L) = product_(l=1)^L cal(E)_(l\/L)$.
Since $cal(E)_(t)$ is a decreasing function of $t$, we have the bounds:

$
  cal(E)_(1)^(L) <= P(hat(x)=x^mu) <= cal(E)_(1\/L)^L.
$
Now suppose we start at time $t$ with an $m_t=1-t$ fraction of correctly unmasked tokens. We have only $L t$ steps left, and the bound becomes:
$
  cal(E)_(t)^(L t) <= P(hat(x)=x^mu| m_(t)=1-t) <= cal(E)_(1\/L)^(L t).
$

We wonder when is it satisfied that, for a given precision $delta$:
$
  P(hat(x)=x^mu| m_(t)=1-t) >= cal(E)_(t)^(L t) >= 1- delta.
$

Let's now look at the greedy algorithm case. Define $x_0 = - delta q_*\/4sqrt(q_*)$: it is the point such that $mu(x_0)=0$. Indeed $mu_*(x_0)$ satisfies $x_0 =( delta q_* )/sqrt(q_*) partial_mu ell (mu_*,q)|_(mu_*=0)$. Then:
$
  partial_mu ell (mu_*, q_*)|_(mu_*=0) = -integral_0^1 dif t (1-t) [1-integral D_y sigma(sqrt(t(1-t)q_*)y)] = -1/2 integral_0^1 dif t (1-t) = -1/4.
$

It can be shown $mu'_* (x) >= sqrt(q_*)/(1+delta q_*\/12)=c_*$. Indeed, differentiating the implicit equation for $mu_*(x)$ we get

$
  mu'_*(x) = sqrt(q_*) -delta q_* partial_(mu mu) ell (mu_*, q_*) mu'_*(x) -> mu'_*(x) = sqrt(q_*)/(1+delta q_* partial_(mu mu) ell (mu_*, q_*)).
$
We bound $partial_(mu mu) ell (mu, q)$ as follows:
$
  partial_(mu mu) ell (mu, q) & = integral_0^1 D y dif t thin (1-t)^2 sigma(sqrt(t(1-t)q)y+(1-t)mu)(1-sigma(sqrt(t(1-t)q)y+(1-t)mu)) \
  & <= integral_0^1 dif t thin (1-t)^2 integral D y sigma(y)(1-sigma(y))<= integral_0^1 dif t thin (1-t)^2 1/4 <= 1/12.
$

We can thus write in the $x>=x_0$ region:

$
  mu_*(x) >= c_* (x-x_0)
$

Therefore, in the greedy case we have
$
  cal(E)_t & >= integral_(x >= x_0) D x D y thin Theta(sqrt(t(1-t)q_*)y + (1-t)c_* (x - x_0))) \
$

where we dropped a nonnegative integral for ther region $x<x_0$ where the linear bound does not hold. Then:
$
  cal(E)_t & >= PP(sqrt(t(1-t)q_*)Y + (1-t)c_* (X - x_0) >= 0, X>=x_0) \
           & >= PP(sqrt(t(1-t)q_*)Y + (1-t)c_* (X - x_0) >= 0) - PP(X<=x_0) \
           & = Phi((-(1-t)c_* x_0)/( sqrt(t(1-t)q_*+(1-t)^2c_*^2))) - Phi(x_0) \
           & = Phi((delta q_*)/(4sqrt(q_*)) 1/sqrt(1+t/(1-t)(1+(delta q_*)/12)^2)) - Phi(-(delta q_*)/(4sqrt(q_*)))
$

From the saddle point equations and the definition of $mu(x)$ we have
$
  partial_(delta q) tilde(G)_E = 1/2 integral D x ((mu_*(x))/(delta q) -x sqrt(q)/(delta q))^2 =1/2 integral D x (partial_mu ell(mu_*(x), q))^2
$

but $q = 2 alpha delta q_*^2 partial_(delta q) tilde(G)_E$ and thus, since $partial_mu ell(mu, q) <=1/2$, we have

$
  (delta q_*) / sqrt(q_*) >= 2/sqrt(alpha)
$
so that

$
  cal(E)_t >= Phi(1/(2sqrt(alpha)) 1/sqrt(1+t/(1-t)(1+(delta q_*)/12)^2))-Phi(-1/(2sqrt(alpha))).
$
Up until this point we obtained a bound for $t>0$ which is rigorous provied the saddle point equations for the order parameters are correct. We continue with handwavy arguments to find a relationship between $M$ and $L$ in order to reconstruct a pattern with precision $delta$. The first step is to realise that the $delta q_* \/ 12$ bound is way too lose, and the quantity $delta q_* partial_(mu mu) ell$ is usually of order $O(1)$ even if $delta q_*$ diverges. This can bee seen numerically (or proven in an appendix).

Then, we are requiring:

$
  Phi(1/(2sqrt(alpha)) c_1/sqrt(1+t/(1-t)c_2)) - Phi(-1/(2sqrt(alpha))) >= (1-delta)^(1/(L t)).
$

Call $kappa_t=c_1\/sqrt(1+t/(1-t)c_2)$. For small $alpha$ we have two exponential tails, and we can impose $1-e^(-c_t/alpha)>= (1-delta)^(1/(L t))$ where $c_t$ depends on $t$, $c_1$, $c_2$ but not on $L$. taking the log:

$
  -c_t/alpha >= log(1-(1-delta)^(1/(L t))) approx log(1-e^(-delta/(L t))) approx log(delta/(L t))
$

so plugging $alpha=M/L$

$
  M <= c_t L/ log((L t)/delta)
$

which means $M=O(L/log(L))$.

== Time-integrated Accuracy approximation

=== Definition and Mean-field ODE
Having obtained an expression for the time-sliced accuracy, $cal(E)_t$, we woul like to obtain an approximated expression for the the overlap of the generated sample $hat(x)^mu$ when starting the generation from a partially masked training example $x^mu$ with masking fraction $t$, that is $x^(mu,t)$.

Assume that at time $t$ the we have a fraction $t$ of masked positions,
and the unmasked positions could be either correct or incorrect with respect to the original training example $x^mu$. We denote with $c_t$ the fraction of correctly unmasked positions, and with $e_t$ the fraction of incorrectly unmasked positions, so that the total masking fraction is $t = 1 - c_t - e_t$.

Our next goal is to generalilse $cal(E)_t$ to $cal(E)_t (c_t, e_t)$: that is the probability that the model correctly predicts a position given that at time $t$ the fraction of correctly unmasked positions is $c_t$ and the fraction of incorrectly unmasked positions is $e_t$.
We can compute this quantity by adapting the previous calculation, and we will recover $cal(E)_(t)=cal(E)_t (c_t=1-t, e_t=0)$.


Neglecting correlations developed during the generation process, we can write the following approximate equations for the evolution of $c_t$ and $e_t$ during the generation process in the high-dimensional limit:
$
  -(dif c_t) / (d t) & = cal(E)_t (c_t, e_t) \
    -(d e_t) / (d t) & =1 - cal(E)_t (c_t, e_t)
$

// TODO change name of accuracy cal(E), because that name is appriopriate for error rather than accuracy

This is to be integrate from a certain time $t$ down to 0, with initial conditions $c_t = 1 - t$, $e_t = 0$.

=== Evaluation of $cal(E)_(t) (c_t, e_t)$

The new expression for the generalised accuuracy will be
$
  cal(E)_t (c_t, e_t)&= EE_X lr(chevron.l EE_(C_t, E_t) thin II(x_i^(mu=1) = hat(x) ~ "Bernoulli"(sigma(z_c - z_e))) chevron.r)
$
where now the argument of the $sigma$ is
$
  z_c- z_e = sum_(j in C_t) w_j/sqrt(L) x_j^(mu=1)-sum_(j in E_t) w_j/sqrt(L) x_j^(mu=1).
$

The average is performed over all sets $C_t$ and $E_t$ of correctly and incorrectly unmasked positions at time $t$, of size $|C_t| = L c_t$ and $|E_t| = L e_t$ such that $c_t+e_t=1-t$. We work in a proportional regime, so that even if for example $e_t<<c_t$ we'll still assume that $L e_t$ is large enough to apply CLT arguments. We rewrite

$
  z_c-z_e = sum_(j) s_j w_j/sqrt(L) x_j^(mu=1)
$

Were $s_j$ is a random variable that takes value $1$ if $j in C_t$, value $-1$ if $j in E_t$ and value $0$ otherwise.
Of course one has $EE s_j = c_t-e_t$. Moreover $EE(s_j^2) = c_t+e_t = 1-t$ thus $"Var"(s_j)=1-t-(c_t-e_t)^2$.
It can be shown that the covariance for $i != j$ is of subleading order. In conclusion, applying CLT,
$
  z_c - z_e tilde cal(N)((c_t - e_t) mu, ((1-t)-(c_t-e_t)^2)q_d)
$
conditioned on $x$ and $t$.
As a sanity check, we recover the previous result for the (non generalised) accuracy for $e_t=0$. Moreover, if we were to consider the random variablle $z_c+z_e$ we should get the previous result for the unmasked positions only, that is mean $(1-t) mu$ and variance $q_d (1-t) t$. Indeed one has that the mean is $(c_t + e_t) mu = (1-t) mu$ and the variance is $q_d (c_t (1-c_t) + e_t (1-e_t) - 2 c_t e_t) = q_d ((c_t + e_t)(1 - (c_t + e_t))) = q_d (1-t) t$.

We can inherit the previous computation for the accuracy leading to @eq:accuracy_t. This gives us

$
  cal(E)_t (c_t, e_t)= integral D x thin D y thin sigma(sqrt(q_* ((1-t)-(c_t-e_t)^2))y + mu_*(x) (c_t-e_t)).
$ <eq:gen_accuracy_t>

Notice that the generalised accuracy for at time $t$ with fraction of correct and errored fractions $c_t$ and $e_t$ is equivalent to the previous accuracy $cal(E)_tilde(t)$ at a rescaled time [not sure this is needed]:

$
  (c_t - e_t)/sqrt((1-t)-(c_t-e_t)^2) = sqrt(1- tilde(t)) / sqrt(tilde(t)) arrow.r tilde(t) = 1- (c_t-e_t)^2/(1-t).
$

Due to the monotonically decreasing nature of the accuracy as a function of time, and since $tilde(t) in [t, 1]$, we rightfully see that including some errors decreases the overall accuracy.

=== Numerical Integration of the Mean-field ODE

We aim at solving the system of ODEs:

$
  -(dif c_t) / (d t) & = cal(E)_t (c_t, e_t) \
    -(d e_t) / (d t) & =1 - cal(E)_t (c_t, e_t)
$

Since we have that $cal(E)_t (c_t, e_t)=cal(E)_t (c_t-e_t)$ we get that the overlap $m_t = c_t - e_t$ satisfies:
$
  (d m_t) / (d t) & = 1-2 cal(E)_t (m_t)= 1- 2 integral D x thin D y thin sigma(sqrt(q_* ((1-t)-m_t^2))y + m_t mu_*(x)).
$


This ODE can be integrated numerically starting from the initial condition $m_t = 1-t$ at time $t$ down to 0. For instance for $alpha=0.5$ we have the following:


#figure(
  grid(
    columns: 2,
    image("plots/trajectories/trajectories_m_vs_t_alpha_0p5_λ=0.01_greedy.png"),
    image("plots/trajectories/trajectories_m_vs_t_alpha_0p5_λ=0.0_greedy.png"),

    image("plots/trajectories/trajectories_m_vs_t_alpha_0p5_λ=0.01_fair.png"),
    image("plots/trajectories/trajectories_m_vs_t_alpha_0p5_λ=0.0_fair.png"),
  ),
  caption: [Integration of the ODE for the proportion of correctly unmasked positions $c_t$ and incorrectly unmasked positions $e_t$ during the generation process for $alpha=0.5$. The initial condition is $c_t = 1-t$, $e_t=0$ at time $t_0$. We plot the resulting trajectories as a function of time for the greedy (top) and fair (bottom) sampling schemes, and for $lambda=0.01$ (left) and $lambda=0$ (right). Theoretical trajectories are plotted in otted lines, while the empirical trajectories obtained from simulations are plotted in solid lines. ],
)

We notice that the biggest discrepancy is obtained for the greedy sampling scheme. The final overlap when starting from a configuration with masking fraction $t$ is given by $m_f = c_0 - e_0$.

We first plot $m_f$ as a function of $alpha$ and, as before $m_i = 1 - t$ as approximated by the ODE vs the empirical results from simulations.

#figure(
  grid(
    columns: 2,
    image("plots/integrated/lambda001/integrated_accuracy_lambda_0p01_alg_greedy_x_0p0_3p1_nt_11.png"),
    image("plots/integrated/lambda0/integrated_accuracy_lambda_0p0_alg_greedy_x_0p0_3p1_nt_11.png"),

    image("plots/integrated/lambda001/integrated_m_i_vs_m_f_λ=0.01_alg_greedy_alpha_0p05_8p0.png"),
    image("plots/integrated/lambda0/integrated_m_i_vs_m_f_λ=0.0_alg_greedy_alpha_0p05_8p0.png"),
  ),
  caption: [Final magnetization $m_f$ for integrated unmasking in the greedy algorithm using the saddle points of before compared to the experimental values. $lambda=0.01$ (left) and $lambda=0$ (right). ],
)

#figure(
  grid(
    columns: 2,
    image("plots/integrated/lambda001/integrated_accuracy_lambda_0p01_alg_fair_x_0p0_3p1_nt_11.png"),
    image("plots/integrated/lambda0/integrated_accuracy_lambda_0p0_alg_fair_x_0p0_3p1_nt_11.png"),

    image("plots/integrated/lambda001/integrated_m_i_vs_m_f_λ=0.01_alg_fair_alpha_0p05_8p0.png"),
    image("plots/integrated/lambda0/integrated_m_i_vs_m_f_λ=0.0_alg_fair_alpha_0p05_8p0.png"),
  ),
  caption: [Final magnetization $m_f$ for integrated unmasking in the fair algorithm using the saddle points of before compared to the experimental values. $lambda=0.01$ (left) and $lambda=0$ (right). ],
)

In the greedy case, the final magnetization shows a nontrivial behaviour for which the step-by-step recovery helps memorization for low values of $alpha$ with respect to one-shot recovery, while it generally hurts memorization at higher $alpha$.
This effect is less present in the fair sampling scheme, where performance at high $alpha$ is almost equal in both cases.
#figure(
  grid(
    columns: 2,
    image("plots/oneshotvsintegrated/oneshot_vs_integrated_mi_vs_mf_λ_0.01_alg_greedy_nα_9.png"),
    image("plots/oneshotvsintegrated/oneshot_vs_integrated_mi_vs_mf_λ_0.01_alg_fair_nα_9.png"),

    image("plots/oneshotvsintegrated/oneshot_vs_integrated_mi_vs_mf_λ_0.0_alg_greedy_nα_10.png"),
    image("plots/oneshotvsintegrated/oneshot_vs_integrated_mi_vs_mf_λ_0.0_alg_fair_nα_10.png"),
  ),
  caption: [Final magnetization $m_f$ for integrated unmasking in the fair algorithm using the saddle points of before compared to the experimental values. $lambda=0.01$ (left) and $lambda=0$ (right). ],
)

Although this simple approximation fails to capture the correct vales of the final magnetization, it correctly captures the qualitative behaviour of the system.


= Hebbian Learning
Here we try a different route: we assign the weights according to a Hebbian rule, that is $W_(i j) = 1/sqrt(L) sum_(mu=1)^M x_i^mu x_j^mu$ (we again ignore $V_(i j)$ here forward) and look at the probability:

$
  p_"Hebb" (x_i=1|x^t) & = sigma(1 / sqrt(L) sum_(j=1)^L (1/sqrt(L) sum_(mu=1)^M x_i^mu x_j^mu (1-m_j^t) x_j))
$

We are interested in comptuing $cal(E)_t$  that in this context is given (for the fair sampling scheeme) by:
$
  cal(E)_t & = EE_X EE_(m^t) thin II(x_i^(mu=1) = hat(x) ~ "Bernoulli"(p_"Hebb" (x_i=x_i^(mu=1)|x^t=x^(mu=1)(1-m^t)))) \
$
The greedy case can be obtained as always by replacing $sigma$ with $Theta$. We condition on $m_i^t=1$ (otherwise we don't need to unmask). The argument of the sigmoid can be written as
$
  & 1 / sqrt(L) sum_(j!=i)^L (1/sqrt(L) sum_(mu=1)^M x_i^mu x_j^mu (1-m_j^t) x_j^(mu=1)) = \
  & = 1 / L x_i^(mu=1) sum_(j!=i)^L x_j^(mu=1) (1-m_j^t) x_j^(mu=1)+1 / L sum_(j!=i)^L sum_(mu!=1)^M x_i^mu x_j^mu (1-m_j^t) x_j^(mu=1)\
  & =(sum_(j!=i)^L (1-m_j^t))/ L x_i^(mu=1) +1 / L sum_(j!=i)^L sum_(mu!=1)^M x_i^mu x_j^mu (1-m_j^t) x_j^(mu=1)=z_h (m^t) x_i^(mu=1)+ nu_h(m^t, X_(\\1))\
$
Where we explicitly separated the signal and noise components.
The trick here is to recognise that we can apply CLT on the new variables.
Indeed, the signal term is a sum of $L$ i.i.d. random variables with average $t$ (it correspond to the $z$ of the previous case) and the noise term is a sum of $L(M-1)$ i.i.d. random variables averaging to 0. We can write

$
            z_h (m^t) & ~ cal(N)(1-t, t(1-t)/L)->^(L->infinity)1-t \
  nu_h (m^t, X_(\\1)) & ~ cal(N)(0, alpha(1-t)) \
$
So we can write the preactivation as a gaussian variable $z~cal(N)((1-t), alpha(1-t))$

We reduce this average to a single gaussian integral:
$
  cal(E)_t = EE_(z~cal(N)((1-t), alpha(1-t))) sigma(z)= integral D z thin sigma((1-t) + sqrt(alpha(1-t)) z).
$
For the greedy sampling scheme we have even more simply:

$
  cal(E)_t = Phi((1-t)/sqrt(alpha(1-t))).
$

#figure(
  grid(
    columns: 2,
    image("plots/one-shot/hebbian/oneshot_accuracy_hebbian_alg_greedy_x_0p0_3p1_nt_11.png"),
    image("plots/one-shot/hebbian/oneshot_accuracy_hebbian_alg_fair_x_0p0_3p1_nt_11.png"),

    image("plots/one-shot/hebbian/m_i_vs_m_f_hebbian_alg_greedy_alpha_0p05_8p0.png"),
    image("plots/one-shot/hebbian/m_i_vs_m_f_hebbian_alg_fair_alpha_0p05_8p0.png"),
  ),
  caption: [Final magnetization $m_f$ for one-shot unmasking in Hebbian learning compared with experimental values. Greedy (left) and Fair (right) algorithms. ],
)

What we have analised so far is the accuracy at a time $t$ given all the previous unmasking steps have been perfect. If there is a fraction $c_t$ and $e_t$ of correct and errored unmaskings we can use the same argument as before to write the accuracy as a function of $c_t$ and $e_t$, or of their difference $m_t = c_t - e_t$. The argument of the sigmoid is now written as:

$
  & 1 / L [ sum_(j in C_t) sum_(mu=1)^M x_i^mu x_j^mu x_j^(mu=1) - sum_(j in E_t) sum_(mu=1)^M x_i^mu x_j^mu x_j^(mu=1) ] =\
  & = 1 / L [ sum_(j in C_t) x_j^(mu=1) x_j^(mu=1) - sum_(j in E_t) x_j^(mu=1) x_j^(mu=1) ]x_i^(mu=1) + 1 / L sum_(j in C_t) sum_(mu!=1)^M x_i^mu x_j^mu x_j^(mu=1) - 1 / L sum_(j in E_t) sum_(mu!=1)^M x_i^mu x_j^mu x_j^(mu=1)= \
  & -> (c_t - e_t) x_i^(mu=1) + 1 / L sum_(j in C_t) sum_(mu!=1)^M x_i^mu x_j^mu x_j^(mu=1) - 1 / L sum_(j in E_t) sum_(mu!=1)^M x_i^mu x_j^mu x_j^(mu=1)= \
$

Following the same argumets of the past chapter, the gaussian variable will now have mean $(c_t-e_t) = m_t$ and variance
$alpha(1-t)$:

$
  cal(E)_t (m_t) = integral D z thin sigma(m_t + sqrt(alpha(1-t)) z).
$
which in the greedy case becomes:
$
  cal(E)_t (m_t) = Phi(m_t/sqrt(alpha(1-t))).
$

The trajectories can be obtained by integrating the ODE for $m_t$:

#figure(
  grid(
    columns: 2,
    image("plots/trajectories/trajectories_m_vs_t_alpha_0p5_hebbian_greedy.png"),
    image("plots/trajectories/trajectories_m_vs_t_alpha_0p5_hebbian_fair.png"),
  ),
  caption: [Integration of the ODE for the proportion of correctly unmasked positions $c_t$ and incorrectly unmasked positions $e_t$ during the generation process for $alpha=0.5$ in the Hebbian learning case.],
)

As before, we can integrate the ODE for $m_t$ to get the final magnetization $m_f$ as a function of the initial magnetization $m_i = 1-t$. We can then compare these theoretical predictions with the experimental values obtained from simulations.


#figure(
  grid(
    columns: 2,
    image("plots/integrated/hebbian/integrated_accuracy_lambda_hebbian_alg_greedy_x_0p0_3p1_nt_11.png"),
    image("plots/integrated/hebbian/integrated_accuracy_lambda_hebbian_alg_fair_x_0p0_3p1_nt_11.png"),

    image("plots/integrated/hebbian/integrated_m_i_vs_m_f_hebbian_alg_greedy_alpha_0p05_8p0.png"),
    image("plots/integrated/hebbian/integrated_m_i_vs_m_f_hebbian_alg_fair_alpha_0p05_8p0.png"),
  ),
  caption: [Final magnetization $m_f$ for integrated unmasking in Hebbian learning compared with experimental values. Greedy (left) and Fair (right) algorithms. ],
)


= Experiments on Random Data

We train the model using pytorch and pytorch lightning. Each model is trained for at least 5000 epochs, using batch sizes of 512 and the AdamW optimizer.


== U-Turn Memorization

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

== Full Memorization

We now check instead if the model generates samples close to the training point when starting from a configuration with all positions masked, that is $t=1$.

In this experiment, for each generated sample $hat(x)$, we compute the overlap with all training examples, and we consider the top 3 overlaps. We then average over all generated samples.

As shown in @fig:fullmem_linear_uniform, we see that the model is not able to fully memorize the training set, since the top overlap is significantly smaller than 1 and
dicreases with $L$. Nonetheless, the top overlap is significantly larger than what would be obtained by random guessing, which is of order $sqrt(2log(M)/L)$. Therefore, at small and medimum data size, the model partially memorizes the training set when $alpha$ is small enough.

We conjecture that the for $M=O(L/log(L))$ we would observe at the same time perfect retrieval and full memorization.
TODO EXPERIMENTS.

#figure(
  grid(
    columns: 2,
    image("plots/top3_overlaps_linear_uniform_alpha0.1_l2reg0.0.png"),
    image("plots/top3_overlaps_linear_uniform_alpha0.5_l2reg0.0.png"),
  ),
  caption: [U-Turn memorization for uniform data and all times. On the $x$-axis, we have the initial time for the reverse process. This corresponds
    to an initial overlap $1-t$. If there is no memorization at all, $1-t$ would also be the final overlap (diagonal line). Data size ],
) <fig:fullmem_linear_uniform>


= Experiments on Real  Datasets
We occlude 50% of the pixels (the bottom half of the image) and use the model to reconstruct the original image.
We vary the training set size $M$ to see how it affects the reconstruction accuracy.

== Binarized MNIST


We compute the reconstruction accuracy in terms of hamming distance.

#figure(
  image("plots/mnist_train_reconstr_alpha0.1.png"),
  caption: [MNIST training point reconstruction starting from masked bottom half. Training set size $0.1 * 28*28$.
    (top) Original images. (middle) Masked images. (bottom) Reconstructed images.],
) <fig:mnist_reconstr>


== Fashion MNIST

Grey-scale dataset. We now use the model:

$
  ...
$


We compute the reconstruction accuracy in terms of hamming distance and MSE.


= Constraint Satisfaction Problem Formulation
== Definition
$
  Z = integral d mu(w) thin product_(mu=1)^M II(integral_0^gamma dif t thin EE_(m_t)[Theta(1/sqrt(L) sum_i w_i x_i^(mu) (1-m_i^t))] <= epsilon)
$

We compute

$
  phi = lim_(epsilon->0) lim_(L->infinity) 1/L log Z
$

Than we derive the capacity line as $alpha_c (gamma, epsilon)$.

== RS Computation
We can write the constraints as



$
  II(integral_0^gamma dif t thin EE_(m_t)[Theta(1/sqrt(L) sum_i w_i x_i^(mu) (1-m_i^t))] <= epsilon) = II(integral_0^gamma dif t thin H(mu_t/sqrt(q_t)) <= epsilon)
$



// $
//   II(integral_0^gamma dif t thin  EE_(m_t)[Theta(1/sqrt(L)  sum_i w_i x_i^(mu) (1-m_i^t))] <= epsilon ) = integral (dif u dif hat(u))/(2pi) II(u <= epsilon ) e^(i u hat(u) -i hat(u)integral_0^gamma dif t thin  EE_(m_t)[Theta(1/sqrt(L)  sum_i w_i x_i^(mu) (1-m_i^t))])
// $




= Discussion and Conclusions

It would be interesting to extend the analysis to uniform discrete diffusion CITE, so that instead of erasure we can handle bit flips corruption.

Extending the theoretical analysis to arbitrary size vocabulary would also be interesting.

#bibliography("bibliography.bib")


#show: arkheion-appendices

= Other U-Turn Plots

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
== Upper bound on accuuracy at $t=0$
<app:accuracy_t0>

Let's look at $cal(E)_0$, the (greedy) accuracy at $t=0$. We look at the $alpha->0$ regime. One has:

$
  cal(E)_0 (q_*, delta q_*) & = lim_(t->0) integral D_x Phi((mu_*(x)) /sqrt(q_*) sqrt((1-t)/t)) = \
                            & = integral D_x Theta(mu_*(x)).
$

Notice that here $mu_*(x)$, $q_*$ and $delta q_*$ all depend implicitly on $alpha$.
We can compute when $mu_*(x)$ changes sign. One has $mu_*(x)=0$ if

$
  x & = (delta q_*)/(sqrt(q_*)) partial_mu ell (0, q_*) = \
    & = - (delta q_*)/(sqrt(q_*)) integral_0^1 dif t (1-t) [1-integral D_y sigma(sqrt(t(1-t)q_*)y)] \
    & = -(delta q_*)/(sqrt(q_*)) 1/2 integral_0^1 dif t (1-t) = -(delta q_*)/(sqrt(q_*)) 1/4.
$

And thus, due to the monotonicity of $mu_*(x)$ in $x$, $cal(E)_0=Phi(1/4 (delta q_*)/(sqrt(q_*)))$. Using
@eq:SP, we have an expression for $delta q_* \/ sqrt(q)$:

$
  (delta q_*)/(sqrt(q_*)) & = 1 / (sqrt(2 alpha partial_(delta q) tilde(G)_E)).
$ <eq:deltaqoversqrtq>

We rewrite

$
  partial_(delta q) tilde(G)_E = integral D_x [(mu_*^2)/(2 delta q_*^2)- (x mu_* sqrt(q_*))/(delta q_*^2)] +1/2 (q_*)/(delta q_*^2) = 1/2 integral D_x ((mu_*(x))/(delta q_*) -x sqrt(q_*)/(delta q_*))^2.
$
It can be cheked numerically (forse si può anche dimostrare?) that $partial_(delta q) tilde(G)_E$ is a decreasing function of $alpha$, so we bound it with its value at $alpha=0$:

$
  2 partial_(delta q) tilde(G)_E >= lim_(alpha->0) 2 partial_(delta q) tilde(G)_E (alpha) = integral D_x ((mu_*(x))/(delta q_*))^2 = (hat(mu)_* lambda)^2
$

where $hat(mu)_* := mu_*(alpha=0)$ that clearly does not depend on $x$, and we used that at $alpha=0$ one has $q=0$ and $delta q_* = 1\/lambda$. So we have

$
  (delta q_*)/(sqrt(q_*)) <= 1 / (hat(mu)_* lambda sqrt(alpha) ).
$

It would be useful to remove the dependence on $hat(mu)_*$, as itself only depends on $lambda$. We can use the saddle point equation at $alpha=0$ to get an expression for $hat(mu)_*$, as we did previously:

$
  hat(mu)_* lambda &= 1/2 - integral_0^1 dif t (1-t) sigma((1-t) hat(mu)_*) = 1/2 - integral_0^1 dif u thin u thin sigma(u hat(mu)_*) = \
  & = 1/2 - integral_0^1 dif u thin u thin (1-sigma(-u hat(mu)_*)) = integral_0^1 dif u thin u thin sigma(-u hat(mu)_*)
$

where we performed the change of variable $u = (1-t)$. We wish to express an upper bound on this quantity. We write:
$
  hat(mu)_* lambda &= integral_0^1 dif u thin u thin sigma(-u hat(mu)_*) &= integral_0^1 dif u thin u/(1+e^(u hat(mu)_*)) = 1/hat(mu)_*^2 integral_0^hat(mu)_* dif v thin v/(1+e^(v)).
$

The integral is a monotonically increasing function of $hat(mu)_*$ in $[0, pi^2\/12]$. Thus for $hat(mu)_*$ large enough (a posteriori it will correspods to $lambda$ small enough) one always finds some constant $C$ (from now on this constant will absorb all operatios that leave it unchanged) for which $hat(mu)_* lambda >= C\/hat(mu)_*^2$. From this we get $hat(mu)_*^3 >=C\/lambda$ and finally $hat(mu)_* >= C\/lambda^(1\/3)$. Plugging this back into the expression for $delta q_*\/sqrt(q_*)$ we get

$
  (delta q_*)/(sqrt(q_*)) <= C/(lambda^(2\/3) sqrt(alpha)).
$

The accuracy can be bouded by monotonicity with:
$
  cal(E)_0 <= Phi(C/(lambda^(2\/3) sqrt(alpha))).
$

= Robust Spherical Perceptron with Coordinate Erasures

This is a version of the pattern recovery under masking, robust to adversarial masking.
The capacity though, is only $O(1)$ so we don't pursue this version further.

The finite capacity can be argued for by looking at the annealed computation. Define the single-pattern acceptance probability
$
  p(w) := EE_x Theta(min_(U subset.eq {1, dots, L}, |U| >= R) 1/sqrt(L) sum_(i in U) w_i x_i).
$
Since the $M$ patterns are i.i.d., the expectation over $X$ factorises:
$
  EE_X Z = integral dif w thin delta(||w||^2 - L) product_(mu=1)^M EE_(x^mu) Theta(dots.h) = integral dif w thin delta(||w||^2 - L) p(w)^M .
$

*Bounding $p(w)$.* We can make a large deviation argument to show that $p(w)$ is exponentially small in $L$.

*Sphere volume.* The surface area of $||w||^2 = L$ in $RR^L$ is, by Stirling's approximation,
$
  integral dif w thin delta(||w||^2 - L) = (2 pi^(L/2)) / Gamma(L/2) L^((L-1)/2) = e^((L/2) log(2 pi e) + O(log L)) = e^(O(L)).
$

Combining,
$
  EE_X Z = underbrace(integral dif w thin delta(||w||^2 - L), e^(O(L))) dot underbrace(p(w)^M, e^(-Omega(M L))) = e^(O(L) - Omega(M L)),
$
which vanishes for any $M = omega(1)$, confirming that the capacity is $O(1)$.



== Definition

We consider $M$ i.i.d. random Ising patterns
$
  x^mu = (x_1^mu, dots, x_L^mu) in {-1, +1}^L, quad mu = 1, dots, M,
$
with independent unbiased coordinates
$
  PP(x_i^mu = +1) = PP(x_i^mu = -1) = 1/2.
$

Our goal is to find a weight vector
$
  w = (w_1, dots, w_L) in RR^L
$
satisfying the spherical normalization
$
  ||w||_2^2 = sum_(i=1)^L w_i^2 = L,
$
such that, for every pattern $mu$, the following robust positivity condition holds:
$
  min_(U subset.eq {1, dots, L}, |U| >= R) sum_(i in U) w_i x_i^mu > 0.
$
We are interested in the regime
$
  R = gamma L, quad gamma in (0, 1),
$
with $L -> infinity$ and $gamma$ fixed.

This section gives:
+ a mathematically precise formulation of the problem;
+ an exact derivation of the inner minimization in several equivalent forms;
+ the correct scaling of the robust field under the spherical normalization $||w||^2 = L$;
+ a Gardner-friendly reformulation suitable for a replica calculation.


== Precise formulation of the robust feasibility problem

Fix $L in NN$, $M in NN$, and an integer $R in {1, dots, L - 1}$. Given patterns
$
  Xi = {x^mu}_(mu=1)^M subset {-1, +1}^L,
$
define, for each $w in RR^L$, the robust raw field on pattern $mu$ by
$
  g^mu (w) := min_(U subset.eq {1, dots, L}, |U| >= R) sum_(i in U) w_i x_i^mu.
$ <eq:def-g-set>

The robust feasibility problem is:

*Problem (robust spherical perceptron with erasures).*
Find $w in RR^L$ such that
$
  sum_(i=1)^L w_i^2 = L
$ <eq:spherical-constraint>
and
$
  g^mu (w) > 0 quad "for all" mu = 1, dots, M.
$ <eq:robust-feasibility>

Equivalently, introducing the binary selectors
$z_i^mu in {0, 1}$,
with $z_i^mu = 1$ meaning that coordinate $i$ is kept in the subset, we can rewrite @eq:def-g-set as
$
  g^mu (w) = min_(z^mu in {0, 1}^L, sum_(i=1)^L z_i^mu >= R) sum_(i=1)^L z_i^mu w_i x_i^mu.
$ <eq:def-g-binary>

Thus the condition @eq:robust-feasibility becomes
$
  min_(z^mu in {0, 1}^L, sum_i z_i^mu >= R) sum_(i=1)^L z_i^mu w_i x_i^mu > 0, quad mu = 1, dots, M.
$ <eq:robust-feasibility-binary>

== Exact solution of the inner minimization

=== Reduction to a deterministic combinatorial problem

Fix a single pattern $mu$ and write
$
  h_i := h_i^mu = w_i x_i^mu.
$
Then the inner problem is
$
  phi_R (h) := min_(z in {0, 1}^L, sum_(i=1)^L z_i >= R) sum_(i=1)^L z_i h_i.
$ <eq:phiR-binary>

We will derive an exact explicit solution.

=== Order-statistics representation

Let
$
  h_((1)) <= h_((2)) <= dots <= h_((L))
$
denote the nondecreasing rearrangement of $(h_1, dots, h_L)$, and let
$
  n_- (h) := \#{i : h_i < 0}
$
be the number of negative coordinates.

*Proposition.*
For every $h = (h_1, dots, h_L) in RR^L$,
$
  phi_R (h) = sum_(j=1)^(K(h)) h_((j)), quad K(h) := max{R, n_- (h)}.
$ <eq:phiR-order>

_Proof._ To minimize the sum, one should always include every negative $h_i$, because adding a negative term can only decrease the objective.

Thus:
- if the number of negative terms is at least $R$, then the minimizer includes all negative terms and no positive terms, so the minimum is the sum of all negative entries;
- if the number of negative terms is smaller than $R$, then after including all negative terms one still must choose enough additional coordinates to reach cardinality $R$, and to minimize the total sum one chooses the smallest nonnegative entries.

Therefore the minimizing set is precisely the set of the $K(h)$ smallest coordinates, where
$K(h) = max{R, n_- (h)}$.
This yields @eq:phiR-order. $square$

Applying this to $h_i^mu = w_i x_i^mu$, we get
$
  g^mu (w) = sum_(j=1)^(K^mu (w)) h_((j))^mu, quad K^mu (w) = max{R, n_-^mu (w)},
$ <eq:g-order>
where
$h_((1))^mu <= dots <= h_((L))^mu$
are the ordered values of ${w_i x_i^mu}_(i=1)^L$, and
$n_-^mu (w) = \#{i : w_i x_i^mu < 0}$.

=== Exact LP relaxation and dualization

We now derive a second exact representation, which is especially useful for later analytical work.

Consider again
$
  phi_R (h) = min_(z_i in {0, 1}, sum_i z_i >= R) sum_(i=1)^L z_i h_i.
$

We first relax the integrality constraint:
$
  phi_R^("LP") (h) := min_(0 <= z_i <= 1, sum_i z_i >= R) sum_(i=1)^L z_i h_i.
$ <eq:phiR-lp>

*Proposition.* The relaxation is exact:
$
  phi_R (h) = phi_R^("LP") (h).
$ <eq:exact-relaxation>

_Proof._ The objective is linear in $z$, and the feasible set
$
  cal(P)_R = {z in [0, 1]^L : sum_(i=1)^L z_i >= R}
$
is a polytope. A linear function attains its minimum at an extreme point of the polytope.

The feasible set $cal(P)_R$ is defined by box constraints $0 <= z_i <= 1$ together with one linear constraint $sum_i z_i >= R$ with all-ones coefficient vector. The constraint matrix of this system is totally unimodular (TU): box constraints contribute the identity and its negative, and the single all-ones row preserves total unimodularity. By the TU theorem, since $R$ is an integer, all extreme points of $cal(P)_R$ are integral, i.e., binary.

Hence the LP relaxation has the same value as the original binary problem. $square$

We now dualize @eq:phiR-lp. Rewrite the cardinality constraint as
$R - sum_(i=1)^L z_i <= 0$.
The Lagrangian is
$
  cal(L)(z, lambda) = sum_(i=1)^L h_i z_i + lambda (R - sum_(i=1)^L z_i), quad lambda >= 0.
$ <eq:lagrangian-robust>
Collecting terms gives
$
  cal(L)(z, lambda) = lambda R + sum_(i=1)^L z_i (h_i - lambda).
$

For fixed $lambda$, the minimization over $z$ factorizes:
$
  inf_(0 <= z_i <= 1) z_i (h_i - lambda) = cases(
    0 & "if" h_i - lambda >= 0,
    h_i - lambda & "if" h_i - lambda < 0.
  )
$
Therefore
$
  inf_(0 <= z_i <= 1) z_i (h_i - lambda) = min(0, h_i - lambda).
$ <eq:site-min>

Summing over $i$, the dual function is
$
  D(lambda) = lambda R + sum_(i=1)^L min(0, h_i - lambda).
$ <eq:dual-function>

Since the primal is a feasible linear program, strong duality holds. Thus:
$
  phi_R (h) = max_(lambda >= 0) [lambda R + sum_(i=1)^L min(0, h_i - lambda)].
$ <eq:phiR-dual-min0>

Using $min(0, x) = -(-x)_+$ where $(x)_+ := max(x, 0)$, we get the equivalent form
$
  phi_R (h) = max_(lambda >= 0) [lambda R - sum_(i=1)^L (lambda - h_i)_+].
$ <eq:phiR-dual-hinge>

Applying this to $h_i = w_i x_i^mu$, we obtain the exact representation
$
  g^mu (w) = max_(lambda^mu >= 0) [lambda^mu R - sum_(i=1)^L (lambda^mu - w_i x_i^mu)_+].
$ <eq:g-dual-raw>

=== Direct verification from order statistics

It is instructive to verify @eq:phiR-dual-hinge directly.

Let
$
  F(lambda) := lambda R + sum_(i=1)^L min(0, h_i - lambda).
$
Suppose $lambda$ lies in an interval $h_((k)) <= lambda <= h_((k+1))$
for some $k in {0, dots, L}$, where by convention $h_((0)) = -infinity$ and $h_((L+1)) = +infinity$.

Then exactly $k$ coordinates satisfy $h_i <= lambda$, hence
$
  F(lambda) = lambda R + sum_(j=1)^k (h_((j)) - lambda) = sum_(j=1)^k h_((j)) + lambda(R - k).
$
Thus, on that interval, the slope is $F'(lambda) = R - k$.

Therefore:
- if $k < R$, then $F$ increases with $lambda$;
- if $k > R$, then $F$ decreases with $lambda$;
- if $k = R$, then $F$ is flat.

Taking also into account the restriction $lambda >= 0$, the maximizer is a threshold that selects precisely the appropriate number of smallest coordinates. This reproduces exactly the order-statistics formula @eq:phiR-order.
