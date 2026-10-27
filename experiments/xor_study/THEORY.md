# Linear abstraction of a nonlinear (MLP) SCM

Conventions follow the linear case: $W_{ik}\neq 0$ means an edge $i\to k$; $T\in\mathbb{R}^{n\times m}$ maps low to high ($\tilde X = T^\top X$);
$S$ maps exogenous terms ($\tilde u = S^\top u$). The derivation below takes $T$ block-structured (constructive, disjoint clusters), but (AC') does not need disjointness (see the note after (AC')): each low node lies in at most one cluster $C_j$, and
$T_{ij}\neq 0 \Rightarrow i\in C_j$. For a high-level node set $\Theta$, $\pi(\Theta)=\bigcup_{j\in\Theta} C_j$.
$F_\Theta$ is the diagonal 0/1 selector of $\Theta$ (high level) and $F_{\pi(\Theta)}$ the selector of $\pi(\Theta)$ (low level).

## Low-level model

$$X = \Phi(W^\top X + u), \qquad \Phi(z) = (\varphi_1(z_1),\dots,\varphi_n(z_n)).$$

$\Phi$ acts node-wise and need not be the same at every node. For the MI-identifiability MLP (2-3-3-1):
- $\varphi = $ LeakyReLU($\alpha=0.01$) on hidden nodes;
- $\varphi = $ LeakyReLU on the output node too: their `MLP` intends a linear last layer, but its `idx < len(layer_sizes) - 1` check is always true, so every layer gets the activation;
- input nodes have no parents, so $X_{\text{in}} = \varphi(u_{\text{in}})$ and $u_{\text{in}} = \varphi^{-1}(X_{\text{in}})$ when $\varphi$ is invertible. The simplest choice is $\varphi=\mathrm{id}$ on inputs, so $u_{\text{in}} = X_{\text{in}}$. This also avoids non-invertible $\varphi$ such as ReLU.

For hidden and output nodes, $u$ holds the biases. These are fixed constants $b$ and are never varied or intervened on. Only $u_{\text{in}}$ (the network input) varies.
So "for all $u$" below means for all inputs $x$, with $u_{\text{hid}} = b$ fixed.

## Interventions: put the value *outside* $\Phi$

$$X = (I-F_{\pi(\Theta)})\,\Phi(W^\top X + u) + F_{\pi(\Theta)}\,v .$$

The alternative is to write $\Phi(\dots + F v)$. Then the intervened low nodes take the value $\Phi(v)$ and the high node takes $\tilde\Phi(T^\top v)$.
Consistency on the intervened clusters would then need $T^\top\Phi(v) = \tilde\Phi(T^\top v)$, which fails for nonlinear $\Phi$ unless all entries of $v$ in each cluster have the same sign.
Putting $v$ outside sets post-activation values directly. The high-level value is $T^\top v$, and that term cancels below.

Parametrising the intervention as $X_{\pi(\Theta)} = \Phi(v)$ is equivalent when $\Phi$ is invertible: substitute $v \to \Phi(v)$ throughout.
The high-level intervention must then set $\tilde X_\Theta = T^\top\Phi(v)$, i.e. high-level pre-activation $\tilde\Phi^{-1}(T^\top\Phi(v))$, **not** $\tilde\Phi(T^\top v)$.
The two coincide only when $T^\top\Phi(v)=\tilde\Phi(T^\top v)$. For $\tilde\Phi=\Phi$ positively homogeneous and $T\ge0$, this holds for single-neuron clusters, and for multi-neuron clusters only when all entries of $v$ in the cluster have the same sign.

## Abstraction condition

Let $X$ denote the **post-intervention** low-level solution. The high-level model is acyclic, so its solution is unique. Hence
$T^\top X$ equals the intervened high-level solution iff it satisfies the high-level structural equations:

$$
T^\top\!\left[(I-F_{\pi(\Theta)})\,\Phi(W^\top X + u) + F_{\pi(\Theta)} v\right]
= (I-F_\Theta)\,\tilde\Phi(\tilde W^\top T^\top X + S^\top u) + F_\Theta\, T^\top v
\qquad \forall\, u,\Theta,v. \tag{AC}
$$

Block structure gives $T^\top F_{\pi(\Theta)} = F_\Theta T^\top$. The $v$-terms cancel, and (AC) becomes

$$
(I-F_\Theta)\left[\,T^\top \Phi(W^\top X + u) - \tilde\Phi(\tilde W^\top T^\top X + S^\top u)\right] = 0 . \tag{AC'}
$$

**Note:** $X$ here is the post-intervention solution $X^{\Theta,v}(u)$. The bracket is the same function for every $\Theta$, but the points at which it must vanish depend on $(\Theta,v)$.
Interventions decouple $X$ from $u$, so the observational case $\Theta=\emptyset$ alone is not sufficient.

**Constructiveness is not needed for (AC').** Each $\tau_j$ reads only its cluster, $\tau_j(x) = \tau_j(x_{\pi(j)})$, but clusters may overlap: the same neurons can contribute to several high-level variables. With exogenous map $\gamma$,
$(I-\tilde F_\Theta)\big[\tau(x) - \tilde\Phi(\tilde W^\top\tau(x) + \gamma(u))\big] = 0$ is still the condition.
The intervened rows still hold automatically: for $j\in\Theta$, $\pi(j)\subseteq\pi(\Theta)$, so $\tau_j$ of the intervened state equals $\tau_j(x')$.
Overlap only means that intervening on $\pi(\Theta)$ can also overwrite part of a non-intervened variable's cluster.

With $\Phi=\mathrm{id}$ this is $(I-F_\Theta)[(T^\top W^\top - \tilde W^\top T^\top)X + (T^\top - S^\top)u]=0$, which is the linear case.

## Piecewise-linear, positively homogeneous $\Phi$ (ReLU, LeakyReLU)

Write $\Phi(z) = D(z)\,z$ with $D(z)$ diagonal. Each entry is the slope of that node's active piece: $\{\alpha, 1\}$ for LeakyReLU and $1$ for identity nodes (inputs). Let
$z = W^\top X + u$ and $\tilde z = \tilde W^\top T^\top X + S^\top u$. Then (AC') reads $T^\top D(z)\,z = \tilde D(\tilde z)\,\tilde z$ on non-intervened high nodes.

**Sufficient conditions** for every context $X$ that is reached:

1. **Pattern commutation:** $T^\top D(z) = \tilde D(\tilde z)\,T^\top$. Every low node with $T_{ij}\neq 0$ has the same slope as high node $j$, i.e. clusters switch on and off together.
2. **Linear condition on the reached contexts:** $(T^\top W^\top - \tilde W^\top T^\top)X + (T^\top - S^\top)u = 0$, i.e. $T^\top z = \tilde z$.

If $T\ge 0$ on hidden clusters, condition 2 together with cluster-homogeneous signs forces $\operatorname{sign}\tilde z_j$ to equal the cluster's sign. So $\tilde D$ is automatically the
pattern the high-level model produces itself, and there is nothing further to check.

**Under all interventions** ($v$ ranging over all reals, so parent-layer $X$ is arbitrary), the conditions are also necessary:
- Condition 2 must hold for every $X$, which gives $T^\top W^\top = \tilde W^\top T^\top$ (the linear commutation), and $T^\top u = S^\top u$ on the bias part.
- Condition 1 must hold for every $z$. A sum $\sum_i t_i\varphi(w_i^\top x + b_i)$ has kinks on each hyperplane $w_i^\top x + b_i = 0$, and $\tilde\varphi(\cdot)$ has a single kink. So every neuron in a cluster
  must share a hyperplane with the same orientation: $(w_i, b_i) = c_i (w, b)$ with $c_i > 0$ (given $T\ge 0$).

In other words, exact abstraction under all interventions only holds for **neuron splitting**: a cluster made of positive rescalings of one neuron. Trained nets do
not have this structure, so exact abstraction requires restricting the intervention set.

**Restricted (interchange) interventions.** Take $v$ only from values the network reaches on the data. For XOR, that means activations at the 4 input corners, as in the paper's IIA.
- The set of contexts is finite, and each context has a fixed pattern $D$.
- Conditions 1 and 2 become finitely many linear equations, one block per context. They have the same bilinear structure as the linear case but are restricted to the span of the reached $X$.
- That span is small, which leaves a lot of freedom in $(T, \tilde W, S)$. This is a likely algebraic source of the paper's non-identifiability.

Gauge: high nodes can be rescaled by $c>0$ ($T_{:j}\to cT_{:j}$, with $\tilde W$ and $S$ adjusted) by positive homogeneity. A rescaling by $c<0$ is allowed only if $\tilde\varphi_j$ is replaced by its
reflection $z\mapsto -\varphi(-z)$.

## The core obstruction

Row $j$ of (AC') demands $\sum_{i\in C_j} t_{ij}\,\varphi(a_i(X)) = \tilde\varphi(\tilde a_j(X))$, where the $a_i$ and $\tilde a_j$ are affine. A sum of nonlinear functions rarely equals one nonlinear function, but whether this matters depends on the **domain** of $X$:
- **Dense domain** (arbitrary interventions on hidden clusters): the obstruction is real. For piecewise-linear $\varphi$ it forces neuron splitting (above).
  A milder version keeps the domain but frees the high-level activation $\tilde\varphi_j$. The LHS must then be a function of one scalar, so the cluster's incoming weights must be collinear, $w_i = c_i \tilde w$, while the biases stay free. Then
  $\tilde\varphi_j(s) = \sum_i t_{ij}\varphi(c_i s + b_i)$ is a piecewise-linear activation with several kinks.
- **Finite domain** (data inputs plus interchange interventions with values from the data): the identity only has to hold at finitely many points. For piecewise-linear $\varphi$ these are linear equations, one per context (pattern $D$). This is no obstruction in principle, and the low-dimensional span of the reached contexts leaves the solutions far from unique.
- **Approximate**: minimise the residual of (AC') over the intervention distribution (cf. IIA, which is a thresholded version of this).

## Smooth $\Phi$ (tanh, sigmoid)

Under all interventions, $\sum_i t_i \tanh(a_i(x)) = \tanh(b(x))$ for all $x$ forces, by analyticity, $a_i = \pm b$. So clusters must be exact (sign-flipped) duplicates, which is even more rigid.
There is also no exact per-context linearisation, only a local Jacobian condition. Piecewise-linear, positively homogeneous $\Phi$ is the case that is both realistic and algebraically tractable.

## (AC') = IIA + well-definedness

As in the lemma, $\tau$ need not be constructive. Each $\tau_j$ reads only its cluster $x_{\pi(j)}$, but clusters may overlap.

Fix a high-level model $\mathcal M_H$, an alignment $\pi$, and an **intervention family** $\mathcal F$, meaning a set of configurations $c$ = (input $u$, interchange intervention).
Each $c$ gives a low-level state $x(c)$ and a high-level state $h(c)$. To get $h(c)$, run $\mathcal M_H$ on $u$ with each intervened variable set to its value under $\mathcal M_H$ on that variable's source input. This is the comparison IIA makes.
Fix $\tau_{\text{out}}$ (a threshold or argmax) and define each hidden $\tau_j$ implicitly by $\tau_j(x_{\pi(j)}(c)) := h_j(c)$.

**Claim.** (AC') holds on $\mathcal F$ for some $\tau$ iff (i) IIA $=1$ on $\mathcal F$ and (ii) the implicit $\tau$ is well defined. When (AC') holds, $\tau$ *is* the implicit map.
- **If.** Patched clusters take their source values, so $\tau$ maps them to the source high-level values, and the high-level intervention matches. Every hidden row then holds by construction. The output row is $\tau_{\text{out}}(y(c)) = h_{\text{out}}(c)$, which is IIA.
- **Only if.** The high-level model has a unique solution under every intervention, so any valid $\tau$ must satisfy $\tau(x(c)) = h(c)$.

**Well-definedness:** $x_{\pi(j)}(c) = x_{\pi(j)}(c') \Rightarrow h_j(c) = h_j(c')$ for every $j$ and all $c, c' \in \mathcal F$.
It fails when the algorithm says a variable changes but its neurons cannot change. A typical case is a high-level parent aligned to a *later* layer than its child: patching it cannot reach the child's neurons.

**Function class.** If $\tau_j$ is also parametric (e.g. $\tau_j = H(t_j^\top x_{\pi(j)} + \beta_j)$), add (iii) **realisability**: the implicit labelling of each cluster's reached values must lie in the class (here, be linearly separable).
Hierarchy, on a fixed $\mathcal F$: IIA $\supseteq$ (AC') with arbitrary $\tau_j$ = IIA + (ii) $\supseteq$ (AC') with thresholded $\tau_j$ = IIA + (ii) + (iii).
Every condition is relative to $\mathcal F$. IIA in the literature usually uses single-variable interventions; the released Méloux et al. code also uses only one ordering of each input pair.

**XOR example** (Méloux et al.'s pretrained network, their 42 explanations, all with disjoint clusters; one-off check, not yet saved as a script):

| family | IIA $=1$ | + well-defined (AC'), arbitrary $\tau_j$ | (AC'), thresholded $\tau_j$ |
|---|---|---|---|
| theirs (single-variable, one ordering per pair) | 42 | 14 | – |
| full (any subset of variables, all ordered pairs) | 2 | 2 | 2 (one algorithm up to De Morgan) |

Example failure (#4): variable $\neg(A\wedge B)\wedge(A\vee B)$ is aligned to neuron (1,1), and its parent $A\vee B$ to (2,0).
Patching $A\vee B$ cannot move (1,1), which stays at 1.3535. Yet $\mathcal M_H$ assigns the variable 1 or 0 depending on the patch, so no $\tau$ exists, even though IIA $=1$.

## Learned thresholded $\tau$: `src/algos/sigmoid_abstract.py`

A fitter for the thresholded case of the hierarchy above, where $\tau$ is *learned* rather than searched. Cut the network at one hidden layer and write $h$ for its post-activations, $p=\dim h$. The high-level model is the network itself up to the cut; from there, $k$ high-level variables

$$\tau_j(h) = \sigma\!\big(\kappa\,(t_j^\top h + \beta_j)\big), \qquad T=[t_1,\dots,t_k]\in\mathbb{R}^{p\times k},\quad T^\top T=I_k,$$

are the **only** parents of the output, so every layer after the cut is marginalised into one mechanism $M:\{0,1\}^k\to\mathcal Y$. The sigmoid is a differentiable stand-in for $H$; $\kappa$ is held constant and made dimensionless by the spread of the projections, so the Boolean reading is *checked* afterwards (it holds where $\kappa\cdot$margin $\gg 1$) rather than forced by annealing.

**Why orthonormal columns.** An interchange intervention on variable $j$ replaces its coordinate, $h' = h + t_j(v - t_j^\top h)$, and for $i\neq j$, $t_i^\top h' = t_i^\top h + (t_i^\top t_j)(v - t_j^\top h) = t_i^\top h$. So intervening on one high-level variable leaves the others *exactly* fixed, $\mathrm{do}(\tilde X_j)$ is well defined, and $\tau(h')$ is the intervened high-level state. Hence the condition needs no separate high-level simulation: the high-level prediction is $M(\tau(h'))$ and the low-level outcome is $\mathrm{tail}(h')$, both at the same $h'$. A non-orthogonal $T$ drags $t_i^\top h$ along and confounds single-variable interventions by construction. Note that a constructive $T$ with disjoint cluster supports already has orthogonal columns, so this is the relaxation of disjointness that keeps non-degeneracy.

**Mechanism.** $M$ is a $2^k$-row table, so realisability (iii) is *not* restricted to labellings that are linearly separable in the bits — any Boolean function is in the class. Given $T$ the optimal table is majority vote per cell in closed form, so $M$ is profiled out rather than optimised (a free $M$ otherwise absorbs gradient by growing its logits). A single-gate head $\sigma(w^\top\tau+c)$ is available for comparison and *does* impose (iii); it cannot express XOR of two high-level variables.

**Objective.** (AC') on a family $\mathcal F$ of interchange interventions, scored as IIA on hard bits. The loss is cross-entropy between the relaxed prediction and the network's own decision at $h'$. Two things matter for it to converge: $h'$ must be built under `no_grad` (letting the gradient move the sampled *point* competes with the gradient that aligns the directions), and the set of $(\text{base},\text{source},\Theta)$ triples must be fixed once, since resampling leaves a gradient-noise floor no number of steps removes. With both, a single run suffices — no restarts, no learning-rate decay, no temperature schedule. On a planted problem ($p=64$, $k=3$, random Boolean $f$) this recovers the subspace with $\lVert UU^\top - TT^\top\rVert_F = 0.05$ at $16384$ triples and shows the knee at $k=3$.

**Diagnostics.** `purity` is the share of the majority decision inside a realised cell, so purity $=1$ iff the output really is constant on each cell of the arrangement $\{\mathrm{sign}(t_j^\top h+\beta_j)\}$ — the claim the abstraction makes. `margin` is the distance to the nearest hyperplane in projection-spread units, the analogue of the certified radius $r(x_0)$. `cells`/`unseen` count realised patterns and the share whose cell the fit never visited (there IIA measures extrapolation). `graded_baseline` fits a *linear* readout of the same $T^\top h$: if it wins by a wide margin the layers after the cut use magnitude, not sign, and no number of bits on this $T$ will do. `degeneracy` compares distinct activations with the $k(p+1)$ free parameters.

**Non-uniqueness is the normal case**, as in the counting table above: many $\tau$ satisfy (AC') and each is a valid abstraction. Measured on the XOR network at $k=1$ by sampling $(t,\beta)$ uniformly ($2\times10^6$ draws, $-\beta$ inside the projection range):

| cut | observational rows only | + all 16 interchange swaps |
|---|---|---|
| $h_1$ | 3.53% of directions | 1.13% |
| $h_2$ | 29.3% | 10.0% |

Interchange interventions shrink the valid set by about $3\times$ at both cuts but leave a wide family, and the fitted direction sits inside it. Note that with 4 inputs the observational rows alone are uninformative: 4 activations in $\mathbb R^3$ are shattered by halfspaces (all 16 labellings realisable), so agreement there cannot fail. The number of linearly separable labellings of $n$ points in general position in $\mathbb R^p$ is $2\sum_{i\le p}\binom{n-1}{i}$, which is all $2^n$ exactly when $n \le p+1$; informativeness needs $n \gg p+1$, which the XOR network cannot provide and MNIST-scale data comfortably can.

## One multi-valued variable versus several binary ones

These carry the same information and are not the same abstraction. A variable boundary is a claim about **what can be set independently of what**, so splitting one variable into bits is an extra causal assertion, not a re-encoding.

**Traffic light.** One variable $L\in\{\text{red},\text{amber},\text{green}\}$, or two bits $b_1=\text{"stop-ish"}$, $b_2=\text{"go-ish"}$ with red $=(1,0)$, amber $=(1,1)$, green $=(0,1)$. As classifiers these are interchangeable. As causal variables they are not: with two binary variables an interchange intervention may take $b_1$ from a red source and $b_2$ from a green one, so the high-level model computes $(1,1)$ and asserts *amber*, while the low-level state is a chimera of a red and a green activation that the network is under no obligation to treat as amber. With one three-valued variable that operation does not exist -- the value is copied whole from a single source, so every intervened value is one that some real input produced.

So the Boolean decomposition is a **strictly stronger claim judged on a strictly larger family**: it must additionally be right about every recombination of the parts. This is also why cells that are empty observationally are not harmless. Nothing ever produces an incoherent digit code, but interchange manufactures one by taking bit 1 from a '3' and bit 3 from a '7'; the `unseen` diagnostic measures how often the evaluation lands in cells the fit never visited (16% at $k=12$ on ten-class MNIST).

Measured on the ten-class MNIST MLP ($784\to128\to128\to10$), cut after the first hidden layer, with the variable count and hence the intervention family held fixed:

| high-level variables | cells | dim | IIA(all) | purity | skill | random-subspace control |
|---|---|---|---|---|---|---|
| 3 binary, 1-D each | 8 | 3 | 0.353 | 0.359 | +0.239 | +0.095 |
| 3 binary, 3-D each | 8 | 9 | 0.384 | 0.404 | +0.266 | +0.141 |
| 3 four-valued, 3-D each | 64 | 9 | 0.625 | 0.663 | +0.570 | +0.214 |

The middle row is the control that matters: same variable count, same intervention family, same subspace dimension, only the number of values per variable differs. Going from two values to four is worth about $+0.24$ IIA, so the gain is **value cardinality**, not the width of the swap. The network does decompose; it does not decompose into independently settable bits.

**These numbers require the readout gauge to be fixed, and are otherwise not comparable.** For an $m$-valued variable the argmax is invariant to $A_c \mapsto A_c + v$, and the softmax sharpness is set by the scale of the pairwise differences $A_c - A_{c'}$, so $\kappa$ is redundant with $\lVert A\rVert$ and the model will otherwise learn its own temperature. Fix it by centring over values ($\sum_c A_c = 0$, $\sum_c a_c = 0$) and scaling so the mean squared pairwise distance is 1, i.e. $\lVert A\rVert_F = \sqrt{(m-1)/2}$; at $m=2$ this reduces to $\lVert t\rVert = 1$, the Boolean convention. Without it the two rows run at effective temperatures differing by an order of magnitude, which inflated an earlier version of this table in both directions (0.160 against 0.779, a gap of 0.62 rather than 0.24).

Note also the floor $d_j \ge m_j - 1$: a variable with $m$ values needs $m-1$ dimensions for its values to be unordered, since the $m$ forms have $m-1$ independent differences. At $d_j = 1$ a four-valued variable degenerates into four ordered bands on a line.

## When a single variable is vacuous

Let the high-level model have one variable $V$ reading a $d$-dimensional block, so the family is $\mathcal F=\{\emptyset,\{V\}\}$.

- **$d=p$ (the whole layer).** The intervention sets $h'=h_{\text{source}}$ exactly, so the network's outcome is the source's outcome and IIA collapses to observational agreement between the readout and the network's decision. Nothing causal is tested: the "abstraction" is a probe wearing different clothes.
- **$d<p$.** Now $h'$ keeps the base's $p-d$ remaining dimensions and takes only the block from the source, so it is a genuine chimera and the condition has content. The claim gets stronger as $d/p$ falls. Empirically it is not satisfied for free: one ten-valued variable on 9 of 128 dimensions gives IIA(all) $0.661$ with purity $0.701$, nowhere near 1.

Informativeness therefore has two axes, and a result should be read on both:
1. **compression** -- $d_{\text{total}}/p$, and the number of cells against the number of distinguishable outcomes;
2. **structure** -- how many variables there are and whether the mechanism genuinely depends on several of them.

A single-variable abstraction has a trivial graph, so even at IIA $=1$ it only tells us which regions of activation space map to which outcome. That is a statement about *where* the decision lives, not about the intermediate steps the network takes to reach it. Several interacting variables are what make the mechanism a claim about the computation.

## A near-perfect abstraction of a ten-class MNIST MLP, and what it does not say

Network $784\to128\to128\to10$ (ReLU), test accuracy $0.965$, cut after the **first** hidden layer ($p=128$).

**Result.** One high-level variable with $m=16$ values on a $9$-dimensional block: IIA(all) $\mathbf{0.980 \pm 0.002}$, purity $\mathbf{0.978}$, over 8 seeds, using $9/128$ dimensions for $10$ classes. A frozen random subspace of the same size reaches $0.07-0.12$, so the test discriminates.

**The subspace is read off the weights, not learned, and learning it is worse.** Take the final layer $W_3$, centre its rows (the $\arg\max$ is shift-invariant, so the shared component is irrelevant and the rank drops to $K-1=9$), pull it back through the second layer as $\tilde W = \mathrm{centred}(W_3)\,W_2$, and take the $9$-dimensional row space. No data, no training, no interventions. Only the *carving* of that subspace -- the $m$ linear forms and the table -- is learned by interchange training. Over 8 seeds:

| subspace | $m=12$ | $m=16$ | $m=20$ |
|---|---|---|---|
| from the weights | $0.894\pm0.057$ | $\mathbf{0.980\pm0.002}$ | $0.980\pm0.002$ |
| learned by interchange training | $0.941\pm0.029$ | $0.960\pm0.006$ | $0.962\pm0.004$ |

Two things to take from this. **(i) $m$ must exceed $K$ by a margin.** At $m=12$ the runs are bimodal (six near $0.88$, two near $0.98$): a randomly initialised readout commits to a poor region-to-value assignment at full sharpness and cannot escape. Slack removes it entirely and $m=20$ adds nothing over $m=16$. A trainable $T$ partly compensates for a bad readout by rotating away from it, which is why learning $T$ *wins* at $m=12$ and loses once the readout has room. **(ii) Freezing $T$ gives a better-posed problem.** With $T$ fixed, $h'$ and its targets are constant for the whole run -- an ordinary stationary supervised fit. With $T$ trainable, every step moves $h'$ and the labels shift underneath the optimiser.

The ReLU between the cut and the output contributes nothing to the subspace: the naive $\tilde W$ row space, the same thing re-weighted by each unit's firing rate, and the SVD of the full data-dependent Jacobian agree to two decimal places (mean principal-angle cosine $0.84$ against the learned subspace in all three cases; a random pair sits at $0.24$). **The weights-only subspace should therefore be a standard control**: wherever the tail is near-linear, the alignment search adds nothing over linear algebra, and the Jacobian spectrum predicts this in advance (rank $\approx K-1$ means do not bother; rank $\gg K-1$ means the subspace is genuinely data-dependent).

**Why nine dimensions.** The decision is $\arg\max$ over ten logits, which is invariant to a common shift, so it lives in nine. Empirically a rank-9 linear probe on $h_1$ reproduces the network's own decision $0.984$ of the time and higher rank adds nothing (rank 3: $0.906$; rank 12: $0.983$; rank 128: $0.984$). The real content of the abstraction is therefore a claim about the *later* layers: **for decision purposes, layers 2-3 amount to a rank-9 linear map followed by $\arg\max$, contributing no decision-relevant nonlinearity.**

**Why more than ten values are needed.** Not for the reason one might expect. Setting $m=K$ does not fail because a class needs two regions -- it fails because the fit does not reliably *claim* all $K$ regions: at $m=10$ and $m=11$ only 8 or 9 digits get a region at all, and IIA sits at $0.77-0.88$. At $m=16$ both seeds realise exactly 10 regions, one per digit, each $97-100\%$ pure, with no digit split. So the surplus values buy optimisation slack rather than covering capacity. (An earlier version of this note attributed it to the convexity of $\arg\max$ regions, on the strength of one fit in which digit `0` occupied two regions -- upright and slanted zeros. That split is real in that fit but is not necessary: it does not recur once all ten regions are found. Convexity remains a genuine restriction of the $\arg\max$ construction relative to signs-plus-table, which can label non-adjacent cells alike; it is just not what drives the $m > K$ requirement here.)

**It does not factorise.** On the same network and layer, with the intervention family matched by variable count:

| high-level variables | dims | IIA(all) |
|---|---|---|
| $\le 12$ binary | 3-12 | $\approx 0.51$ |
| 3-4 categorical, 3-4 dims each | 9-16 | $0.63-0.75$ |
| **1 variable, 9 dims, 16 values** | **9** | $\mathbf{0.980}$ |

Every multi-variable configuration pays for imposing a factorisation the network does not have. Read against the two axes above, this result is **strong on compression and empty on structure**: a trivial graph, one node feeding the output. It localises the decision and says nothing about intermediate computation, which is exactly the limit the vacuity section predicts. A task whose labels are compositional by construction (MNIST-PVR, where a pointer selects among digits) is the test that can distinguish "the method cannot find factorisations" from "this network has none".

**Practical.** Set $m$ comfortably above the number of classes -- $m=K+6$ was enough here and $m=K+2$ was not -- and let surplus values die. Take the subspace from the weights before trying to learn it, and check the Jacobian spectrum to see whether learning it could even help. At $m=K+2$ the seed spread is $\pm0.03$ to $\pm0.06$ and a bad seed can leave a class unclaimed; at $m=K+6$ it collapses to $\pm0.002$.

## Finding the dimension first, rather than searching for it

The $9$-dimensional code above was found by trial and error over the number of variables, which is the wrong axis: 3 variables of 3 dimensions is also 9 dimensions, it merely splits them wrongly. The dimension itself is obtainable **before fitting anything, and without interventions**.

The decision depends on $h$ only through differences of logits, so the object to look at is the Jacobian of the tail with the common shift removed, $\;J(h) - \frac{1}{K}\mathbf{1}\mathbf{1}^\top J(h)$, stacked over inputs. Its singular spectrum gives the effective dimension. On the ten-class MNIST MLP:

| cut | top singular values (normalised) | dims for 95% / 99% |
|---|---|---|
| $h_2$ | $1.00,\,.92,\,.90,\,.81,\,.76,\,.75,\,.65,\,.62,\,.54,\,\mathbf{0},\,0,\,0$ | 9 / 9 |
| $h_1$ | $1.00,\,.89,\,.73,\,.68,\,.52,\,.44,\,.36,\,.28,\,.26,\,\mathbf{.10},\,.07,\,.06$ | 8 / 11 |

At $h_2$ the rank is exactly $K-1=9$ because the tail is one linear layer. At $h_1$ there is a $2.6\times$ gap after the ninth value. Both agree with the rank-9 probe ($0.984$) and with the fitted abstraction. **Order of operations:** estimate $d$ from the spectrum, set $m$ generously, and only then vary the number of variables -- which is the axis that carries the scientific question.

**The spectrum does not give a usable initialisation, which is worth recording because it is the obvious thing to try.** Initialising $T$ from the top-$d$ right singular vectors gives IIA $0.752 \pm 0.09$ against random initialisation's $0.957 \pm 0.01$ (3 seeds each; the assignment was verified exact, subspace error $4\times 10^{-7}$). The cause is not that the spectrum points somewhere wrong: the principal-angle cosines between the Jacobian subspace and the subspace the fit chooses are $0.70-0.95$, mean $0.84$, where a random 9-dimensional subspace of $\mathbb R^{128}$ would sit near $\sqrt{9/128}\approx0.27$. Starting $84\%$ aligned is worse than starting $27\%$ aligned, which points at the landscape rather than the subspace -- with $T$ already near optimal the randomly initialised readout commits to a region-to-value assignment at full sharpness and cannot escape it, while from a random $T$ the two co-adapt. Consistent with this, the seed spread grows from $0.025$ to $0.184$. A warm-up at low $\kappa$, or fitting the readout before releasing $T$, would be the thing to try.

More generally the spectrum is an **observational and local** object -- it linearises at the data points, whereas the criterion is interventional and moves $h$ far off the data manifold. It is a cheap prior on where to look, not a substitute for the fit.

## Why one variable is always the easy answer

Across every experiment, one multi-valued variable beat several smaller ones. Three distinct things contribute, and they should not be conflated.

1. **Definitional.** $j$ variables means the family is indexed by all $2^j$ subsets, and every recombination of parts must be honoured. More variables is a *strictly larger* family and hence a *strictly stronger* claim, independent of any network. Some of "easier" is just "claims less".
2. **About this network.** The $9$-dimensional decision code does not split into independently settable blocks: every multi-variable configuration lost, at matched dimension and matched family. The code is holistic.
3. **About this task.** MNIST digit classification has no compositional label structure -- the outcome *is* one categorical variable -- so there is nothing in the data for a factorisation to correspond to. Contrast MNIST-PVR, where the label is determined by a pointer selecting among digit values, so a factorisation exists by construction.

**Methodological corollary.** A one-variable abstraction is the weakest non-trivial claim available: it says little more than "the decision is a function of a low-dimensional code", and it is guaranteed to exist whenever that is true. So a good one-variable fit should never be reported as having *discovered a high-level model*; the discovery claim lives entirely in whether **more than one** variable also works. Reporting the one-variable fit is reporting compression; reporting a multi-variable fit is reporting structure.
