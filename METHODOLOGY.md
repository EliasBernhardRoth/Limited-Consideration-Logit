# Technical Methodology

**Logit Models under Limited Consideration**

This document describes the static logit model, maximum likelihood estimators and Monte Carlo experiments implemented in Python, based on my research at Heinrich Heine University Düsseldorf (2024).

[Return to repository overview](README.md)

<br>

## 1. Model specification

Consumers choose among ten products across repeated market occasions. The consumer, market and product indices are

```math
\begin{alignedat}{2}
&i &\quad &\in\{1,\ldots,N\},\\
&t &\quad &\in\{1,\ldots,T\},\\
&j &\quad &\in\mathcal C=\{1,\ldots,J\},\qquad J=10.
\end{alignedat}
```

<br>

Each consumer makes one choice per market, giving $`n=NT`$ observations per repetition. The market index denotes a choice occasion rather than a dynamic state: past choices do not affect subsequent decisions. There is no outside option.

Utility contains an observed component, an additional Gaussian component and a Type I Extreme Value shock:

```math
U_{itj}=V_{itj}+\eta_{itj}+\varepsilon_{itj},
\qquad V_{itj}=\beta_p p_{itj}+\delta_j.
```

<br>

The deterministic component captures price sensitivity and product attractiveness. Preferences are homogeneous. Product 1 is the reference category because a common shift in utilities leaves choice probabilities unchanged:

```math
\delta_1=0,\qquad \beta_p^0=-0.173,
\qquad \theta=(\beta_p,\delta_2,\ldots,\delta_{10})'.
```

<br>

The additional utility components satisfy

```math
\eta_{itj}\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma_\eta^2),
\qquad \sigma_\eta=0.25,
\qquad \varepsilon_{itj}\overset{\mathrm{iid}}{\sim}\mathrm{EV1}(0,1).
```

<br>

The Gaussian component adds unobserved utility variation; the Gumbel shocks yield closed-form logit probabilities conditional on those draws. Both shock families are independent of each other, prices and consideration draws. The Gumbel scale is normalised to one.

<br>

## 2. Stochastic prices and consideration sets

### Price generation

Prices vary independently across consumers, markets and products:

```math
p_{itj}=\frac{\overline p_j}{3}+16v_{itj},
\qquad v_{itj}\overset{\mathrm{iid}}{\sim}\mathcal U(0,1).
```

<br>

The list prices are given in Section 5. Random price variation provides information about price sensitivity separately from fixed product effects. Prices are exogenous by construction.

### Consideration probabilities

The probability that product $`j`$ enters a proposed consideration set is

```math
\phi_{itj}=\Lambda(\gamma_p p_{itj}+\gamma_j),
\qquad \Lambda(z)=\frac{1}{1+\exp(-z)}.
```

<br>

The price coefficient is $`\gamma_p=0.15`$. Consideration intercepts equal −2.5 for products 1–5 and zero for products 6–10.

For each product, an independent uniform draw is compared with its consideration probability. The product enters the proposed set $`\mathcal D_{it}`$ if the draw is smaller than that probability.

Price therefore affects choice through two opposing channels in this calibration: a higher price makes a product less attractive but more likely to be considered. Products 6–10 are also more likely to enter consideration than products 1–5 at the same price. Observed choices consequently reflect both preferences and attention.

### Empty-set treatment

Empty sets are redrawn at fixed prices, conditioning the Bernoulli proposal on nonempty sets:

```math
q(\mathcal D\mid\mathbf p_{it})=
\frac{
\prod_{j\in\mathcal D}\phi_{itj}
\prod_{k\notin\mathcal D}(1-\phi_{itk})
}{1-\prod_{k\in\mathcal C}(1-\phi_{itk})},
\qquad \varnothing\ne\mathcal D\subseteq\mathcal C.
```

<br>

The numerator is the probability of drawing a particular set; the denominator is the probability of drawing any nonempty set. Conditioning on nonemptiness introduces dependence between the accepted consideration indicators. The implementation records redraws, caps attempts at 10,000 and optionally raises an error on an empty set.

<br>

## 3. Choice probabilities and estimation sets

### Conditional choice probabilities

The consumer chooses the product that maximises $`V_{itj}+\eta_{itj}+\varepsilon_{itj}`$ within $`\mathcal D_{it}`$.

Integrating over the Gumbel shocks while conditioning on the Gaussian draws gives

```math
P_{itj}^{\mathrm{sim}}=
\begin{cases}
\displaystyle
\frac{\exp(V_{itj}+\eta_{itj})}
{\sum_{k\in\mathcal D_{it}}\exp(V_{itk}+\eta_{itk})},
&j\in\mathcal D_{it},\\[8pt]
0,&j\notin\mathcal D_{it}.
\end{cases}
```

<br>

Only considered products can be chosen. The simulation draws Gaussian shocks, evaluates the probabilities with `softmax` and samples a choice using a uniform draw and cumulative probabilities. Integrating out the Gumbel shocks avoids simulating them explicitly. If only one product is considered, it is chosen with probability one.

### Choice sets used in estimation

The three estimators use identical choices and prices but different availability masks:

```math
\begin{aligned}
\mathcal S_{it}^{\mathrm{FILM}}&=\mathcal C,\\
\mathcal S_{it}^{\mathrm{MLM}}&=\{y_{it}\}\cup\mathcal B_{it},\\
\mathcal S_{it}^{\mathrm{WLM}}&=\mathcal D_{it}.
\end{aligned}
```

<br>

Here $`y_{it}`$ is the chosen product and $`\mathcal B_{it}`$ contains four products sampled uniformly without replacement from the other nine. FILM treats every unchosen product as a rejected alternative. MLM uses a smaller estimation set, which may still include unconsidered products. WLM instead observes the actual set by construction, providing a benchmark for the value of this information.

<br>

All three fitted models use the observed utility component $`V_{itj}(\theta)`$ and omit $`\eta_{itj}`$. Their probabilities are

```math
P_{itj}^{m}(\theta)=
\frac{\mathbf 1\{j\in\mathcal S_{it}^{m}\}\exp(V_{itj}(\theta))}
{\sum_{k\in\mathcal S_{it}^{m}}\exp(V_{itk}(\theta))},
\qquad m\in\{\mathrm{FILM},\mathrm{MLM},\mathrm{WLM}\}.
```

<br>

The estimators differ in the alternatives entering the denominator, not in their utility specification. The simulated choice probabilities are not used in estimation.

### Sampling alternatives and the likelihood correction

Let $`\pi(\mathcal S\mid j)`$ be the sampling probability of set $`\mathcal S`$ given chosen product $`j`$. Under a correctly specified full-set logit, Bayes' rule gives

```math
\begin{aligned}
\Pr_\theta(y=j\mid\mathcal S)&=
\frac{P_j(\theta)\pi(\mathcal S\mid j)}
{\sum_{k\in\mathcal S}P_k(\theta)\pi(\mathcal S\mid k)}\\[6pt]
&=\frac{\exp(V_j(\theta)+\log\pi(\mathcal S\mid j))}
{\sum_{k\in\mathcal S}\exp(V_k(\theta)+\log\pi(\mathcal S\mid k))}.
\end{aligned}
```

<br>

This is the sampling correction associated with McFadden's approach. For the implemented chosen-plus-four design,

```math
\pi(\mathcal S\mid j)=\binom{9}{4}^{-1}=\frac{1}{126},
\qquad j\in\mathcal S,\quad |\mathcal S|=5.
```

<br>

Every candidate chosen product has the same probability of producing a given five-product set. The sampling term therefore cancels from the ratio. No additional correction is needed under a correctly specified full-set logit, but this does not recover unknown consideration sets. See McFadden (1978) and [Train (2009), Section 3.7.1](https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf).

<br>

## 4. Structural maximum likelihood estimation

### Conditional likelihood

Let $`\mathbf z_{itj}=(p_{itj},\mathbf 1\{j=2\},\ldots,\mathbf 1\{j=10\})'`$ collect the price and product indicators. The deterministic utility is $`V_{itj}(\theta)=\mathbf z_{itj}'\theta`$.

For model $`m`$, the conditional log-likelihood is

```math
\ell_m(\theta)=\sum_{i=1}^{N}\sum_{t=1}^{T}
\left[
\mathbf z_{it,y_{it}}'\theta
-\log\left(\sum_{j\in\mathcal S_{it}^{m}}
\exp(\mathbf z_{itj}'\theta)\right)
\right].
```

<br>

The first term rewards the utility of the observed choice; the second normalises it relative to the available alternatives. Maximising the likelihood over the price coefficient and nine product effects finds the parameters that best explain the observed choices. Consideration parameters are not estimated.

### Analytic derivatives

Write $`L_m=-\ell_m`$ and index the $`n=NT`$ observations by $`d`$. Let $`\overline{\mathbf z}_d(\theta)=\sum_{j\in\mathcal S_d^m}P_{dj}^{m}(\theta)\mathbf z_{dj}`$ be the probability-weighted mean of product characteristics in the estimation set.

The gradient compares the model-implied expected characteristics of a chosen product with the characteristics actually observed:

```math
\nabla L_m(\theta)=\sum_{d=1}^{n}
\left[\overline{\mathbf z}_d(\theta)-\mathbf z_{d,y_d}\right],
```

<br>

The Hessian describes how that gradient changes with the parameters. It measures the curvature used to guide optimisation and calculate standard errors:

```math
H_m(\theta)=\sum_{d=1}^{n}
\left[
\sum_{j\in\mathcal S_d^m}P_{dj}^{m}(\theta)\mathbf z_{dj}\mathbf z_{dj}'
-\overline{\mathbf z}_d(\theta)\overline{\mathbf z}_d(\theta)'
\right].
```

<br>

Each Hessian contribution is a covariance matrix, so the negative log-likelihood is convex. In Python, the derivatives are evaluated directly using NumPy arrays rather than numerical differentiation. A singleton set supplies no information about relative preferences, and convexity alone does not guarantee finite, identified estimates.

### Stable probability evaluation

Direct exponentiation of large utilities can cause numerical overflow. The implementation uses SciPy's `logsumexp` to evaluate the log of the denominator stably and subtracts it from each available alternative's utility. Unavailable utilities are set to $`-\infty`$ using Boolean masks, so they contribute zero probability. This also avoids taking the logarithm of a probability that has rounded to zero.

### Numerical optimisation

The optimiser minimises $`Q_m(\theta)=L_m(\theta)/n`$, scaling the gradient and Hessian by the same factor. This leaves the minimiser unchanged.

Optimisation starts from zero and uses SciPy's `trust-exact` method, which uses both the analytical gradient and Hessian to determine parameter updates. BFGS with the analytical gradient is the fallback. Both methods are subject to convergence and numerical validity checks.

### Conventional model-based standard errors

Covariance is computed from the Hessian of the **total** negative log-likelihood:

```math
\begin{aligned}
\widehat{\mathrm{Var}}_{\mathrm{model}}(\widehat\theta_m)
&=H_m(\widehat\theta_m)^{-1},\\[6pt]
\widehat{\mathrm{SE}}_{m,k}
&=\sqrt{\left[H_m(\widehat\theta_m)^{-1}\right]_{kk}}.
\end{aligned}
```

<br>

The curvature measures how strongly the likelihood changes around the estimated parameters. A relatively flat likelihood implies greater estimation uncertainty. The information matrix is checked for finiteness, conditioning and positive definiteness, and its inverse is computed using Cholesky decomposition. The total rather than average likelihood is used to obtain the appropriate sample-size scaling. These are conventional model-based standard errors, not sandwich-robust standard errors under misspecification.

<br>

## 5. Monte Carlo experimental design

### Baseline calibration

| Product | Utility effect $`\delta_j`$ | List price $`\overline p_j`$ | Consideration intercept $`\gamma_j`$ |
| ---: | ---: | ---: | ---: |
| 1 | 0.000 | 22.98 | −2.5 |
| 2 | 0.368 | 24.98 | −2.5 |
| 3 | −0.497 | 22.98 | −2.5 |
| 4 | 0.093 | 22.98 | −2.5 |
| 5 | 0.088 | 24.98 | −2.5 |
| 6 | 0.306 | 25.00 | 0 |
| 7 | −0.581 | 24.98 | 0 |
| 8 | −1.075 | 19.98 | 0 |
| 9 | −0.909 | 24.98 | 0 |
| 10 | −0.405 | 24.98 | 0 |

### Experiment profiles

Table 1 uses 50 consumers, 100 markets and 100 repetitions. The density experiments use 100 consumers and 10, 50, 100 or 200 markets, with 50 repetitions per setting.

Together these comprise 300 dataset runs, 2.3 million simulated choices and 900 model estimations. All three estimators use identical data within each repetition.

### Simulation and repetition

Each repetition generates new prices, consideration sets and choices. Fitting all three models to the same dataset isolates differences in estimation sets. Varying the market count changes sample size without changing the underlying model.

Each repetition uses NumPy's `RandomState` with seed $`22{,}102{,}024+r`$, starting at $`r=1`$. The seed rule restarts across experiments, so sample-size results can be dependent through shared random streams. Matching MATLAB seeds does not imply identical draws.

<br>

## 6. Statistical evaluation

### Parameter recovery

For a parameter with true value $`\theta_k^0`$, calculate

```math
\begin{aligned}
\overline{\widehat\theta}_k
&=\frac{1}{R}\sum_{r=1}^{R}\widehat\theta_k^{(r)},\\
\widehat{\mathrm{Bias}}_k
&=\overline{\widehat\theta}_k-\theta_k^0,\\
\widehat{\mathrm{RMSE}}_k
&=\sqrt{\frac{1}{R}\sum_{r=1}^{R}
\left(\widehat\theta_k^{(r)}-\theta_k^0\right)^2}.
\end{aligned}
```

<br>

Bias measures how far the average estimate lies from the true parameter. RMSE incorporates both this displacement and the variation of estimates across repetitions. An estimator can become more precise as the sample grows without becoming more accurate if its bias persists.

### Monte Carlo uncertainty

The sample standard deviation and Monte Carlo standard error of the mean are

```math
\begin{aligned}
s_{\mathrm{MC},k}
&=\sqrt{\frac{1}{R-1}\sum_{r=1}^{R}
\left(\widehat\theta_k^{(r)}-\overline{\widehat\theta}_k\right)^2},\\
\mathrm{MCSE}(\overline{\widehat\theta}_k)
&=\frac{s_{\mathrm{MC},k}}{\sqrt R}.
\end{aligned}
```

<br>

The standard deviation describes variation across simulated datasets. The Monte Carlo standard error instead measures the precision of the estimated mean and bias given finitely many repetitions.

### Table 1 convention

Table 1 reports the average of the model-based standard errors from individual fits, not the Monte Carlo standard error of the mean.

The original star convention compares the absolute ratio of the mean estimate to this average standard error with thresholds 1.645, 1.960 and 2.576. It is descriptive, not a formal test of the Monte Carlo mean or of equality to the true parameter.

The README measures understatement as $`100(|\beta_p^0|-|\overline{\widehat\beta}_p|)/|\beta_p^0|`$.

<br>

## 7. Density estimation

For each model and market count, the $`R=50`$ price estimates are smoothed with a Gaussian kernel:

```math
\widehat f(b)=\frac{1}{Rh}\sum_{r=1}^{R}
\varphi\left(\frac{b-\widehat\beta_p^{(r)}}{h}\right),
\qquad
\varphi(z)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{z^2}{2}\right).
```

<br>

Each estimate contributes a bell-shaped kernel centred at its value. Adding these kernels reveals the location and dispersion of the sampling distribution, rather than only its mean. The bandwidth controls the amount of smoothing and follows R's `bw.nrd0` rule, based on the sample standard deviation, interquartile range and number of repetitions. The curves use common axes, so they show directly whether increasing sample size concentrates estimates around the true coefficient or around a displaced value.

<br>

## 8. Interpretation

### Structural estimation under misspecification

Without observed consideration sets, the true choice probability integrates over both sets and Gaussian shocks (suppressing consumer-market indices):

```math
\Pr_0(y=j\mid\mathbf p)=
\sum_{\mathcal D\ni j}q(\mathcal D\mid\mathbf p)
\mathbb E_\eta\left[
\frac{\exp(V_j^0+\eta_j)}
{\sum_{k\in\mathcal D}\exp(V_k^0+\eta_k)}
\right].
```

<br>

The full-set logit generally cannot reproduce this mixture of attention and preference effects at the true utility parameters. Under standard regularity conditions, a misspecified estimator instead targets

```math
\theta_m^*=\arg\max_\theta\;
\mathbb E_0\left[\log P_m(y\mid\mathcal S^m,\mathbf p;\theta)\right].
```

<br>

This best-fitting parameter vector need not equal the true parameter vector $`\theta^0`$. More data can therefore concentrate estimates around an incorrect value rather than eliminate the specification error.

### Preference recovery with different sets

The numerical comparison and density figures are presented in the [repository overview](README.md#selected-results).

FILM interprets unconsidered products as rejected alternatives. In this calibration, the positive effect of price on attention offsets part of its negative effect on utility, leading to substantially weaker estimated price sensitivity. MLM retains this information problem despite using fewer alternatives. WLM compares choices only with genuinely considered products, improving parameter recovery, although it still omits the Gaussian shocks.

Conditional logit satisfies IIA for fixed sets and utilities, but these odds need not describe choices after unobserved consideration sets and Gaussian shocks are averaged out.

**Central insight:** Sampling alternatives does not identify which products consumers actually considered.

<br>

## 9. Limitations and reproducibility

Each experiment saves estimates, summaries, comparison tables and settings with software versions and completion status. See the [repository overview](README.md#quick-start) for reproduction commands.

- Gaussian shocks follow the MATLAB calibration (standard deviation 0.25), rather than the standard normal component described in Section 3.1 of the 2024 paper. Empty sets are resampled at fixed prices.
- WLM observes true sets; the consideration process is not estimated. All fitted models omit Gaussian utility shocks, so closer recovery does not establish unbiasedness or consistency.
- Curvature-based standard errors and descriptive stars are not misspecification-robust inference.
- Findings are calibration-specific and do not establish performance with endogenous prices or empirical heterogeneity. Density curves use new simulated datasets, not bootstrap resamples.

<br>

## References

- Abaluck, J., & Adams-Prassl, A. (2021). What do consumers consider before they choose? Identification from asymmetric demand responses. *The Quarterly Journal of Economics*, 136(3), 1611–1663.
- McFadden, D. (1978). Modelling the choice of residential location. In *Spatial Interaction Theory and Planning Models*, pp. 75–96. North-Holland.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*, second edition. Cambridge University Press. [Chapter 3: Logit](https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf).
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.

---

**Author:** Elias Roth

Research conducted at Heinrich Heine University Düsseldorf in 2024.

[Return to repository overview](README.md)
