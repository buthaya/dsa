# Finding the best machine learning model for a task

## Some theoretical foundations of  machine learning

### Goal 

The goal of *machine learning* is to build a function that creates rules between an input and an output from examples. 

### Focus on supervised machine learning

For the purpose of this course, we will focus on supervised machine learning. The **goal of supervised machine learning is to predict the label $Y$ associated to each new observation $X$** (where we assume $Z=(X,Y)\overset{i.i.d}{\sim} \mathbb{P}$ is a new observation of the law $\mathbb{P}$).

### Vocabulary and notations

- A **feature**, denoted $V_i$ is a measurable property that describes the state of a system. It is an input variable for the model. For example, for churn prediction, a feature would denote the recent activity of the customer, the number of products bought. Features are the "columns" of our dataset.
- The **label** $Y_i$ is the output variable we want to predict. This is a single column of our dataset. 
- An **observation** is a couple $Z_i=(X_i, Y_i)$ of instances and labels. We assume that $Z_i\overset{i.i.d}{\sim} \mathbb{P}$ are independent realisations of the same unknown distribution law $\mathbb{P}$. Typically, $X_i \in \mathcal{X}=\mathbb{R}^p$ and $Y_i \in \mathcal{Y}$ which is a finite subset of $\mathbb{N}$ (for classification) or a subset of $\mathbb{R}$ (for regression).
- An **instance** $X_i=(X_{i,1}, ...X_{i,p})$ is a realization of the random variables of all the considered features (i.e. each margin $X_{i,k}\overset{i.i.d}{\sim}V_k$). It is an observed example of the inputs variables. Instances are the "rows" of our dataset.
- The dataset $(Z_1, ..., Z_n)$ is called the **training set**.  
- A **prediction function** is a (measurable) function from $\mathcal{X}$ in $\mathcal{Y}$. The set of prediction function is called  $\mathcal{F}(\mathcal{X}, \mathcal{Y})$. 
- A **machine learning algorithm** is a function that maps a training set and a prediction function, i.e. it is a a function between $\underset{n\in \mathbb{N}}{\cup}(\mathcal{X} \times \mathcal{Y})^n \to \mathcal{F}(\mathcal{X}, \mathcal{Y})$. 
- A **loss function** is a function $l:\mathcal{Y} \times \mathcal{Y} \to \mathbb{R} $. It represents the "cost" when the actual value is $y$ and the predicted value is $y'$. For classification, a common loss function is $l(y,y')=\mathbb{1}_{y\neq y'}$ ; for regression, we often use  $l(y,y')= \lvert y-y' \rvert^p $ which is called regression $L^p$ (with p=2 it is the ordinary least square regression).

### Machine learning as a general optimisation problem

The quality of a prediction function $g: \mathcal{X} \to \mathcal{Y}$ is measured by its "risk" (the generalisation error): $\mathcal{R}_{\mathbb{P}}(g)= \mathbb{E}_\mathbb{P}[l(Y, g(X))]$.

The (in fact, "one") "best" prediction function is the one that minimizes the average loss in regards to the law $\mathbb{P}$. Hence, and with all previous notations, machine learning problems can be rewritten as the search for an optimum $g_\mathbb{P}^*$:

$$
g_\mathbb{P}^* \in \underset{g \in \mathcal{F}(\mathcal{X}, \mathcal{Y})}{\argmin}{\space \mathbb{E}_\mathbb{P}[l(Y, g(X))]}
$$

$g_\mathbb{P}^*$ is called an **oracle function** or a **Bayes predictor**. 

### A specific set of solution for L^p and classification loss

Note that there is no guarantee that $g_\mathbb{P}^*$ exist all the times, but it does for the loss function previously introduced. 

> [!TIP]
> Intuitively, before the formalism: if you had infinite data, what is the single best number to predict for a group of similar risks? For squared error, it is simply the *average* outcome within that group (e.g. the average severity for a given rating cell). For classification, it is the *most frequent class* within that group (e.g. "most claims in this segment are not fraudulent"). The theorem below just makes this precise and shows it holds for any $x$, not only "on average".

**Theorem**: We assume that $X \sim P_X$ and we denote the conditional probability distribution of Y given x by  $P_{Y|X}$. Then a function which minimizes the conditional expectation for all x is an oracle function, i.e: 

$$
\forall x \in \mathcal{X}, g_\mathbb{P}^*(x) \in  \underset{y \in  \mathcal{Y}}{\argmin}{\space \mathbb{E}_\mathbb{P}[l(Y, y)|X=x]} \Rightarrow g_\mathbb{P}^* \in \underset{g \in \mathcal{F}(\mathcal{X}, \mathcal{Y})}{\argmin}{\space \mathbb{E}_\mathbb{P}[l(Y, g(X))]}
$$

Specifically we can derive oracle function for several problems: 
- For least square regression, $\eta_{\mathbb{P}}^{*}(x)=\mathbb{E}_\mathbb{P}[Y|X=x]= \int_{\mathcal{Y}}y d\mathbb{P}_{Y|X}(y|x)$ is an oracle function
- For classification, $g_\mathbb{P}^*(x) \in \underset{y \in \mathcal{Y}}{\argmax}{\space \mathbb{P}(Y=y|X=x)}$ is an oracle function
- For binary classification, $\mathcal{Y}=\{0,1\}$ and the function $x \mapsto \mathbb{1}_{\{\eta_{\mathbb{P}}^{*}(x)>1/2\}}$ is an oracle function

### Choosing a loss function in practice

The theorem above tells you what the oracle function *is* for a given loss, but not *which loss to pick* for a given business problem — and that choice matters more than most other modeling decisions.

> [!CAUTION]
> Picking a loss implicitly assumes a distribution for $Y|X$. If that assumption is wrong, your model will be systematically biased even with infinite data and a perfect optimizer — no amount of tuning fixes a mismatched loss.

| Loss | Implicit assumption on $Y$ | Typical actuarial use case | Why |
|---|---|---|---|
| Squared error (L2) | Symmetric, roughly constant-variance noise (Gaussian-like) | Rarely ideal on its own for claims data | Penalizes large errors quadratically — a handful of extreme claims can dominate the fit |
| Absolute error (L1) / Huber | Heavier-tailed or outlier-prone noise | Robust regression, reserving percentiles | Grows linearly with the error instead of quadratically, so extreme values matter less |
| Poisson deviance | Non-negative counts, variance $\propto$ mean | Claim **frequency** (number of claims) | Respects that a count can't go negative and that high-frequency segments are naturally noisier; needs an **exposure offset** (see below) |
| Gamma deviance | Strictly positive, right-skewed, variance $\propto$ mean$^2$ | Claim **severity** (average cost per claim) | Matches the shape of severity data much better than squared error |
| Tweedie deviance | Mass at zero + positive continuous tail | **Pure premium** (frequency $\times$ severity) in one model | Avoids training two separate models when the zero-claim mass is large |
| Log-loss / cross-entropy | Well-calibrated probabilities | Fraud detection, lapse/churn classification | Convex and differentiable (unlike 0-1 loss below), and it punishes confidently-wrong predictions harder than a hesitant wrong prediction |
| 0-1 loss | — | Final reporting metric only (e.g. accuracy) | Not differentiable, so it can't be used to *train* a model — only to *report* on one already trained with a smoother surrogate loss |

Concretely: if you are modeling **claim frequency**, use a Poisson deviance, not squared error — see "Exposure and offsets" just below. If you are modeling **claim severity**, a Gamma deviance (or, more simply, minimizing on a log-transformed target) will track the true cost far better than squared error, precisely because severity data is right-skewed: a global average is dragged around by a few very large claims, while a loss that respects the skew is not. If you need a **single pure-premium model**, Tweedie deviance lets you skip the frequency/severity split. For **fraud or lapse classification**, train on log-loss (or an equivalent convex surrogate) even though the business ultimately cares about a 0-1 decision — you then pick the decision threshold separately (see "Redefine the problem" below).

The figure below shows why the loss choice matters concretely, using the `claims.csv` dataset used later in this document's worked example: squared error and absolute error are both computed for (a) a single global average severity prediction, and (b) an average per `claim_type`. Segmenting by `claim_type` alone cuts the mean squared error by about 85% and the mean absolute error by about 74% — not because the model got smarter, but because the loss is finally being evaluated against a prediction that respects the shape of the data.

![Loss shapes and effect of segmenting by claim_type](assets/loss_functions_comparison.png)

### Exposure and offsets in frequency modeling

Suppose you want to model how many claims a policy will file (a Poisson-style frequency model). Two policies with the same features but very different **durations** — one insured for a full year, one insured for one month before switching insurer — are not comparable if you just look at their raw claim counts: the year-long policy has twelve times the opportunity to file a claim. The raw count is not the quantity you actually want to model; the **claim rate per unit of exposure** is.

This is why frequency GLMs use $\log(\text{exposure})$ as an **offset**: instead of modeling $\mathbb{E}[\text{count}|X]$ directly, the model effectively predicts $\log(\text{rate}) = \eta(X)$ and adds $\log(\text{exposure})$ to it before comparing to the observed count, so that $\mathbb{E}[\text{count}|X, \text{exposure}] = \text{exposure} \times e^{\eta(X)}$. Two policies with identical features but different durations then get the same *rate*, scaled by their own exposure — which is exactly the comparison you want.

> [!TIP]
> Whenever you are modeling counts (claims, calls, visits) across units with different observation windows, check whether you need an exposure offset before reaching for a fancier model — it is usually the single highest-leverage fix, and a much simpler one than switching algorithms.

This connects directly back to the resampling warning earlier in this document: resampling changes $\mathbb{E}[Y_{resampled}|X]$ away from the true $\mathbb{E}[Y|X]$, which silently breaks the probability/rate estimate. Exposure offsets are the standard, principled way actuaries keep frequency rates comparable across policies *without* resorting to resampling — they correct for a known, measurable difference (duration) rather than an artificial one introduced by rebalancing the dataset.

## Computing the oracle functions

### Estimating the risk with the empirical counterpart

Let remind that our goal is to find a function  $g: \mathcal{X} \to \mathcal{Y}$ which minimizes the risk $\mathcal{R}_\mathbb{P}(g)= \mathbb{E}_\mathbb{P}[l(Y, g(X))]$.  Given that the distribution $\mathbb{P}$ is unknown, both the risk $\mathcal{R}_{\mathbb{P}}(g)$ and the oracle function are unknown. 

For computing purpose, we will replace $\mathcal{R}_{\mathbb{P}}(g)$ by its empirical counterpart:

$$
\hat{\mathcal{R}}_{n}(g)= \frac{1}{n} \sum_{i=1}^{n}l(Y_i, g(X_i))
$$

Thanks to the central limit theorem (under some assumptions), we have:

$
\hat{\mathcal{R}}_{n}(g) \overset{a.s}{\underset{n \to \infty}{\rightarrow}} \mathcal{R}_{\mathbb{P}}(g)
$
and 
$
\sqrt{n}(\hat{\mathcal{R}}_{n}(g) -\mathcal{R}_{\mathbb{P}}(g)) \overset{\mathcal{L}}{\underset{n \to \infty}{\rightarrow}} \mathcal{N}(0, \mathbb{V}[l(Y,g(X))]
$

which mean that our empirical risk $\hat{\mathcal{R}}_{n}(g)$ is asymptotically $\mathcal{O}(1/\sqrt{n})$ far from its mean $\mathcal{R}_{\mathbb{P}}(g)$. 

In order to minimize the risk, we can  minimize the empirical counterparts which is a "good" approximation and it is intuitive to consider the machine learning algorithm : 

$$\hat{g}_{n, \mathcal{G}}=\underset{g \in\mathcal{G}}{\argmin}\space \hat{\mathcal{R}}_{n}(g)$$

where $\mathcal{G}$ is a subset of $\mathcal{F}(\mathcal{X}, \mathcal{Y})$. 

We can decompose the risk in two (positive!) parts: 

$$
\mathcal{R}_{\mathbb{P}}(\hat{g}_{n, \mathcal{G}})-\mathcal{R}_{\mathbb{P}}(g_{\mathbb{P}}^*)=\underbrace{\mathcal{R}_{\mathbb{P}}(\hat{g}_{n, \mathcal{G}})-\mathcal{R}_{\mathbb{P}}(\hat{g}_{\mathbb{P}, \mathcal{G}}^*)}_{\text{stochastic error}\simeq \text{variance}}+\underbrace{\mathcal{R}_{\mathbb{P}}(\hat{g}_{\mathbb{P}, \mathcal{G}}^*)-\mathcal{R}_{\mathbb{P}}(g_{\mathbb{P}}^*)}_{\text{approximation error}\simeq \text{bias}}
$$

This is called the **bias-variance dilemma**. 

> [!NOTE]
> **Bias** is a *systematic* miss — your model is consistently off in the same direction because $\mathcal{G}$ is too restrictive to capture the true pattern (e.g. fitting a straight line to a curved relationship). **Variance** is *instability* — retrain on a slightly different sample and you get a noticeably different model, because $\mathcal{G}$ is flexible enough to chase noise in that particular sample.

There is a balance to find: counterintuitively, we do NOT want to have $\mathcal{G}=\mathcal{F}(\mathcal{X}, \mathcal{Y})$ because there are an infinite number of "bad" functions that minimizes the empirical risk (e.g a functions which maps exactly each $x_i$ to $y_i$ and set 0 elsewhere) which leads to **overfitting**. In practice, we want $\mathcal{G}$ "big enough" so we will be able to approximate any function, and "small enough" to avoid generalization issue and keep the function "smooth".

The figure below makes this concrete on a toy regression problem: as model complexity (here, polynomial degree) increases, train error keeps decreasing but test error eventually turns back up — that turning point is where variance starts to dominate bias.

![Bias-variance tradeoff: train vs test error](assets/bias_variance_tradeoff.png)

## Applied consequences when training a machine learning model

Reminder: we want to solve the optimisation problem

$$
\underset{g \in\mathcal{G}}{\argmin}\space \frac{1}{n} \sum_{i=1}^{n}l(Y_i, g(X_i))
$$

Assume that we already computed an estimator. What can we modify to get a better one ? 

### What can we optimise ?

#### Reduce the variance : Optimise the search "inside" $\mathcal{G}$

There are many possibilities that we will explore in the rest of this course, including: 
1. **Change the optimizer** method to make convergence faster / more precise, taking into account the loss function. For example, LBFGS (a quasi-Newton method) typically converges faster and more precisely than plain gradient descent for smooth, convex losses like the ones behind GLMs, because it uses curvature information instead of a single learning rate — useful when you notice a linear model's coefficients are unstable or slow to converge. For non-convex losses (e.g. deep neural networks), variants of stochastic gradient descent (Adam, etc.) are preferred instead because second-order information is too expensive to compute at that scale.
2. **Change the loss** $l$ to a "more regular" or "more convex" function (e.g. logit / softmax instead of error rate) to make convergence easier. See "Choosing a loss function in practice" above for *which* loss to pick and *why* — the short version is that the loss should match the assumed distribution of $Y|X$ (Gamma for severity, Poisson for frequency, log-loss for classification), not just be picked for numerical convenience.
3. **Reformat features** to help convergence : for instance, discretizing continuous features may help by reducing the $\mathcal{X}$ space  
4. **Add regularisation hyperparameter** (e.g. l1 - lasso regression, l2 - ridge regression) to "smoothen" the $\mathcal{G}$ space. **L1 (lasso)** pushes some coefficients exactly to zero, which is valuable when you want a *sparse, interpretable* set of tariff variables (fewer coefficients to justify to a regulator or a pricing committee). **L2 (ridge)** shrinks all coefficients toward zero without eliminating any, which is preferable when features are correlated (e.g. several closely related geographic variables) because it spreads the effect across them instead of arbitrarily picking one and zeroing the rest — this makes the fit more stable across resamples (lower variance) at the cost of a small, controlled increase in bias.
5. Convexify the loss by **using bagging estimator**: 
The key idea is that if we have $C_1, ..., C_T$ classifiers with the same distribution of mean $m$ and variance $\sigma^2$, then we can consider the average $C=\frac{1}{T}\sum_{k=1}^{T}C_k$. We have :

- $\mathbb{E}[C]=\frac{1}{T}\mathbb{E}[\sum_{k=1}^{T}C_k]=\frac{1}{T}\sum_{k=1}^{T}\mathbb{E}[C_k]=\frac{T}{T}m=m$
- $\mathbb{V}[C]=\frac{1}{T^2}\mathbb{V}[\sum_{k=1}^{T}C_k]=\frac{1}{T^2}(\sum_{k=1}^{T}\mathbb{V}[C_k]+2 \sum_{1 \leq i < j\leq T} Cov(C_i, C_j))=\frac{\sigma^2}{T} + \frac{2}{T} \sum_{1 \leq i < j\leq T} Cov(C_i, C_j)$ so if the correlation of the $C_k$ is "low" (which can be achieved by randomly removing rows and columns in the original dataset, or by downsampling), the variance is greatly reduced. In the limit case where they are independent, the variance is divided by T while the bias stay the same. For this reason it is highly recommended to create a bagged estimator when the estimator is unstable — in practice, this means bagging (and its relatives, random forests and boosting) helps the most for high-variance, unstable base learners such as a single deep decision tree, and helps much less for an already-stable, low-variance model such as a well-regularized linear GLM.

6. **Resampling** (down, up, SMOTE) an unbalanced dataset to help convergence. Notice that it changes the distribution: instead of estimating $\mathbb{E}[Y|X]$ we estimate $\mathbb{E}[Y_{resampled}|X]$, and $\mathbb{E}[\mathbb{E}[Y|X]]=\mathbb{E}[Y] \neq \mathbb{E}[Y_{resampled}]$.

> [!WARNING]
> Resampling makes every estimated probability wrong relative to the true population. This is *not an issue if you only need to rank* (the order is preserved), but it is *a big issue if the exact level of probability matters* — for instance to evaluate an insurance premium, where over- or under-estimating the risk of claims directly produces a wrong premium.

#### Reduce the bias: Optimise the search globally

1. *Transform the features* to modify $\mathcal{G}$: for instance, if you use powered features or create product between features with a linear model, you now allow to search in a polynomial space. Notice that it does NOT create information, and it is not useful for all algorithm : for a tree a monotonic transformation of a continuous feature will not help. 
2. **Explore new spaces $\mathcal{G}$ by changing algorithm** (e.g. GLM, random forest, SVM, gradient boosted trees...)
3. **Explore new spaces $\mathcal{G}$ by changing hyperparameter** of a given family of algorithm : for instance if you are using boosted trees with fixed tree depth, you can change the tree depth and retrain to see if this new class $\mathcal{G}$ offer better predictive performance. This process is referred to as **finetuning hyperparameters**. As general guidance (not tied to any specific dataset): shallow trees (small `max_depth`) underfit complex interactions but are stable; deep trees fit more but overfit faster and need more regularization (subsampling rows/columns, minimum samples per leaf) to stay reliable. Systematically searching this hyperparameter space (grid search, random search, or smarter methods) is exactly "exploring $\mathcal{G}$" — you are asking whether a *different* function class fits better, not just whether the *same* class was optimized well.

**Important**:  

> [!IMPORTANT]
> - **Parametric algorithms** (e.g GLM) make assumptions about the underlying distribution (the relation between X and Y is linear), which makes them **more underfitting prone** and makes more important the work on the features spaces $\mathcal{X}$ to redefine a good underlying space $\mathcal{G}$.
> - On the opposite **non parametric algorithm** like tree based methods or neural network makes very little assumptions on the underlying distribution. It makes them much more prone to overfitting, but also able to fit very complex distribution and less biased in many situations. Regularisation is very important for these kind of algorithms. 
> - Concretely: a linear/GLM-style model on raw features will systematically underfit a relationship with sharp thresholds or interactions (high bias, low variance across resamples), while a tree-based model can capture that same relationship almost exactly on the training set but may swing wildly on a slightly different sample if left unregularized (low bias, high variance). Neither is "better" in the abstract — it depends on how much signal vs. noise is in your data and how much you can regularize the flexible model.

4. Change $\mathcal{X}$ : Add new features to your dataset gives more information, hence the conditional expectation of Y given X is necessarily greater. It has often much more impact on the end result than changing the space $\mathcal{G}$. You can try for instance different aggregates on different time periods, look for new data sources...

#### Redefine the problem to make it easier

- **Change the loss function $l$** : Depending on the problem, you may have different optima with different loss functions (see "Choosing a loss function in practice" above for concrete guidance — e.g. Poisson deviance for frequency, Gamma deviance for severity, log-loss for classification). Some metrics tend to flattten quickly (e.g. AUC is insenstitive to the absolute value of probabilities if the order of instances remain unchanged : it tends to "plateau" quickly while the convergence is not fully finished). It is recommended to **use several evalution metrics** to detect these different optima.
- For classification problems, it can be useful to **finetune the threshold** because many metrics are very threshold sensitive (e.g. precision, recall, accuracy) and the default 50% threshold may be *very* inadequate for your problem. For instance for an imbalanced dataset with 1% fraud, a 5% fraud prediction may be very high regarding the rest of the data, and worth investigating.

  Concretely, on the `claims.csv` `fraudulent` column (about 23% of claims are fraudulent in this data), a simple classifier at the default 0.50 threshold reaches 79% accuracy but a recall of only 7% — it is barely catching any fraud, because with a 23% base rate, always predicting "not fraudulent" already gets you most of the way to a high accuracy score. Lowering the threshold to around 0.15 (the point that maximizes F1 here) instead gives 93% recall at the cost of precision dropping to 27% (a lot more false positives to investigate). Neither threshold is "correct" in the abstract — it depends on whether a missed fraud or a wasted investigation costs your business more, which is a business decision, not a modeling one.

  > [!CAUTION]
  > There is no universally "correct" threshold — every choice trades missed fraud against wasted investigations. Get the relative costs from the business *before* optimizing a threshold, not after.

  ![Precision/recall/accuracy vs threshold on claims.csv](assets/threshold_tradeoff.png)
- **Evaluate differently** : When data are time sensitive, the distribution can change over time which violates our hypothesis. It is strongly recommended to create a train / test split  not at random but "time based" (i.e. train on one year and test on the following year) to detect these distribution shift (e.g. claims cost increase due to inflation).
- **Change $\mathcal{Y}$** : A problem can be framed in several way, which impact the underlying modelling hypothesis. For instance, the risk to churn (terminate your contract) will be identified by different features if we predict churn in 1 month (likely "hot data" like recent claims, call to the call center, bad NPS, people with recent preimums increase...) vs 1 year (likely "cold data" like population with "above the market" premiums).

## Worked example: pricing auto/home claims from raw claims data

To see several of the ideas above act together on real (if small) data, this section walks through `data/01_raw/claims.csv`: 1,100 auto and home insurance claims with columns such as `claim_type` (Material only / Injury only / Material and injury), `claim_area`, `incident_cause`, `police_report`, `claim_amount`, `total_policy_claims`, and a `fraudulent` flag. All numbers below come from actually running the analysis, not from illustrative guesses.

**Business problem**: estimate how much a claim is likely to cost (severity), and separately, flag claims that are more likely to be fraudulent so they can be routed for investigation. Two different targets, and as the loss-function section above argues, they call for two different losses.

**Severity is bimodal, not a single smooth distribution.** Grouping `claim_amount` by `claim_type` gives:

| `claim_type` | Mean | Median | Count |
|---|---|---|---|
| Material only | ~$2,065 | ~$2,090 | 620 |
| Injury only | ~$26,780 | ~$27,520 | 185 |
| Material and injury | ~$28,885 | ~$28,150 | 230 |

A model that predicts a single global average severity for every claim is implicitly assuming one distribution for all three groups — but "Material only" claims are an order of magnitude cheaper than claims involving injury. This is exactly the right/skewed, segment-dependent severity pattern the "Choosing a loss function in practice" section warned about.

**Segmenting changes the loss, dramatically.** Comparing a "predict the global mean for everyone" model against a "predict the mean for this claim's `claim_type`" model on the same data:

| Model | MSE | MAE |
|---|---|---|
| Global mean only | ~189,300,000 | ~$12,430 |
| Mean per `claim_type` | ~28,000,000 | ~$3,190 |

Segmenting by `claim_type` alone — no fancy model, just a smarter feature — cuts MSE by about 85% and MAE by about 74%. This is the "bias-reduction: add features" lever from earlier in this document, made concrete: the single biggest gain here came from adding one categorical feature, not from a more sophisticated algorithm. It also illustrates why squared error can be misleading in isolation: MSE is dominated by the few large injury claims, so a model that only reduces MSE might still be a poor fit for the much more frequent "Material only" claims — checking MAE (and the per-segment breakdown) alongside MSE avoids that blind spot.

**The `fraudulent` column needs its own loss and its own threshold.** About 23% of claims in this data are flagged as fraudulent — enough imbalance to matter, but not extreme. Fraud rates also differ by `claim_type` (about 14% for "Injury only" vs about 25% for the two "Material..." categories), which is itself a useful feature. As shown in the "finetune the threshold" example above, a classifier trained on this column behaves very differently at different thresholds: high accuracy but poor recall at the default 0.50, versus much higher recall (catching far more real fraud) at a lower threshold — at the cost of more false positives. Picking between them is a business tradeoff (cost of a missed fraud vs. cost of an unnecessary investigation), not something the loss function alone can decide.

**Takeaway.** Nothing in this example required an exotic algorithm: group means and a plain logistic regression were enough to see (a) why the loss/target framing has to match the shape of the data, (b) how much a single well-chosen feature can reduce error compared to a smarter algorithm on the same features, and (c) why the decision threshold is a separate lever from the loss function. When you get to the practical exercises in this module, you will build on exactly these same ideas — loss choice, feature engineering, and threshold tuning — on a different (and richer) insurance dataset.

## Quick decision cheat-sheet

**Which loss for which problem:**

| Problem | Recommended loss | Avoid |
|---|---|---|
| Claim frequency (counts) | Poisson deviance (with an exposure offset) | Squared error (ignores non-negativity and variance-mean link) |
| Claim severity (average cost) | Gamma deviance, or MAE | Squared error alone (dominated by a few large claims) |
| Pure premium (frequency × severity) | Tweedie deviance | Fitting one Gaussian model to a zero-inflated, skewed target |
| Fraud / lapse / churn classification | Log-loss (train), then tune a threshold for the business decision | 0-1 loss for training (not differentiable) |
| Reserving percentiles / robust regression | Quantile (pinball) loss, or MAE/Huber | Squared error if outliers should not dominate the fit |

**Which lever, bias or variance, and when:**

| Lever | Targets | Use it when |
|---|---|---|
| Change the optimizer | Variance (convergence quality) | Coefficients are unstable or slow to converge under the current solver |
| Match the loss to the data's distribution | Bias (correctness of the objective) | The target is skewed, non-negative, a count, or a probability — see the loss table above |
| Reformat / discretize features | Variance (smaller, smoother $\mathcal{X}$) | A continuous feature is noisy or has a non-monotonic relationship with $Y$ |
| L1 regularization | Variance, at some bias cost | You need a sparse, interpretable set of coefficients |
| L2 regularization | Variance, at some bias cost | Features are correlated and you want a stable, non-arbitrary fit |
| Bagging / random forests / boosting | Variance | The base learner (e.g. a single deep tree) is unstable across resamples |
| Resampling | Changes the target distribution (not free) | You only need ranking, not calibrated probabilities — otherwise prefer exposure/offsets or class weights |
| Add features | Bias | You suspect there is signal in the data that $\mathcal{X}$ doesn't capture yet (as in the `claim_type` example above) |
| Change algorithm / hyperparameters | Bias and variance together | You want to explore a genuinely different function class $\mathcal{G}$, not just re-optimize the current one |
| Change the loss or the target $\mathcal{Y}$ | Redefines the problem entirely | The business question itself is better answered by a different framing (e.g. severity vs. frequency vs. pure premium; churn in 1 month vs. 1 year) |
| Tune the decision threshold | A separate lever from the loss | Classification metrics (precision/recall/accuracy) are threshold-sensitive and the default 0.50 doesn't match the business cost of errors |
| Time-based evaluation | Detects distribution shift | Data is time-sensitive (e.g. inflation, seasonality, changing risk pools) |

