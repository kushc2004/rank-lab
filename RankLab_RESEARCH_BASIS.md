# RankLab — Research Basis and Design Decisions
## Source-backed rationale for the implementation specification

This file records the reasoning behind the RankLab implementation plan so that a coding agent does not silently change the methodology.

---

# 1. Why KuaiRand is the primary dataset

KuaiRand was released specifically to address exposure bias in recommender-system research.

Official project:

https://kuairand.com/

Paper:

https://arxiv.org/abs/2208.08696

Official repository:

https://github.com/chongminggao/KuaiRand

The dataset contains recommendation interactions from Kuaishou and includes random interventions inserted into standard recommendation feeds.

The official documentation states that, during random exposure, a recommended video can be replaced by a video randomly sampled from an item pool. It provides:

- user IDs,
- timestamps,
- user features,
- item features,
- multiple feedback signals,
- a flag indicating random intervention,
- separate random and standard logs.

This is materially better for an exposure-bias project than MovieLens or ordinary review datasets, where non-exposure and dislike are confounded.

---

# 2. Why KuaiRand-Pure is V1

Official KuaiRand guidance says:

- KuaiRand-27K / 1K are better for rigorous sequential/OPE/RL use,
- KuaiRand-Pure is appropriate when full sequential logs are unnecessary and for debiasing / collaborative-filtering research.

KuaiRand-Pure is small enough for fast iteration but retains the random candidate pool.

Official statistics reported by KuaiRand:

```text
KuaiRand-Pure:
27,285 users
~7,551 standard items
~1.44M standard interactions
~1.19M random interactions
30 user features
62 item features
12 feedback signals
```

This supports the core research question without requiring tens of gigabytes.

---

# 3. Why KuaiRand-1K is only a scale extension

KuaiRand-1K has:

```text
1,000 users
~4.37M items
~11.7M interactions
~43K random interactions
```

The very large item corpus is excellent for retrieval/ANN scaling, but the randomized sample is much smaller.

Therefore:

```text
Pure -> debiasing / randomized evaluation
1K   -> large-catalog retrieval scaling
```

---

# 4. Why the primary outcome is long_view

KuaiRand's official log description notes that `is_click` has different semantics under different UI modes.

`long_view` has an explicit duration-based definition.

Therefore `long_view` is a cleaner primary engagement target across scenarios.

Other actions such as:

```text
like
follow
comment
forward
```

should be treated as secondary outcomes.

---

# 5. Why random-exposure evaluation matters

Ordinary implicit recommendation logs are Missing-Not-At-Random: feedback is observed only for exposed items, and exposure is chosen by an existing recommender.

KuaiRec / KuaiRand papers explicitly study the effect of exposure/data missingness on recommender evaluation.

KuaiRec paper:

https://arxiv.org/abs/2202.10842

KuaiRand paper:

https://arxiv.org/abs/2208.08696

The project therefore evaluates model quality separately on:

```text
standard recommendation interactions
random intervention interactions
```

The goal is not to call random data “perfect preference ground truth.”

The goal is to reduce dependence on the historical recommender's exposure mechanism.

---

# 6. Why exact IPS is NOT the main KuaiRand method

Inverse Propensity Scoring requires probabilities from the policy that generated each logged action, or another justified propensity model.

The KuaiRand random intervention mechanism is documented, but this does not justify inventing exact action propensities for the full standard recommender.

Therefore the project does not claim:

```text
exact IPS correction of KuaiRand standard logs
```

Instead it uses:

```text
randomized-target density-ratio / importance weighting
```

to shift the observed standard-domain covariate distribution toward the random-domain covariate distribution.

Exact logged-policy OPE is moved to Open Bandit Dataset.

---

# 7. Why density-ratio weighting is defensible here

Suppose:

```text
D=0 -> standard-exposure domain
D=1 -> randomized-exposure domain
```

A probabilistic classifier estimates:

```text
q(x) = P(D=1 | x)
```

Classifier odds, adjusted for the sampling priors used to construct the domain-classification dataset, estimate a density ratio:

```text
p_random(x) / p_standard(x)
```

These weights can be used for covariate-shift-style reweighting.

This should be described as:

```text
randomized-target importance weighting
```

not as exact logging-policy propensity scoring.

Diagnostics must include:

```text
weight distribution
clipping
effective sample size
```

---

# 8. Why Two-Tower retrieval

Large recommenders normally cannot score every item with an expensive ranking model.

Two-Tower models independently encode:

```text
user/context
item
```

into a shared embedding space, enabling fast nearest-neighbor candidate retrieval.

TensorFlow Recommenders' official retrieval tutorial describes this exact query-tower / candidate-tower setup and ANN export:

https://www.tensorflow.org/recommenders/examples/basic_retrieval

The project implements the architecture in PyTorch so that it integrates naturally with the rest of the stack.

---

# 9. Why FAISS

FAISS is Meta/FAIR's dense-vector similarity-search library.

Official sources:

https://github.com/facebookresearch/faiss

https://faiss.ai/

The project uses:

```text
IndexFlatIP
```

for exact inner-product validation on KuaiRand-Pure.

Approximate indices are evaluated only on a larger corpus where latency/recall trade-offs are meaningful.

---

# 10. Why BPR remains a required baseline

BPR is a classic pairwise ranking objective for implicit-feedback recommendation.

Paper:

https://arxiv.org/abs/1205.2618

A 2024 replicability study also shows that properly tuned BPR remains a strong baseline, reinforcing that it should not be treated as a weak strawman.

The project uses BPR before the neural retrieval model.

---

# 11. Why exposed negatives matter

Unobserved items are not equivalent to disliked items.

A user may never have had a chance to interact with them.

KuaiRand provides exposure logs, allowing the project to construct negatives from items that were actually shown but did not produce the target engagement.

The project therefore compares:

```text
random unobserved negatives
vs
exposed negatives
```

This is a meaningful exposure-bias experiment.

---

# 12. Why LightGBM LambdaRank

The ranker should optimize list quality rather than ordinary pointwise classification only.

Official LightGBM documentation currently supports:

```text
objective = lambdarank
metric = ndcg
label_gain
lambdarank_truncation_level
ranking groups
```

Documentation:

https://lightgbm.readthedocs.io/en/latest/Parameters.html

The project uses LightGBM because:

- it handles heterogeneous engineered features well,
- it is highly interpretable,
- it is extremely common in ranking/tabular ML,
- it complements the neural retrieval stage,
- it gives the CV a stronger Data Science rather than “all-deep-learning” profile.

---

# 13. Why standard vs randomized model ranking is a core result

KuaiRec research found that evaluation conclusions can change as missingness/exposure conditions change.

Therefore RankLab should not report one metric table only.

For every model, produce:

```text
standard-test metrics
randomized-test metrics
gap
```

and compare whether apparent model ordering changes.

This is more important than adding many architectures.

---

# 14. Why Open Bandit Dataset is a separate OPE module

Open Bandit Dataset was released to enable realistic, reproducible off-policy evaluation.

Paper:

https://arxiv.org/abs/2008.07146

Repository:

https://github.com/st-tech/zr-obp

Documentation:

https://zr-obp.readthedocs.io/

It contains fashion e-commerce recommendation data collected under multiple policies, including random and Thompson-sampling policies, with logged information needed for OPE.

This makes it appropriate for evaluating:

```text
DM
IPS/IPW
SNIPS
DR
```

against empirical target-policy values.

This separation makes the project more accurate:

```text
KuaiRand -> recommendation + exposure bias
OBD      -> exact OPE estimator validation
```

---

# 15. Why Doubly Robust is useful but not mandatory in KuaiRand

Doubly Robust estimators combine:

```text
reward/outcome model
+
importance weighting
```

and can improve bias-variance behavior in OPE.

Relevant ranking-policy reference:

https://arxiv.org/abs/2202.01562

However, without verified logging propensities, it should not be casually used to make exact KuaiRand policy-value claims.

Implement DR in the OBD module first.

---

# 16. Why popularity analysis is required

Historical recommenders create highly non-uniform exposure.

A model can improve standard offline metrics while concentrating recommendations further on head items.

Therefore report:

```text
catalog coverage
average recommended popularity
head/mid/tail recommendation share
Gini concentration
```

alongside NDCG.

This is not a moral/fairness claim.

It is a system-behavior analysis.

---

# 17. Why category calibration is optional but valuable

Calibrated recommendation asks whether the composition of a user's recommendation list reflects the composition of their interests.

Steck, “Calibrated Recommendations,” RecSys 2018:

https://dl.acm.org/doi/10.1145/3240323.3240372

KuaiRand now provides supplementary category information for videos.

Therefore RankLab can compute user historical category preferences and compare them with recommendation-list category distributions.

This provides a multi-objective:

```text
relevance vs calibration
```

trade-off.

Do not call it algorithmic fairness.

---

# 18. Why Amazon Reviews 2023 is not the primary dataset

Amazon Reviews 2023 is extremely large and includes:

- 571M+ reviews,
- 54M+ users,
- 48M+ items,
- rich item metadata,
- fine-grained timestamps.

Official project:

https://amazon-reviews-2023.github.io/

This is excellent for large-scale retrieval experiments.

But review data does not expose the full recommendation-impression process.

Therefore it cannot support the main exposure-bias claims as cleanly as KuaiRand.

Use it only as an optional scale/e-commerce extension.

---

# 19. Why the project is not “generic recommender systems”

The project is not differentiated by the Two-Tower architecture alone.

Two-Tower + FAISS is useful, but common.

The differentiator is the methodological chain:

```text
biased implicit feedback
        ↓
randomized intervention data
        ↓
standard vs randomized evaluation
        ↓
exposure-distribution diagnostics
        ↓
importance-weighted training
        ↓
cohort/popularity analysis
        ↓
independent OPE validation
```

This produces a project with clear statistical and decision-science depth.

---

# 20. Metrics and their roles

## Retrieval

```text
Recall@50
Recall@100
Recall@200
MRR
```

## Ranking

```text
NDCG@5
NDCG@10
NDCG@20
Recall@K
MAP/MRR
```

## Randomized response

```text
Random-NDCG
ROC-AUC
PR-AUC
```

depending grouping density.

## Popularity / catalog

```text
coverage
Gini
average popularity
tail exposure share
```

## Calibration

```text
Jensen-Shannon divergence
```

with smoothing.

## Statistical reliability

```text
user-level bootstrap confidence intervals
paired bootstrap differences
```

## Importance weighting

```text
weight percentiles
ESS
domain-classifier AUC
```

## OPE

```text
estimated policy value
absolute estimation error
relative error
```

---

# 21. What not to optimize for

Do not select the final model based only on:

```text
highest standard NDCG
```

The main final model should be selected from a predeclared multi-metric view:

```text
randomized NDCG
standard NDCG
coverage
weight stability
cohort robustness
```

If the weighted/debiased model does not beat the unweighted model, report that result.

A negative result is acceptable if the evaluation is rigorous.

---

# 22. Recommended final CV framing

Do not write metrics until they are real.

Potential bullets after completion:

- Built two-stage personalized recommender combining Two-Tower retrieval, FAISS candidate search and LambdaRank re-ranking over implicit-feedback interaction logs.
- Evaluated models on randomized-exposure interactions to quantify exposure-bias generalization, comparing standard and debiased ranking performance across user/item cohorts.
- Implemented randomized-target importance weighting with clipping and effective-sample-size diagnostics, reducing dependence on the historical recommendation policy's exposure distribution.
- Benchmarked off-policy estimators on real e-commerce bandit logs, comparing IPS, self-normalized and doubly robust policy-value estimates against observed policy outcomes.

These should later be compressed to the actual CV line-length limit and populated with measured numbers.

