# Gate 3 — black-box attribution

Date: 2026-08-17

Status: **Gate 3a failed; Gate 3b passed; Gate 3c replicated the pass across eight warm-start seeds.**

Gate 3 removed the model-internal access used by Gate 2c. The selector received only raw candidate text, baseline black-box scores, and budgeted paired outcomes for probes it chose. It did not receive classifier coefficients, gradients, the classifier vectorizer, or hidden full-pool A/B effect arrays.

The hard pair stayed the same: logistic-regression regularization `C=1.0 -> 0.1` versus post-hoc score scaling `0.1783495`. On ordinary held-out traffic:

```text
eta            0.014879
alias cosine   0.999889
```

The probe pool contained 7,103 ordinary and synthetic corpus-derived texts. Paid paired-probe budget: 48.

## Gate 3a — eta was the wrong objective

A hashed-text surrogate, learned only from paid paired probes, tried to seek departures from the shared direction.

```text
ACTIVE       eta=0.015003  S(A)=6.4759  S(B)=6.5176
UNCERTAINTY  eta=0.888452  S(A)=0.0471  S(B)=0.0132
RANDOM p90   eta=0.024954
```

Gate 3a failed.

The failure exposed a metric problem. Uncertainty sampling obtained a beautiful angle by driving one effect almost to zero. High eta alone can therefore reward an identifiable but practically tiny signal. This is exactly why Kynnys keeps sensitivity and identifiability separate.

Workflow: `32011258497`.

## Gate 3b — weakest Fisher direction

For a two-cause additive attribution problem, define the effect design matrix

```text
J = [g_A  g_B]
```

Under isotropic observation noise the weakest Fisher-information direction is governed by the smallest singular value `s_min(J)`. This penalizes both collinearity and vanishing effect magnitude.

The black-box selector was changed, before rerunning, to choose the next paid probe by predicted gain in that weakest direction.

```text
                   s_min      eta
Fisher active      0.065373   0.014765
uncertainty        0.011589   0.888452
static mixed       0.036717   0.023132
random median      0.043510      -
random p90         0.049713      -
random max         0.055877      -
```

The active selector beat uncertainty, static mixed, random p90, and the preregistered absolute `s_min >= 0.05` threshold. An exact scalar-copy negative control remained singular (`s_min ~1.3e-15`).

Gate 3b passed. Workflow: `32011447279`.

## Gate 3c — replication

The same black-box Fisher strategy was rerun with eight independent warm-start seeds.

```text
seed 101   s_min 0.066929
seed 211   s_min 0.064556
seed 307   s_min 0.069870
seed 419   s_min 0.064399
seed 503   s_min 0.062116
seed 607   s_min 0.062563
seed 709   s_min 0.062186
seed 811   s_min 0.063944
```

Summary:

```text
ACTIVE
min s_min        0.062116
median s_min     0.064171
max s_min        0.069870
median eta       0.013795

RANDOM, 400 SETS
median s_min     0.043723
p90 s_min        0.049262
max s_min        0.058845

STATIC MIXED, 400 SETS
median s_min     0.037223
p90 s_min        0.042018
max s_min        0.047106

UNCERTAINTY
s_min            0.011589
```

All 8/8 active runs beat the random 90th percentile. All 8/8 beat the static-mixed 90th percentile and uncertainty sampling. The scalar-copy control remained singular (`s_min ~7.7e-16`).

Gate 3c passed every preregistered criterion. Workflow: `32011588081`.

## Interpretation

The positive result is narrower than “automatic experiment design.” The selector did **not** rotate the effects far apart; eta remained about 0.014 and condition numbers stayed large. Instead, it learned to collect enough **absolute independent signal** to strengthen the weakest information direction.

The supported statement is:

> **When ordinary traffic does not support unique attribution, this controlled black-box pipeline can use a small adaptive paired-probe budget to collect more independent attribution information than random, uncertainty, or a static mixed design.**

Important limits remain: paired counterfactual access is assumed; baseline candidate scores were not charged to the 48-probe budget; the model is deterministic and low-dimensional; the candidate causes are known in advance; and the result has not yet survived stochastic structured outputs such as LLM/tool pipelines.

## Code

```text
experiments/gate3_blackbox_probe.py
experiments/gate3b_blackbox_fisher.py
experiments/gate3c_blackbox_replication.py
.github/workflows/gate3-blackbox.yml
.github/workflows/gate3b-fisher.yml
.github/workflows/gate3c-replication.yml
```

Next serious attacker: a stochastic structured-output pipeline with measured noise covariance and all probe costs charged.
