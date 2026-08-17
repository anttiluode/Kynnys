# Gate 4 — stochastic structured-output attribution

Date: 2026-08-17

Status: **failed in three increasingly charitable stages. This is a kill gate for the current automatic black-box probe planner under noisy structured outputs.**

Gate 3 had shown that, in a deterministic scalar-output setting, a black-box planner could spend a fixed paired-probe budget more effectively than random, uncertainty sampling, or a static mixed design. Gate 4 attacked the assumptions that made that result easy:

1. output is stochastic;
2. output is structured rather than one scalar;
3. output fields are correlated;
4. all baseline/scout observations are charged;
5. intervention effects must be estimated from repeated noisy runs;
6. success requires fresh cause recovery, not merely a favorable geometry statistic.

The underlying task remained the same real 20 Newsgroups binary text classifier used in Gates 2 and 3. The deliberately hard candidate causes remained:

```text
A: retrain logistic regression with stronger regularization, C 1.0 -> 0.1
B: keep the model fixed and apply post-hoc score scaling, multiplier 0.1783495
```

On the latent deterministic margin these effects are still almost perfectly aliased:

```text
eta             0.014879
alias cosine    0.999889
```

Gate 4 does **not** use an external LLM. It wraps this real classifier in an LLM/tool-shaped stochastic structured-output layer. That distinction matters.

## Structured observable

Each stochastic run emits a five-dimensional decision vector:

```text
answer_is_space
tool_is_search
confidence_low
confidence_high
hedge
```

The fields share a noisy latent decision variable, so their covariance is deliberately non-diagonal. The benchmark therefore resembles the problem of attributing changes in a stochastic agent's answer/tool/confidence behavior without pretending each output field is independent.

Observed high-precision structured-output covariance trace:

```text
0.9741
```

## Fully charged budget

A fixed 512-candidate scout subset is drawn from the 7,103-candidate black-box pool.

Every strategy pays for one baseline structured output on each scout candidate:

```text
512 charged scout calls
```

A selected location is then measured repeatedly under baseline, A, and B. The already-paid scout baseline sample is reused, so with `r` repeats the additional cost per selected location is:

```text
3r - 1 calls
```

Forty-eight locations are selected. At the original `r=6` setting:

```text
512 + 48 * 17 = 1328 charged calls per strategy
```

No strategy receives free baseline scores.

---

# Gate 4a — raw Fisher geometry mistakes noise for information

Workflow run: `32024595417`

Artifact: `9286609938`

Configuration:

```text
candidate pool       7103
scout candidates      512
selected locations     48
repeats / variant        6
charged calls         1328
```

The active planner used the Gate-3-style weakest-Fisher-direction objective after whitening by its measured structured-output covariance.

The result was a decisive failure:

```text
                 hidden true   observed     recovery
                 s_min         s_min
active             0.5071      11.2644       0.510
uncertainty        0.4739      11.5696       0.544
static mixed       0.5507      12.1987       0.518
random             0.5103      11.8768       0.508

random truth-s_min median      0.5106
random truth-s_min p90         0.5428
random truth-s_min max         0.5766
```

The active strategy's measured geometry said `s_min ~= 11.3`; hidden high-precision truth said `0.51`.

Fresh cause recovery was essentially chance.

The exact scalar-copy negative control remained singular:

```text
s_min ~= 1.4e-15
```

### What failed

The planner was treating **estimation noise as an independent causal direction**.

For repeated stochastic measurements,

```text
eA_hat = eA + epsA - eps0
eB_hat = eB + epsB - eps0
```

where the same estimated baseline appears in both effects. If each run has covariance `Sigma` and there are `r` independent repeats, then approximately

```text
Var(eA_hat)       = 2 Sigma / r
Var(eB_hat)       = 2 Sigma / r
Cov(eA_hat,eB_hat)=   Sigma / r
```

The shared baseline therefore creates correlated estimation error, while the squared norm of each noisy effect estimate is upward biased. A raw Gram/Fisher matrix computed from noisy sample means can look far more identifiable than the latent effects actually are.

Gate 4a is an empirical receipt for that failure mode.

---

# Gate 4b — noise-bias correction is necessary, but insufficient

Workflow run used for the scientific result: `32024895128`

Artifact: `9286722621`

An earlier run `32024790165` executed the experiment but failed during result formatting because `median` was not imported. No thresholds or acquisition rules were changed; the canonical rerun used a wrapper that supplied that missing standard-library symbol.

Gate 4b kept the exact same 1,328-call budget and added three corrections:

1. estimate the structured sample covariance;
2. subtract the expected self- and cross-noise contribution from the two-cause Gram matrix;
3. use a conservative acquisition score and shrink noisy effect templates toward a regularized black-box surrogate.

The noise correction is not tiny. At the final design state, the estimated whitened bias contribution per selected location was:

```text
A self term       4.707
A/B cross term    2.354
```

Results:

```text
                  truth s_min   fresh recovery
active                0.5173        0.497
uncertainty           0.4739        0.548
static mixed          0.5507        0.513
random                0.5103        0.503

random p90            0.5428
```

Active cause recovery was asymmetric:

```text
A recovery  0.827
B recovery  0.167
mean        0.497
```

So the estimator had learned a bias toward calling the change A, not a reliable distinction between A and B.

The scalar-copy control stayed singular:

```text
s_min ~= 2.1e-15
```

### Gate 4b verdict

**Fail.**

Noise-bias subtraction removes a mathematical lie from the information estimate, but it cannot manufacture signal that six repeats never measured.

---

# Gate 4c — sample-complexity boundary sweep

Workflow run: `32025028927`

Artifact: `9286786989`

The last attacker did not tune the selector to the observed failure. Instead it fixed:

```text
same candidate pool
same 512 scout candidates
same 48 selected locations budget
same stochastic world
same active selector family
same denoised estimator family for ALL strategies
```

and changed only the number of paid repeats:

```text
r = 6, 12, 24, 48
```

Higher-repeat runs see longer prefixes of the same stochastic streams; they do not receive a luckier world.

Total charged calls become:

```text
r=6      1328 calls
r=12     2192 calls
r=24     3920 calls
r=48     7376 calls
```

The preregistered recovery criterion was 75% cause recovery. At the first repeat level where active crossed 75%, it also had to beat the best baseline by at least 5 percentage points.

## Results

### 6 repeats — 1,328 calls

```text
                 truth s_min   eta      recovery
active              0.4542    0.0551     0.535
uncertainty         0.4584    0.1501     0.500
static mixed        0.4212    0.0718     0.483
random              0.4551    0.0695     0.548
```

### 12 repeats — 2,192 calls

```text
                 truth s_min   eta      recovery
active              0.6324    0.0536     0.548
uncertainty         0.6348    0.1501     0.481
static mixed        0.5833    0.0718     0.498
random              0.6301    0.0695     0.496
```

### 24 repeats — 3,920 calls

```text
                 truth s_min   eta      recovery
active              0.8754    0.0639     0.537
uncertainty         0.8646    0.1501     0.504
static mixed        0.7949    0.0718     0.558
random              0.8580    0.0695     0.504
```

### 48 repeats — 7,376 calls

```text
                 truth s_min   eta      recovery
active              1.1323    0.0497     0.581
uncertainty         1.1476    0.1501     0.469
static mixed        1.0578    0.0718     0.508
random              1.1386    0.0695     0.433
```

The active strategy never reached the preregistered 75% recovery criterion.

Even at 48 repeats, recovery was asymmetric:

```text
A recovery  0.963
B recovery  0.200
mean        0.581
```

No strategy produced a convincing general A-vs-B decoder.

The exact scalar-copy negative control remained singular:

```text
s_min ~= 8.7e-15
```

The estimated whitened noise-bias term did decline monotonically with replication:

```text
r=6      4.729
r=12     4.537
r=24     4.170
r=48     3.668
```

Yet recovery did not track the growth in `s_min`.

That discrepancy is the key result.

---

# Interpretation

Gate 3's smallest-singular-value result was real for the deterministic scalar setting, but it does **not** transfer naively to a noisy structured observable.

There are three different things that must not be conflated:

```text
1. latent causes are mathematically non-collinear
2. an estimated information matrix has a nonzero weak direction
3. fresh noisy observations permit reliable cause classification
```

Gate 4 shows that (2) can look excellent while (3) remains at chance, and that even a noise-debiased version of (2) can improve steadily while actual cause recovery stays poor.

The five-field structured wrapper discards and quantizes much of the already tiny distinction between regularization and score scaling. More repeated samples reduce estimator variance, but they do not restore information that the observable itself barely carries.

The supported conclusion is therefore:

> **Kynnys must not admit an attribution claim merely because an estimated Fisher/identifiability score crosses a threshold. Under stochastic structured outputs, the runtime needs a direct evidence-quality or held-out recovery criterion and must be able to refuse attribution when the observable does not carry a stable distinction.**

A shorter rule is:

> **measured distinction != recoverable distinction**

and, more specifically:

> **replication reduces noise; it does not undo information destroyed by the observation map.**

This is closely aligned with the older `audit_effects` rule that low identifiability means “important but confounded,” not harmless.

# Feature verdict after Gate 4

## `audit_effects` refusal

**Still supported.**

The safe part of the audit remains the ability to say that large observed effects do not justify unique attribution.

## Automatic black-box `plan_probes(...)`

**Not earned.**

Gate 3 demonstrated a useful deterministic special case. Gate 4 shows that a generic planner would be dangerously overconfident once noisy structured outputs enter the loop unless it models estimation noise and validates actual recoverability.

Do not promote the current experiment code into the public Kynnys API.

## New candidate primitive: evidence admission

Gate 4 suggests a more Kynnys-native direction than “cleverly pick the next probe”:

```text
current evidence
    -> estimate noise / covariance
    -> estimate identifiable effect
    -> test recoverability / calibration
    -> ADMIT attribution | REPEAT | REFUSE
```

This fits the project's core semantics better than an optimizer that always returns another experiment.

A computation can exist without being worth executing.

Likewise, a distinction can appear in a measurement without being strong enough to justify a causal claim.

# Code and receipts

```text
experiments/gate4_stochastic_structured.py
experiments/gate4b_noise_corrected.py
experiments/gate4b_noise_corrected_runner.py
experiments/gate4c_budget_sweep.py
.github/workflows/gate4-stochastic.yml
.github/workflows/gate4b-noise-corrected.yml
.github/workflows/gate4c-budget-sweep.yml
```

Workflow history:

```text
Gate 4a   32024595417   FAIL — raw Fisher geometry mistook noise for information
Gate 4b   32024895128   FAIL — noise correction insufficient at same budget
Gate 4c   32025028927   FAIL — no 75% recovery crossover through 48 repeats / 7376 calls
```

Artifacts:

```text
Gate 4a   9286609938
Gate 4b   9286722621
Gate 4c   9286786989
```

# Next move

Do **not** increase repeats again merely to seek a green badge.

The stronger next experiment is to turn the failure into an admission rule:

1. estimate whether an observed cause distinction is above the empirical noise floor;
2. predict whether another unit of replication can materially improve recovery;
3. compare three actions: `CLAIM`, `REPEAT`, `REFUSE`;
4. score calibration — false causal claims must be strongly penalized;
5. test both a recoverable stochastic pair and the deliberately unrecoverable Gate-4 pair.

If that works, the audit subsystem stops being “an active learner bolted onto Kynnys” and becomes something much closer to the central programming idea:

> **admit only the claims that the available evidence can support.**
