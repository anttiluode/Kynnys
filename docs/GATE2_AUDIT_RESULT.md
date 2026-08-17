# Gate 2 — confounded attribution and separating measurements

Date: 2026-08-17

Status: **run in three stages. The final workflow remains red by design; the attribution result itself is positive but narrower than the original claim.**

Gate 2 asked whether Kynnys's `audit_effects` can do something useful that ordinary profiling often does not: distinguish **large effect** from **identifiable effect**, refuse false blame when two pipeline changes look the same in the observed outputs, and point toward a measurement that separates them.

The experiment uses a real text-classification pipeline, not hand-written collinear vectors.

## Pipeline

Dataset: scikit-learn 20 Newsgroups, binary task:

```text
rec.autos
sci.space
```

Observed run geometry:

```text
train examples   1187
test examples     790
TF-IDF features 27138
baseline model   logistic regression, C=1.0
```

Two genuinely different changes were then constructed:

1. **Regularization change** — retrain the classifier with stronger L2 regularization.
2. **Temperature/score scaling** — leave the classifier intact and multiply its decision score after inference.

The regularization path was searched only to construct a deliberately hard attribution case: choose a non-trivial regularization change whose output signature most closely resembles a global score rescaling. The chosen model was:

```text
C: 1.0 -> 0.1
fitted post-hoc score multiplier: 0.1783495
```

This is a stress test of the audit, not an accuracy-tuning exercise.

## Gate 2a — can the audit detect the confounder?

On 320 natural held-out documents:

```text
alias cosine                  0.999889
eta(regularization)           0.014879
eta(temperature)              0.014879
regularization sensitivity    9.9857
temperature sensitivity       9.9846
```

So both changes are **large** and almost perfectly **non-identifiable** from those observed score changes.

`audit_effects` correctly reports both as confounded.

That is the first useful receipt:

> **large effect != identifiable cause**

A naïve norm-based attribution would say both changes strongly affect the system. The audit says the stronger thing: the measurement geometry does not support deciding which one caused a score-shrinkage pattern.

### The first pass criterion was too weak

The first experiment then searched a disjoint pool of ordinary test documents for the 24 examples with the largest residual departure from the shared direction.

Result:

```text
best 24 ordinary examples eta       0.019412
alias cosine                         0.999812
random-24 median eta                 0.012438
random-24 p90 eta                    0.015474
```

The targeted set beat random and therefore satisfied the first written gate.

We rejected that pass.

`eta=0.019` is still essentially unidentifiable. "Better than random" is not the same as "separating the causes."

This matters because a benchmark could otherwise manufacture a positive result by choosing an easy relative baseline while the absolute geometry remained useless.

## Gate 2b — tighten the criterion and add active one-token probes

The gate was tightened before rerunning:

```text
meaningful separation requires eta >= 0.25
```

Instead of merely reselecting natural documents, the next instrument chose unigram features where the regularized model's weights depart most from a pure scaling explanation and queried one-token diagnostic documents.

Result:

```text
active one-token probe eta       0.063095
alias cosine                     0.998008
random-probe median eta          0.025593
random-probe p90 eta             0.037940
```

The active probes were genuinely better than random, but still nowhere near the preregistered meaningful-separation threshold.

**Gate 2b failed.**

That is an important negative result. The ordinary output space was so close to one-dimensional that merely selecting more diagnostic examples did not create enough new information.

This is exactly the epistemic rule inherited from TransientWaveCompiler:

> **more optimization of the same measurement cannot create a distinction that the measurement barely contains.**

## Gate 2c — null the shared direction

The next hypothesis was stronger and was written before the run:

> If the two effects share an almost one-dimensional direction, construct inputs that suppress that shared direction while exciting the residual direction.

For this linear text model, positive temperature scaling acts through the baseline decision margin. Gate 2c therefore generated **20,025 two-token / repeated-token diagnostic documents**, kept the **3,004** with unusually small baseline-margin magnitude, and within that near-null pool chose 24 probes spanning both positive and negative residual departures from the shared scaling explanation.

The resulting probes included examples such as:

```text
how thanks thanks thanks
how got
thanks think
from today
discussion we
people information
```

These are diagnostic stimuli, not natural-language benchmark examples. Their job is to excite the model geometry.

### Gate 2c result

Natural held-out outputs were still the same hard confounder:

```text
natural cosine       0.999889
natural eta          0.014879
```

The orthogonalized 24-probe set produced:

```text
eta(regularization)  0.450445
eta(temperature)     0.450445
alias cosine         0.892804
```

Matched random controls drawn from the same near-null candidate pool:

```text
random median eta    0.317556
random p90 eta       0.389683
```

So the designed probe set exceeded both the preregistered `eta >= 0.25` requirement and the random 90th percentile.

Observed sensitivities on the probe set remained non-trivial:

```text
regularization sensitivity   0.303894
temperature sensitivity      0.215861
```

An exact scalar-duplicate negative control remained completely unseparable:

```text
alias cosine          1.0
eta                    ~1.66e-16
residual energy        ~1.24e-32
```

That control matters: the probe construction did not magically manufacture independent information when the two effects were literally scalar copies.

## Why the Gate 2c workflow is still red

The workflow had one additional auxiliary check:

```text
chosen median |baseline margin| <= near-null pool median |baseline margin|
```

Observed:

```text
chosen median |margin|     0.050334
pool median |margin|       0.042686
```

So that check failed and the job correctly exited non-zero.

We did **not** rewrite the criterion after seeing the result.

In retrospect, the check is not a very good measure of whether the shared direction was suppressed: the candidate pool had already been restricted to the lowest 15% of baseline-margin magnitudes. Requiring the residual-extreme probes to also fall below the *median of that already-near-null pool* adds a second ordering constraint that is unrelated to whether attribution actually improves.

Nevertheless, the red receipt is preserved. A future confirmatory gate should specify a better nulling metric before execution rather than retroactively turning this run green.

## Downstream consequence of the confounder

The two changes were not literally the same program.

On the full test set:

```text
                    accuracy   log loss   coverage
baseline             0.9038     0.4766     0.7494
regularization       0.8987     0.6462     0.0392
temperature          0.9038     0.6458     0.0329
both                 0.8987     0.6845     0.0000
```

The score-level behavior of regularization and temperature scaling was almost perfectly collinear even though regularization slightly changed classification accuracy while positive temperature scaling did not.

That is exactly the type of situation where "which optimization caused this?" can be ill-posed under one observable and answerable under another.

## Verdict

### `audit_effects` as a refusal mechanism

**Pass.**

The strongest result of Gate 2 is not the synthetic probe construction. It is that the audit correctly says:

> these two effects are both important, but the natural measurement does not identify them separately.

That is a real and useful distinction.

### Passive "here is the example that separates them" recommendation

**Fail.**

Selecting the best ordinary held-out examples raised eta only from about `0.015` to `0.019`. That is not meaningful separation.

### Simple active diagnostic probes

**Fail, but informative.**

One-token coefficient-residual probes reached eta `0.063`, better than random but still strongly confounded.

### Orthogonalized active measurement

**Promising controlled result, not yet a general feature.**

By explicitly suppressing the shared direction and exciting the residual one, the probe set raised eta to `0.450` and reduced cosine to `0.893`, beating matched random controls.

However, this construction used **model internals** (linear coefficients) and synthetic diagnostic text. It therefore does not yet justify a generic Kynnys API that promises to invent the next experiment for arbitrary black-box LLM/tool pipelines.

## What changed in our belief

Before Gate 2 it was tempting to say:

> `audit_effects` can tell you that A and B are confounded and then tell you what paired trial separates them.

After Gate 2 the defensible statement is narrower:

> **`audit_effects` can reliably expose when the current observations do not support unique attribution. In a controlled differentiable/inspectable pipeline, that geometry can guide the construction of a more informative measurement, but automatic black-box trial design is not yet earned.**

That is still interesting.

It also reinforces a recurring Kynnys theme:

```text
computation exists != computation should execute
large effect exists != cause is identifiable
state exists != current observation is sufficient
```

The common object is a threshold on what the available evidence actually permits the system to do or claim.

## Code and receipts

```text
kynnys/audit.py
experiments/gate2_audit_text_pipeline.py
experiments/gate2c_orthogonal_probe.py
.github/workflows/gate2-audit.yml
.github/workflows/gate2c-orthogonal.yml
```

Gate 2a workflow run: `32009241650` — completed success under the original permissive gate, later rejected as scientifically too weak.

Gate 2b workflow run: `32009473290` — failed the tightened eta threshold.

Gate 2c workflow run: `32009809400` — attribution and random-control criteria passed; overall workflow failed the preserved auxiliary nulling-median condition.

## Next gate

Do **not** promote an automatic `suggest_trial()` API yet.

The next serious test is a **black-box attribution gate**:

1. use a pipeline whose internals are unavailable to the audit;
2. expose only input/output probes and costs;
3. create two real pipeline changes that are confounded on ordinary evaluation traffic;
4. allow the audit to spend a small probe budget;
5. compare probe-selection strategies against random and simple uncertainty sampling;
6. require a large absolute identifiability gain, not merely a relative win;
7. preserve an exactly non-identifiable negative control.

If that works, then "here is the next experiment to run" starts to become a Kynnys feature rather than an interesting hand-designed demonstration.
