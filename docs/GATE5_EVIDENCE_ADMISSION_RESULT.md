# Gate 5 — evidence admission

Date: 2026-08-17

Status: **PASS in a controlled linear-Gaussian capability test.**

This gate tests a different object from the failed Gate 4 probe planner.

Gate 4 asked whether a black-box planner could spend stochastic probes cleverly enough to recover two nearly aliased causes. Gate 5 moves the decision boundary earlier:

> **A demand for a causal claim does not authorize measurement or attribution. It creates obligations.**

The admission policy must decide among:

```text
CLAIM
REPEAT
ROUTE
REFUSE
```

and it must do so in this order:

```text
structural capability
    -> statistic obligations
    -> whitened nuisance-residual capability
    -> finite-budget recoverability ceiling
    -> empirical confidence
    -> CLAIM | REPEAT | ROUTE | REFUSE
```

The experiment is deliberately synthetic and inspectable. It is not evidence that generic causal inference is solved. Its purpose is to test whether the Kynnys/Vahti/TWC decision primitive behaves correctly on cases where the correct action is known from the declared model.

## Prior-work transplant

The gate combines three already-existing ideas:

1. **TransientWaveCompiler topology capability audit** — check whether a proposed distinction is structurally observable before measurement. Exact aliases can be refused with zero measurement calls.
2. **TWC nuisance-aware identifiability** — whiten by measurement covariance and project the candidate contrast away from declared nuisance directions before scoring what remains.
3. **Vahti metric obligations** — a statistic must declare enough about its type that cheap obligations can be generated before its output is trusted. In particular, an evidence statistic must not reward estimator variance or ignore shared-baseline covariance.

The Gate 5 implementation is a generic linear-observation analogue, not a copy of TWC's reciprocal-filter `so(N)` gauge machinery.

## Model

Two candidate causes are represented by latent vectors `a` and `b`.

A receiver / observation map is

```text
y = C x + N beta + epsilon
```

where:

- `C` is the declared observation/readout map;
- `N` contains declared nuisance directions;
- `beta` is an unknown nuisance coefficient;
- `epsilon ~ N(0, Sigma)` is measurement noise.

The policy first whitens by `Sigma`, then residualizes the cause contrast against the whitened nuisance span:

```text
delta       = C(a-b)
W           = Sigma^(-1/2)
delta_w     = W delta
N_w         = W N
delta_perp  = (I - P_Nw) delta_w
```

The per-sample recoverable separation is

```text
||delta_perp||.
```

For the controlled equal-covariance binary Gaussian model, the best-achievable classification accuracy after `n` independent repeats is

```text
Phi( ||delta_perp|| sqrt(n) / 2 ).
```

This gives an explicit distinction between:

```text
WAIT / REPEAT: increase n while keeping C fixed
ROUTE:         change C
```

If the finite repeat budget cannot cross the declared recovery threshold but an alternate readout can, the policy returns `ROUTE`, not `REPEAT`.

## Metric contract

The deliberately broken metric is a Gate-4a-style raw Fisher/Gram statistic estimated from A/base and B/base differences sharing the same baseline.

Its declared contract says:

```text
shared baseline      yes
covariance aware     no
variance reward safe no
null centered        no
```

Before any claim-specific observation is requested, the policy therefore reports:

```text
NULL_CENTERED
NO_VARIANCE_REWARD
CORRELATED_ESTIMATES
```

A separate Vahti-shaped metamorphic receipt generated true-zero A and B effects and varied only observation noise. The mean raw smallest Gram eigenvalue rose:

```text
noise sd 0.5    9.892
noise sd 1.0   39.814
ratio           4.02x
```

The statistic therefore rewards estimator variance while the true information remains exactly zero.

## Canonical cases

Canonical workflow run: `32026531901`

Artifact: `9287274158`

An earlier run `32026447788` failed before the scientific test because the exact-alias current receiver was one-dimensional while its alternate receiver was two-dimensional and the covariance had been stored on the case. The rerun changed only plumbing: the one-dimensional sum readout was embedded as `[x1+x2, 0]`, preserving the exact same information, policy, thresholds, expected actions and route.

All eight canonical actions were correct:

```text
case                     expected   observed   current ceiling
exact_alias_refuse       REFUSE     REFUSE       0.500
exact_alias_route        ROUTE      ROUTE        0.500
low_ceiling_route        ROUTE      ROUTE        0.597
low_ceiling_refuse       REFUSE     REFUSE       0.597
nuisance_route           ROUTE      ROUTE        0.500
wrong_metric_refuse      REFUSE     REFUSE       1.000
repeat_helpful           REPEAT     REPEAT       0.999
claim_now                CLAIM      CLAIM        1.000
```

The important rows are not just the exact aliases.

### Exact observation alias

`exact_alias_refuse` has

```text
||C(a-b)|| = 0
```

and no declared alternate readout. The policy refuses with **zero new observation calls**.

`exact_alias_route` has the same current alias, but a distributed receiver restores the distinction. The policy returns `ROUTE` with zero new observation calls.

### Nonzero but budget-limited readout

`low_ceiling_route` is not exactly aliased:

```text
raw contrast norm       0.07071
whitened residual       0.07071
48-repeat ceiling       0.597
```

More of the same measurement cannot reach the required 0.90 recovery target. An alternate receiver can, so the policy routes rather than waits.

The matched `low_ceiling_refuse` case has no alternate receiver and is refused before spending the 48-repeat budget.

### Nuisance-confounded readout

`nuisance_route` has a large raw contrast:

```text
raw contrast norm       1.414
```

but after whitening and projection against the declared nuisance direction:

```text
residual norm           ~7.45e-16
```

The current measurement therefore cannot support unique attribution even though the raw effect is large. A nuisance-breaking receiver restores a residual direction and the policy returns `ROUTE`.

This is the direct Gate-5 analogue of TWC's warning that a large physical response does not imply an identifiable physical cause once supported nuisance is included.

## Randomized recoverability

The canonical action table only checks the state-machine branches. The empirical CLAIM/REPEAT behavior was separately tested on randomized noisy episodes with balanced true A/B causes and unknown nuisance coefficients where applicable.

### REPEAT-helpful case

800 episodes:

```text
claim accuracy      0.987
claim coverage      0.999
mean calls          8.45 / 32 allowed
```

So the policy does not confuse "not enough evidence yet" with "unrecoverable." It repeats when replication can genuinely raise the evidence above the threshold, and stops well before the maximum budget on average.

### CLAIM-now case

800 episodes:

```text
claim accuracy           0.999
immediate claim fraction 0.989
mean calls               4.04 / 16 allowed
```

A strong already-supported distinction is not forced through unnecessary repetitions.

### After ROUTE

The three route-required cases were then evaluated using the recommended alternate receiver:

```text
case                  ceiling before -> after    claim accuracy   coverage
exact_alias_route       0.500 -> 0.998              0.987          0.997
low_ceiling_route       0.597 -> 1.000              0.985          1.000
nuisance_route          0.500 -> 0.958              0.988          0.858
```

So `ROUTE` is not merely a refusal label in this controlled model. It points to a changed observation map under which the distinction becomes empirically recoverable.

## Cost boundary

For the six cases whose correct initial action is `REFUSE` or `ROUTE`, the admission policy requests:

```text
0 new observation calls
```

A counterfactual policy that ignores structure/statistic obligations and simply exhausts each current readout's repeat allowance would spend:

```text
232 calls
```

before reaching the same boundary or making an unsupported attribution.

This is not a universal speedup claim. It is a capability-ordering result: when impossibility or instrument invalidity is already implied by the declared structure, spending observations first is dominated.

## Gate criteria

Every preregistered criterion passed:

```text
canonical_actions_all_correct                         true
pre_spend_refuse_or_route_costs_zero                  true
wrong_metric_generates_vahti_style_obligations        true
raw_gram_null_rewards_variance                        true
repeat_helpful_claim_accuracy_ge_0_95                 true
repeat_helpful_coverage_ge_0_90                       true
repeat_helpful_mean_calls_lt_half_max                 true
claim_now_immediate_ge_0_90                           true
claim_now_false_claim_rate_le_0_05                    true
all_routes_raise_ceiling_by_0_10                      true
all_routes_reach_target_ceiling                       true
post_route_claim_accuracy_ge_0_95                     true
wait_only_would_spend_positive_calls_before_precheck  true

PASS                                                   true
```

## What Gate 5 supports

The supported statement is narrow:

> **In a controlled declared linear-Gaussian observation model, a Kynnys-style evidence-admission policy can correctly separate structural refusal, readout routing, useful repetition, and calibrated claiming; it can refuse or route structurally unsupported claims before spending claim-specific observation calls.**

A more useful programming sentence is:

> **A demand creates obligations. It does not authorize an action.**

For computation, the action may be execution.

For evidence, the action may be a causal claim.

## What Gate 5 does not support

Do not claim that:

- generic causal identifiability can always be derived from a program graph;
- real AI systems expose a correct linear observation map `C` or nuisance matrix `N`;
- covariance is known for free in deployed systems;
- the Gaussian ceiling formula applies to arbitrary structured LLM outputs;
- ROUTE candidates will normally be declared in advance;
- TWC's topology gauge audit directly generalizes to every computation graph;
- Vahti has already been generalized to multivariate evidence statistics;
- Gate 5 is a real-world product validation.

The difficult part in a real system is declaring or learning enough trustworthy structure for the pre-spend audit without smuggling the answer into the model.

## Feature verdict

### `audit_effects`

Still supported as a refusal/reporting primitive, but its current Euclidean implementation is only a small subset of the stronger pipeline exercised here. A future version should distinguish raw sensitivity, whitened nuisance-residual sensitivity, structural exact aliasing, and empirical recoverability.

### `plan_probes(...)`

Still **not earned** as a generic public API. Gate 4 remains the kill gate for a planner that always tries to buy more observations.

### Evidence admission

**Earned as the next experimental Kynnys primitive, not yet as a fully generic public API.**

The useful state machine is:

```text
claim demand
    -> structural capability
    -> metric obligations
    -> whitened nuisance residual
    -> repeat ceiling
    -> empirical calibration
    -> CLAIM | REPEAT | ROUTE | REFUSE
```

The most important property is that `REFUSE` and `ROUTE` can occur before new observation spending.

## Code and receipts

```text
experiments/gate5_evidence_admission.py
experiments/gate5_evidence_admission_runner.py
.github/workflows/gate5-evidence-admission.yml
```

Workflow history:

```text
32026447788   FAIL — receiver/covariance shape plumbing before scientific test
32026531901   PASS — canonical evidence-admission gate
```

Artifact:

```text
9287274158
```

## Next attacker

The obvious attacker is to remove the strongest assumption in Gate 5: that the admission runtime is handed a correct observation map and nuisance model.

A serious Gate 6 should introduce **model misspecification**:

- omit a real nuisance direction;
- give a slightly wrong covariance;
- give an observation map with a hidden many-to-one collapse;
- include an apparently useful route whose declared model is wrong;
- ask whether cheap metamorphic / held-out obligations catch the misspecification before CLAIM.

That would test whether evidence admission remains conservative when its own structural model is imperfect — the place a real implementation is most likely to become confidently wrong.
