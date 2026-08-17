# Kynnys v0.1 design

## 1. Research object

The primitive is not `MEXPAND`, a manifold register, an oscillator, or a scheduler algorithm.

It is the semantic split:

```text
call construction     -> represent possible computation

demand                -> require an acceptable consequence

admission              -> decide whether execution is necessary
```

That gives existing mechanisms—memoization, validity, probing, in-flight work, budgets, private local state—a common socket.

## 2. Why a library first

A standalone parser adds surface area without evidence that new syntax is needed. Python is sufficient to test the semantics:

```python
@gate(...)
def f(x):
    ...

possible = f(x)       # lazy GateCall
out = demand(possible)
```

The transition to a language/IR is earned only when important guarantees cannot be expressed honestly as runtime checks.

## 3. v0 execution semantics

Each distinct gate call is keyed by gate name + serialized arguments (or a user-supplied `key_fn`). The cache therefore represents exact declared-input identity.

A gate may additionally declare hidden-world uncertainty with continuous-time `hazard_rate = lambda`:

```text
P(invalid after age t) = 1 - exp(-lambda t)
```

This is intentionally named a *rate*. It is not the discrete Bernoulli probability used in Fusion1's synthetic Gate 0; callers with a discrete per-step probability should convert parameterizations explicitly rather than mixing formulas.

For a cached call:

```text
expected REUSE loss = p_invalid * error_cost
expected RUN cost   = compute_cost
expected PROBE cost = probe_cost + p_invalid * compute_cost
```

The reference runtime chooses the lowest expected cost. A probe is assumed to answer the cache-validity question exactly. Imperfect/probabilistic probes require a richer observation model and are not smuggled into v0.

## 4. Exactness anchor

`exact()` sets `error_cost = infinity`.

Consequences:

- `hazard_rate == 0`: same declared call may reuse indefinitely;
- hidden uncertainty + exact probe: validate or run;
- hidden uncertainty without probe: run;
- missing cache: run;
- insufficient explicit `max_spend`: `HOLD` rather than silently violating the contract.

This provides an executable from-scratch/exact comparison mode for future benchmarks.

## 5. Persistent private state

`pass_context=True` gives a gate a `GateContext`:

```python
state = ctx.local("name", factory)
```

Local state is scoped to `(runtime, gate name, slot name)` and persists across calls to the gate. This makes the represented local machine larger than its returned consequence without serializing the state through each edge.

Current limitations are explicit:

- it is process-local;
- no snapshot/recovery protocol exists;
- mutation is ordinary Python mutation;
- Python cannot prove a gate did not deliberately leak a private object elsewhere.

## 6. Egress contract

`max_egress_bytes` measures the pickled return representation and rejects oversize output.

`ctx.private(value)` marks a value that is forbidden in the returned object graph.

These checks make the intended boundary executable, but they are not the final claim. A compiler/IR could eventually enforce:

```text
no pointer/reference escape
no serialization escape
bounded carrier type
private allocation placement
```

That is the point at which Kynnys would become materially more language-like.

## 7. In-flight state

Equivalent concurrent demands share a single execution. The second demander waits on the first and receives action `WAIT`.

This deliberately tests only the uncontested case:

```text
same call + same runtime + execution already active
    -> do not duplicate it
```

Kynnys v0 does not yet pretend to optimize deadline slack, cancellation, supersession, or speculative execution.

## 8. Audit geometry

For empirical gate-effect vector `g_i`, define:

```text
S_i   = ||g_i||
eta_i = ||(I - P_Jminus_i) g_i|| / ||g_i||
```

`Jminus_i` contains every other gate effect plus declared nuisance directions.

Interpretation:

```text
S high, eta high  -> important and distinguishable
S high, eta low   -> important but confounded
S low, eta high   -> unique but small
S low, eta low    -> weak and confounded
```

Kynnys intentionally avoids the invalid inference:

```text
eta ~ 0  therefore gate does not matter
```

Low eta means the observation does not support unique attribution.

## 9. Fusion1 boundary

Fusion1 remains useful as an admission-policy research repo and realistic workflow benchmark. Kynnys should not fork all of Fusion1 indefinitely.

A future backend boundary can look like:

```text
Kynnys source semantics
        |
        v
AdmissionBackend
        |
        +-- reference runtime
        +-- Fusion1
        +-- distributed runtime
        `-- compiler/static backend
```

This should be built only after at least one real Kynnys program requires it.

## 10. Language / IR threshold

Do not build a standalone language because the name is good.

Build it when one or more of these become repeated blockers:

1. static non-escape of private state;
2. carrier/egress types that must be compile-time guaranteed;
3. effect tracking for exact vs risk-aware computation;
4. resource/budget contracts that optimizers must preserve;
5. explicit pending values and cancellation semantics;
6. lowering gate-private regions to GPU scratchpad/local SRAM/remote workers;
7. whole-program exact-vs-opportunistic equivalence checks.

MLIR is a plausible substrate at that point because Kynnys constructs map naturally to operations, regions, types, attributes and lowering passes. It is deliberately not a dependency today.
