# Kynnys

**Demand a consequence. Admit only the computation that is worth crossing the threshold.**

*Kynnys* is Finnish for **threshold**.

Kynnys is currently a **Python library / programming-model experiment**, not a standalone language. The object under test is a language-level distinction that ordinary function-call syntax hides:

```text
ordinary call
    result = f(x)
    means: execute f

Kynnys
    possible = f(x)
    result = demand(possible, requirements)
    means: make an acceptable consequence available;
           decide whether execution is actually necessary
```

That difference lets the runtime explicitly choose among cached consequence reuse, cheap validation, expensive execution, waiting for equivalent work already in flight, or refusing work that violates the demand contract.

It does **not** make Python more computationally powerful. Anything here can ultimately be implemented in ordinary code. The research question is whether making these semantics first-class produces programs that are easier to reason about, audit, and eventually compile efficiently.

## The primitive

```python
from kynnys import Runtime, gate, risk

rt = Runtime()

@gate(compute_cost=20, hazard_rate=0.01)
def interpret(x):
    print("EXPENSIVE WORK")
    return expensive_model(x)

possible = interpret(input_data)   # body has NOT run

out = rt.demand(
    possible,
    risk(error_cost=5),
    now=100,
)

print(out.action, out.value)
```

A decorated gate call constructs a `GateCall`. Only `demand(...)` gives the runtime permission to execute the body.

The first reference interpreter supports:

- `REUSE` — the cached consequence is adequate under the demand contract;
- `PROBE_REUSE` — a cheaper observation confirms that the cached consequence is still valid;
- `PROBE_RUN` — probing rejects the cache, so the gate runs;
- `RUN` — execution is the cheapest admissible way to satisfy the demand;
- `WAIT` — equivalent work is already in flight, so Kynnys joins it instead of duplicating it;
- `HOLD` — the demand's explicit spend bound will not admit the required work.

## Why `PROBE` is not a heuristic

For a cached result with probability `p` of being invalid:

```text
REUSE   = p * C_wrong
RUN     = C_run
PROBE   = C_probe + p * C_run
```

Kynnys probes only when its expected cost beats **both** reuse and execution.

Example:

```text
C_run   = 12
C_probe = 5
C_wrong = 60
```

Then PROBE exists only in the band:

```text
0.104 < P(invalid) < 0.583
```

Below it, reuse is cheaper. Above it, just run. This rule came from the Fusion1 `make` reconciliation rather than from tuning toward a win.

## Exact and risk-aware execution

Kynnys deliberately supports two very different contracts.

```python
from kynnys import exact, risk

rt.demand(call, exact())          # uncertain reuse is forbidden
rt.demand(call, risk(3.0))        # stale/wrong reuse has explicit cost 3
```

`exact()` is the correctness anchor. If a result has hidden-world hazard, Kynnys must validate it or execute again. If the declared inputs fully determine the result (`hazard_rate=0`), exact mode may reuse the exact same call indefinitely.

Risk-aware mode makes approximation visible in source instead of hiding it inside an undocumented cache policy.

## Persistent private state

A gate can keep rich local state while exposing a narrow consequence:

```python
from kynnys import gate

@gate(pass_context=True, max_egress_bytes=256)
def track(ctx, frame_signature):
    tracker = ctx.local("tracker", Tracker)
    tracker.update(frame_signature)
    return tracker.small_summary()
```

`ctx.local(...)` persists across executions of that gate in the same runtime. It does not have to be serialized into every public result.

The reference interpreter also performs two dynamic boundary checks:

1. values wrapped with `ctx.private(...)` may not appear in a returned result;
2. `max_egress_bytes=N` rejects a result whose serialized representation exceeds the declared boundary.

These are **runtime checks**, not a static security/capability guarantee. Python cannot prevent a deliberately malicious gate from leaking an object through a global variable. A future Kynnys IR/compiler would be the place to make `private` and carrier width statically enforceable.

## In-flight work is part of the state

If two threads demand the same expensive gate call while one execution is already running, the second demand receives `WAIT` and joins the same work rather than launching a duplicate.

This is an important semantic distinction:

```text
not computed
!=
already computing
```

A future asynchronous Kynnys can make pending values and deadline slack richer; v0 only establishes equivalent-work de-duplication without inventing a fake WAIT benchmark.

## The TWC transplant: refuse false blame

Kynnys includes a tiny `audit_effects(...)` instrument inspired by the epistemic lesson from [TransientWaveCompiler](https://github.com/anttiluode/TransientWaveCompiler): **a large effect is not necessarily uniquely attributable**.

Suppose paired exact-vs-risk runs estimate output-effect vectors for several gates:

```python
from kynnys import audit_effects

report = audit_effects({
    "detect":   [0.8, 0.1, 0.0],
    "describe": [0.4, 0.05, 0.0],
    "verify":   [0.0, 0.0, 1.0],
})
```

For each effect `g_i`, the audit reports:

```text
sensitivity = ||g_i||
eta         = ||(I - P_Jminus_i) g_i|| / ||g_i||
```

So:

```text
large sensitivity + large eta   -> important and independently attributable
large sensitivity + tiny eta    -> important but confounded
small sensitivity + large eta   -> unique but small
small sensitivity + tiny eta    -> probably unimportant and non-identifiable
```

The dangerous quadrant is **large impact, low identifiability**. Kynnys does not interpret that as permission to loosen the gate. It means the current experiment cannot tell which gate deserves blame; force paired interventions or budget the aliased group jointly.

Nuisance directions can also be supplied so GPU warmup, network jitter, or another measured artifact is not silently attributed to a gate policy.

## Relationship to Fusion1

[Fusion1](https://github.com/anttiluode/Fusion1) is the control-plane laboratory that motivated the admission arithmetic:

```text
validity / dependency / pending work / probe cost / compute cost
                              |
                              v
                  REUSE | PROBE | WAIT | WAKE | HOLD
```

Kynnys moves one layer up:

```text
Fusion1 question:
    What should this persistent computational node do now?

Kynnys question:
    What should a program mean when the caller requires a consequence
    but execution is only one possible way to satisfy that requirement?
```

Kynnys does not currently depend on Fusion1. The small reference runtime duplicates only enough admission logic to make the programming contract executable and testable. If the semantics survive, Fusion1 can become one backend rather than a hard dependency.

## Relationship to DifferentMachine / Y

[DifferentMachine](https://github.com/anttiluode/DifferentMachine) asks whether the represented machine can be much larger than the causal frontier executed for one event.

Kynnys expresses a systems-level analogue:

> **The amount of computation represented by a program need not equal the amount of computation admitted for a demand.**

The stronger future `Y` idea—large private local degrees of freedom with narrow receiver-specific consequences—belongs later at the compiler/IR boundary. v0 keeps only the parts Python can test honestly: persistent private local state and bounded observable egress.

## Is this a language?

**Not yet.** Deliberately.

The ladder is:

```text
Python library / executable semantics       <- now
        |
        v
several useful real programs
        |
        v
stable constructs Python cannot verify cleanly
        |
        v
Kynnys IR / MLIR dialect
        |
        v
compiler-enforced private/egress/resource contracts
        |
        v
specialized hardware instructions only if measurements justify them
```

A parser would be premature. If `gate`, `demand`, private state, freshness/error contracts and auditability repeatedly matter in real programs, the syntax will have earned a language.

## What might it be useful for?

The strongest early targets are workloads where the *world continues existing while the program is not looking*:

- AI agents deciding whether to reread files, rerun tests, re-query CI, call another model, or reuse an interpretation;
- always-on perception where cheap change evidence can guard expensive vision/audio inference;
- remote services and sensors where checking truth has a nonzero cost and local content hashing is impossible;
- interactive build/publish/data pipelines with uncertain external dependencies and in-flight work;
- distributed model/tool routing where computation has latency or monetary cost;
- later, accelerators where private state and narrow egress can become compiler-visible communication constraints.

The wrong target is ordinary deterministic local code with cheap exact dependency versions. There, `make`, memoization, and ordinary incremental computation are already formidable baselines and Kynnys should tie rather than manufacture an advantage.

## Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/hello_kynnys.py
```

No runtime dependency beyond Python 3.10+ is required.

## First kill gates

Kynnys is useful only if these survive real programs:

1. **Non-execution matters:** `demand` avoids meaningful wall-clock/tool/API work relative to explicit always-run and exact-invalidation baselines at matched correctness.
2. **Uncertainty matters:** benefits remain on sources that cannot simply be hashed/versioned locally.
3. **The contract helps:** users can see and control when approximation is allowed instead of merely getting a clever opaque cache.
4. **Audit changes decisions:** exact-vs-risk interventions reveal confounded gate effects often enough that refusing false attribution is useful.
5. **Private/egress semantics matter:** real programs benefit from making local persistent state and narrow outward consequences explicit.

If those fail, Kynnys should remain a small library or collapse back into Fusion1 rather than becoming a language for its own sake.

## Status

**v0.1 research prototype. No novelty or production-performance claim.**

The central sentence is intentionally modest:

> **Ordinary function syntax asks the computer to execute. Kynnys lets a program demand a consequence and makes execution an admission decision.**

See [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).
