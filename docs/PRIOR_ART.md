# Prior-art and claim boundary

Kynnys is **not** a claim that lazy evaluation, memoization, build invalidation, incremental computation, futures, caching, approximate computing, or resource-aware scheduling are new.

The point of this document is to make the attackers explicit before the project accumulates mythology.

## Self-adjusting / incremental computation

Self-adjusting computation already records dependencies and propagates changes through previously executed computation. Work by Acar, Blelloch, Harper and collaborators established dynamic dependence / change-propagation approaches and libraries for self-adjusting programs.

Adapton later made incremental computation demand-driven: changed inputs do not imply that every affected output needs immediate recomputation if no observer currently demands it.

**Attack on Kynnys:** if exact dependency/change information is cheap, ordinary incremental computation is the baseline. Kynnys must not claim an advantage from merely remembering dependencies or deferring unused work.

Kynnys's narrower experiment is the combination of demand with *epistemic* state:

```text
input may have changed, but checking is itself costly
cached semantic consequence may remain adequate
an exact/cheap probe may exist
work may already be in flight
wrong reuse has an explicit consequence cost
```

Whether that combination deserves a programming model is empirical.

References:

- Umut A. Acar, Guy E. Blelloch, Robert Harper, *Adaptive Functional Programming* / subsequent self-adjusting computation work: https://www.cs.cmu.edu/~rwh/papers.html
- Matthew A. Hammer, Khoo Yit Phang, Michael Hicks, Jeffrey S. Foster, *Adapton: Composable, Demand-Driven Incremental Computation* (PLDI 2014 artifact/paper): https://www.cs.umd.edu/projects/PL/adapton/

## Build systems / `make`

Exact dependency invalidation is an especially strong attacker. Fusion1's Gate 0 correctly forced this comparison: if local sources can be cheaply versioned/hashed, a probabilistic validity layer should not manufacture a win.

Kynnys therefore targets cases where truth is not a free local hash: remote CI, external services, sensors, expensive model-derived semantics, or other opaque state.

## Lazy evaluation / futures / async

Lazy evaluation already separates expression construction from evaluation in important ways. Futures/promises already represent unfinished work, and task runtimes already de-duplicate or coordinate work in many systems.

**Kynnys claim boundary:** `demand` is not interesting merely because evaluation is delayed. The proposed contract combines demand with validity, cost-of-being-wrong, value-of-information probes, explicit egress/private state, and an exact comparison mode.

If real examples reduce to `functools.cache` + `asyncio`, use those instead.

## Approximate programming

Approximate-programming research already makes accuracy/resource tradeoffs explicit. EnerJ, for example, used approximate data types and static separation between precise and approximate computation.

**Attack on Kynnys:** "allow the program to be approximate to save resources" is occupied territory.

Kynnys is testing a more temporal/runtime-specific object: the same persistent gate can be demanded exactly or opportunistically, and the runtime records whether it reused, validated, waited, or executed. Its audit layer then asks whether exact-vs-opportunistic output effects can actually be attributed to individual gates.

Reference:

- Adrian Sampson et al., *EnerJ: Approximate Data Types for Safe and General Low-Power Computation* (PLDI 2011): https://ece.uwaterloo.ca/~wdietl/publications/pubs/EnerJ11-abstract.html

## TWC / identifiability

The `audit_effects` idea is explicitly derived from the epistemic lesson developed in Antti Luode's TransientWaveCompiler:

> optimization cannot create information that the measurement does not contain.

Kynnys applies that lesson to program approximation: if two gate perturbations create collinear output effects, the profiler should report an attribution-equivalence/confounding problem rather than confidently blame one gate.

Source project:

- https://github.com/anttiluode/TransientWaveCompiler

## MLIR / future compiler path

MLIR is intentionally not part of v0. Its dialect model is a plausible future substrate if Kynnys earns compile-time constructs such as:

```text
gate.region
gate.private
gate.carrier
gate.demand
gate.probe
gate.pending
```

MLIR already supports domain-specific operations/types/attributes and progressive lowering, which is precisely why it is a better future experiment than designing an ISA first.

References:

- MLIR dialect documentation: https://mlir.llvm.org/docs/DefiningDialects/
- Chris Lattner et al., *MLIR: A Compiler Infrastructure for the End of Moore's Law*: https://arxiv.org/abs/2002.11054

## Novelty discipline

The repo should use formulations like:

> "Kynnys tests whether ..."

not:

> "Kynnys invents the first ..."

until a real literature review and external benchmarks justify something stronger.

The present scientific claim is simply that the combination is executable and has clear kill gates.
