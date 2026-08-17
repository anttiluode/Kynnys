# Gate 1 — real opaque-source trace

Date: 2026-08-17

Status: **run twice on GitHub Actions; second run is the canonical receipt.**

This gate replaces the synthetic cost world with real network and command latency while preserving a fair comparison: one external-state trace is collected first, then every policy is replayed on the exact same trace. Oracle truth is used only for scoring and is not charged to any policy.

The benchmark is in `experiments/gate1_real_trace.py`; the workflow is `.github/workflows/gate1-real.yml`.

## What was tested

The live sources were deliberately heterogeneous:

- `anttiluode/Kynnys` main-head state with an exact remote Git ref probe;
- the same Kynnys head guarding a deliberately heavier local downstream transform;
- `python/cpython` main-head state with an exact remote Git ref probe;
- GitHub's latest public event id, which changes very quickly and has no cheaper exact probe in this instrument;
- Hacker News `maxitem`, which supplied the desired intermediate real change rate;
- Hacker News top story, stable during this trace and without a cheaper exact probe;
- the current NumPy release on PyPI, stable during this trace and without a cheaper exact probe.

Each run also contains a real single-flight receipt: eight concurrent demands target one gate whose body launches a slow subprocess.

Policies:

```text
always
    refresh on every demand

remote_exact
    exact remote version/probe when available, refresh on change;
    otherwise refresh every time

kynnys_exact
    deliberately asserted to tie remote_exact exactly

kynnys_adaptive
    online Beta-Bernoulli change model with a conservative upper estimate;
    four real observations before uncertain reuse is permitted;
    compare expected REUSE / PROBE / REFRESH cost;
    sweep C_wrong / C_refresh = 0.5, 2, 10, 100
```

The adaptive model accumulates invalidation probability across skipped demands; it cannot reuse forever merely because it stopped observing.

## Calibration run 1

The first live run exposed two instrument problems rather than being hidden:

1. the GitHub ref "probe" was almost as expensive as the fuller commit refresh (roughly 305–326 ms versus 321–339 ms), so it was not a useful value-of-information instrument;
2. the intended slow-compute source added too little work, and the natural sources happened to split into 0/15 changes or 15/15 changes with no middle regime.

The benchmark was therefore strengthened, not tuned toward a win:

- the downstream transform was made genuinely slower;
- Hacker News `maxitem` was added as another real source.

## Canonical run 2: observed source geometry

Sixteen samples were collected.

| source | observed changes in 15 intervals | median refresh | median exact probe |
|---|---:|---:|---:|
| Kynnys head | 0 | 281.9 ms | 248.0 ms |
| Kynnys head + slow compute | 0 | 702.7 ms | 234.9 ms |
| CPython head | 0 | 281.7 ms | 250.9 ms |
| GitHub global event | 15 | 437.5 ms | — |
| HN maxitem | 2 | 124.7 ms | — |
| HN top story | 0 | 122.4 ms | — |
| PyPI NumPy version | 0 | 78.9 ms | — |

This is already informative. A remote metadata operation should not be called a "cheap probe" merely because the payload is conceptually smaller. For ordinary GitHub head checks in this runner, the measured latency gap was small. When the downstream consequence was made genuinely expensive, the same exact ref check became economically useful.

## Selected policy results

### Kynnys head — cheap-ish refresh, stable trace

```text
always                         4386.4 ms   0 errors
remote_exact / kynnys_exact    3946.0 ms   0 errors
adaptive ratio 2               2338.7 ms   0 errors
adaptive ratio 10              4246.0 ms   0 errors
```

Exact remote checking saved only about 10% versus always-run because the probe itself was close to refresh cost.

The ratio-2 adaptive policy skipped further checks after accumulating evidence of stability and saved more on this short trace, but **zero observed changes means this is not a reliability result**. It is evidence only that explicit approximate reuse can save checks during a stable interval.

At high wrongness cost, the four forced cold-start probes become pure overhead and adaptive is worse than the exact baseline. That suggests the cold-start policy itself needs to learn probe economics sooner.

### Kynnys head + genuinely slower downstream compute

```text
always                        11295.3 ms   0 errors
remote_exact / kynnys_exact    4188.8 ms   0 errors
adaptive ratio 2               3046.3 ms   0 errors
adaptive ratio 10              4188.8 ms   0 errors
```

Here the exact probe finally has the intended shape:

```text
median probe    234.9 ms
median refresh  702.7 ms
```

Exact remote invalidation saved about **62.9%** versus always-run.

On this stable trace, adaptive ratio 2 saved about **73.0%** versus always-run and **27.3%** beyond exact checking by declining some probes after the cold start. Again, because the source did not change during the trace, that extra 27.3% is an approximation receipt, not evidence of equal long-run correctness.

At ratio 10 and 100, adaptive collapses back to the exact policy, as it should.

### Fast source — GitHub global event

```text
changes                       15 / 15
always / exact                7183.3 ms   0 errors
adaptive ratio 0.5            2269.6 ms  11 errors
adaptive ratio 2              7183.3 ms   0 errors
adaptive ratio 10             7183.3 ms   0 errors
```

This is a useful positive control for the online estimator. Once wrongness matters, the policy learns that this source changes essentially every demand and stops gambling. Cheap wrongness simply buys stale answers; the benchmark reports those errors rather than calling the lower cost a win.

### Intermediate source — Hacker News maxitem

This was the most important new source in the second run:

```text
changes                        2 / 15
always / exact                2082.8 ms   0 errors
adaptive ratio 0.5             709.1 ms   9 errors
adaptive ratio 2              1367.0 ms   2 errors
adaptive ratio 10             2082.8 ms   0 errors
adaptive ratio 100            2082.8 ms   0 errors
```

This is the hard result.

At ratio 2, the online hazard model reduces measured I/O cost by about **34.4%**, but it serves **two stale answers**. Therefore:

> **Online hazard estimation does not turn risk-aware reuse into matched-correctness caching.**

The mid-rate source is precisely where confidence can lag reality. The estimator improves the policy's adaptation, but a finite `C_wrong` still authorizes errors by design.

At ratio 10, the policy refuses that trade and becomes exact.

### Stable sources without a cheap probe

HN top story and the NumPy PyPI version did not change during this trace.

At ratio 2, adaptive reuse reduced measured I/O cost by about 44% on both with zero observed errors. That is useful as a workload receipt but not a safety claim: a longer trace containing a real change is required before characterizing the miss rate.

## Single-flight / WAIT receipt

Eight concurrent callers demanded the same Kynnys gate. The gate body launched a real subprocess that sleeps for 350 ms.

Observed:

```text
demanders            8
subprocess launches  1
RUN actions          1
WAIT actions         7
wall time            ~378 ms
```

This is the cleanest current Kynnys feature. It requires no hazard estimate and no approximation: equivalent concurrent work is coalesced into one execution.

## Verdict

### WAIT / single-flight

**Passes this gate.** It is useful ordinary request coalescing exposed through the Kynnys demand model. It is not novel scheduling theory, but it is immediately defensible functionality.

### Exact remote probing

**Conditionally useful.** The important variable is measured `C_probe / C_refresh`, not whether one operation sounds conceptually lighter.

- GitHub ref versus commit fetch: probe only slightly cheaper; modest exact saving.
- GitHub ref guarding a real expensive downstream consequence: probe strongly worthwhile; large exact saving.

So Kynnys should learn/measure probe economics and should not assume `HEAD`, metadata, or ref checks are cheap enough to matter.

### Online hazard / risk mode

**Survives as an explicit approximation mechanism, not as a safe default.**

It correctly moved toward full refresh on the every-demand source and it saved checks on stable sources. But the intermediate HN source produced stale answers at ratio 2. Hazard learning reduced ignorance; it did not remove the fundamental cost-versus-error trade.

Therefore the online estimator should remain experimental rather than being silently wired into the core runtime as an automatic policy.

### High-cost-of-wrong / irreversible domains

This gate strengthens the conservative rule: use `exact()` where stale output is unacceptable. Kynnys should not infer that payments, medical, legal, publishing, destructive filesystem changes, or other irreversible actions are safe for risk mode merely because a learned hazard is small.

The library cannot derive the true semantic cost of being wrong from runtime statistics.

### `max_egress_bytes`

Not tested here and still not a systems guarantee in Python. It remains a placeholder for the future static-language/IR argument, not a headline v0 feature.

### `audit_effects`

Not tested here. It remains a separate promising axis and deserves a deliberately confounded paired-intervention gate rather than being bundled into the admission benchmark.

## What changed in our belief

Before Gate 1 it was tempting to say:

> learn the hazard online and PROBE becomes usable.

After Gate 1 the stronger statement is:

> **The runtime can learn source volatility and operation costs, but it cannot learn the user's cost of a stale consequence from source behavior alone.**

That semantic cost is part of the demand contract.

This is good news for the language/programming-model idea and bad news for a magical autonomous cache. `demand(..., exact())` and `demand(..., risk(error_cost=...))` really do mean different programs.

## Next gates

1. **Cost-aware cold start:** after one measured probe/refresh pair, stop forcing probes whose observed cost is too close to refresh; continue learning hazard through refresh outcomes.
2. **Longer repeated traces:** especially intermediate-rate sources, to measure miss distribution rather than one short lucky/stale window.
3. **Deadline-aware single-flight:** make `WAIT` choose between joining in-flight work and launching/rejecting based on an explicit deadline, with identical execution semantics across baselines.
4. **Audit Gate 2:** construct a real two-change pipeline where two approximation effects are intentionally collinear, then test whether `audit_effects` both refuses false blame and recommends a separating paired intervention.

## Current scientific boundary

The live result does **not** show that Kynnys beats `make`, memoization, or exact invalidation at equal correctness in general.

It shows three narrower things:

1. request coalescing works exactly as expected;
2. exact probing has large value only when the measured probe-to-refresh cost ratio is favorable;
3. learned risk-aware admission can save real I/O, but the middle-hazard trace demonstrates that those savings can be purchased with stale results exactly when the demand contract permits them.

That is a better result than a synthetic "Kynnys wins" table because it tells us what the primitive actually means.
