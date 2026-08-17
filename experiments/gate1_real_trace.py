from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, Any
from urllib.request import Request, urlopen


@dataclass
class Sample:
    state: str
    refresh_ms: float
    probe_state: str | None
    probe_ms: float | None


@dataclass
class SourceTrace:
    name: str
    samples: list[Sample]
    has_exact_probe: bool

    @property
    def changes(self) -> int:
        return sum(
            self.samples[i].state != self.samples[i - 1].state
            for i in range(1, len(self.samples))
        )

    @property
    def median_refresh_ms(self) -> float:
        return statistics.median(s.refresh_ms for s in self.samples)

    @property
    def median_probe_ms(self) -> float | None:
        xs = [s.probe_ms for s in self.samples if s.probe_ms is not None]
        return statistics.median(xs) if xs else None


@dataclass
class PolicyResult:
    policy: str
    source: str
    error_ratio: float | None
    cost_ms: float
    errors: int
    refreshes: int
    probes: int
    reuses: int


class BetaHazard:
    """Online per-demand change probability with an uncertainty-aware upper estimate.

    Beta(1,1) is deliberately broad. The policy uses mean + z*sd rather than
    the posterior mean so a source with little evidence is treated as risky.
    This model is per demand interval, not a continuous-time hazard model.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, z: float = 1.645) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.z = float(z)
        self.observations = 0

    def observe(self, changed: bool) -> None:
        if changed:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.observations += 1

    def upper(self) -> float:
        a, b = self.alpha, self.beta
        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        return min(1.0, max(0.0, mean + self.z * math.sqrt(var)))


def _http_json(url: str, token: str | None = None) -> tuple[Any, float, int]:
    headers = {
        "User-Agent": "Kynnys-Gate1/0.1",
        "Accept": "application/vnd.github+json, application/json",
    }
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    req = Request(url, headers=headers)
    t0 = time.perf_counter()
    with urlopen(req, timeout=15) as response:
        body = response.read()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return json.loads(body), elapsed_ms, len(body)


def _cpu_transform(seed: str, rounds: int = 45000) -> str:
    import hashlib

    x = seed.encode("utf-8")
    for _ in range(rounds):
        x = hashlib.blake2b(x, digest_size=32).digest()
    return x.hex()


@dataclass
class LiveSource:
    name: str
    refresh: Callable[[], tuple[str, float]]
    probe: Callable[[], tuple[str, float]] | None = None


def live_sources() -> list[LiveSource]:
    token = os.environ.get("GITHUB_TOKEN")

    def gh_head(owner_repo: str, *, slow: bool = False) -> LiveSource:
        owner, repo = owner_repo.split("/", 1)
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/main"
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/main"

        def probe() -> tuple[str, float]:
            data, ms, _ = _http_json(ref_url, token)
            return str(data["object"]["sha"]), ms

        def refresh() -> tuple[str, float]:
            data, ms, _ = _http_json(commit_url, token)
            sha = str(data["sha"])
            if slow:
                t0 = time.perf_counter()
                _cpu_transform(sha)
                ms += (time.perf_counter() - t0) * 1000.0
            return sha, ms

        suffix = "_slow_compute" if slow else "_head"
        return LiveSource(owner_repo.replace("/", "_") + suffix, refresh, probe)

    def github_global_event() -> LiveSource:
        url = "https://api.github.com/events?per_page=1"

        def refresh() -> tuple[str, float]:
            data, ms, _ = _http_json(url, token)
            return str(data[0]["id"]), ms

        return LiveSource("github_global_event", refresh, None)

    def hn_top_story() -> LiveSource:
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"

        def refresh() -> tuple[str, float]:
            data, ms, _ = _http_json(url, None)
            return str(data[0]), ms

        return LiveSource("hn_top_story", refresh, None)

    def pypi_numpy() -> LiveSource:
        url = "https://pypi.org/pypi/numpy/json"

        def refresh() -> tuple[str, float]:
            data, ms, _ = _http_json(url, None)
            return str(data["info"]["version"]), ms

        return LiveSource("pypi_numpy_version", refresh, None)

    return [
        gh_head("anttiluode/Kynnys"),
        gh_head("anttiluode/Kynnys", slow=True),
        gh_head("python/cpython"),
        github_global_event(),
        hn_top_story(),
        pypi_numpy(),
    ]


def collect_live(rounds: int, sleep_seconds: float) -> list[SourceTrace]:
    sources = live_sources()
    traces = [SourceTrace(s.name, [], s.probe is not None) for s in sources]
    for r in range(rounds):
        print(f"collect round {r + 1}/{rounds}", flush=True)
        for source, trace in zip(sources, traces):
            state, refresh_ms = source.refresh()
            probe_state: str | None = None
            probe_ms: float | None = None
            if source.probe is not None:
                probe_state, probe_ms = source.probe()
                if probe_state != state:
                    # The source changed between refresh and probe. Use probe as
                    # the round's terminal truth so exact-probe replay is coherent.
                    state = probe_state
            trace.samples.append(Sample(state, refresh_ms, probe_state, probe_ms))
            print(
                f"  {source.name:32s} state={state[:14]:14s} "
                f"refresh={refresh_ms:7.1f}ms "
                f"probe={'-' if probe_ms is None else f'{probe_ms:.1f}ms'}",
                flush=True,
            )
        if r + 1 < rounds:
            time.sleep(sleep_seconds)
    return traces


def fixture_traces() -> list[SourceTrace]:
    # Three deliberately different traces for offline logic validation.
    stable = [Sample("A", 100, "A", 5) for _ in range(10)]
    mid_states = ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C"]
    mid = [Sample(s, 120, s, 8) for s in mid_states]
    fast_states = [str(i) for i in range(10)]
    fast = [Sample(s, 30, None, None) for s in fast_states]
    return [
        SourceTrace("fixture_stable", stable, True),
        SourceTrace("fixture_mid", mid, True),
        SourceTrace("fixture_fast", fast, False),
    ]


def replay_always(trace: SourceTrace) -> PolicyResult:
    return PolicyResult(
        "always", trace.name, None,
        sum(s.refresh_ms for s in trace.samples), 0,
        len(trace.samples), 0, 0,
    )


def replay_exact(trace: SourceTrace, policy: str) -> PolicyResult:
    cost = 0.0
    errors = 0
    refreshes = probes = reuses = 0
    cache: str | None = None
    for sample in trace.samples:
        if cache is None:
            cost += sample.refresh_ms
            refreshes += 1
            cache = sample.state
            continue
        if trace.has_exact_probe:
            assert sample.probe_ms is not None and sample.probe_state is not None
            cost += sample.probe_ms
            probes += 1
            if sample.probe_state != cache:
                cost += sample.refresh_ms
                refreshes += 1
                cache = sample.state
            else:
                reuses += 1
        else:
            cost += sample.refresh_ms
            refreshes += 1
            cache = sample.state
        errors += int(cache != sample.state)
    return PolicyResult(policy, trace.name, None, cost, errors, refreshes, probes, reuses)


def replay_adaptive(trace: SourceTrace, error_ratio: float) -> PolicyResult:
    cost = 0.0
    errors = 0
    refreshes = probes = reuses = 0
    cache: str | None = None
    hazard = BetaHazard()
    seen_refresh_costs: list[float] = []
    seen_probe_costs: list[float] = []
    age_steps = 0

    for sample in trace.samples:
        if cache is None:
            cost += sample.refresh_ms
            refreshes += 1
            seen_refresh_costs.append(sample.refresh_ms)
            cache = sample.state
            age_steps = 0
            continue

        age_steps += 1
        c_refresh = statistics.mean(seen_refresh_costs) if seen_refresh_costs else sample.refresh_ms
        c_wrong = error_ratio * c_refresh
        q_upper = hazard.upper()
        p_bad = 1.0 - (1.0 - q_upper) ** age_steps

        # Cold start is deliberately conservative: gather four actual source
        # observations before uncertain REUSE can win. This is real probe or
        # refresh work, not oracle knowledge.
        if hazard.observations < 4:
            action = "probe" if trace.has_exact_probe else "refresh"
        else:
            reuse_cost = p_bad * c_wrong
            refresh_cost = c_refresh
            probe_cost = math.inf
            if trace.has_exact_probe and seen_probe_costs:
                c_probe = statistics.mean(seen_probe_costs)
                probe_cost = c_probe + p_bad * c_refresh

            action = min(
                [(reuse_cost, "reuse"), (probe_cost, "probe"), (refresh_cost, "refresh")],
                key=lambda pair: (pair[0], {"reuse": 0, "probe": 1, "refresh": 2}[pair[1]]),
            )[1]

        if action == "reuse":
            reuses += 1
        elif action == "probe":
            assert sample.probe_ms is not None and sample.probe_state is not None
            cost += sample.probe_ms
            probes += 1
            seen_probe_costs.append(sample.probe_ms)
            changed = sample.probe_state != cache
            hazard.observe(changed)
            if changed:
                cost += sample.refresh_ms
                refreshes += 1
                seen_refresh_costs.append(sample.refresh_ms)
                cache = sample.state
            else:
                reuses += 1
            age_steps = 0
        else:
            old = cache
            cost += sample.refresh_ms
            refreshes += 1
            seen_refresh_costs.append(sample.refresh_ms)
            cache = sample.state
            hazard.observe(cache != old)
            age_steps = 0

        errors += int(cache != sample.state)

    return PolicyResult(
        "kynnys_adaptive", trace.name, error_ratio,
        cost, errors, refreshes, probes, reuses,
    )


def run_singleflight_receipt() -> dict[str, float | int]:
    # Use the actual Kynnys runtime and a real subprocess. Wall time is not the
    # main claim: the receipt is one launched subprocess instead of N.
    from kynnys import Runtime, gate, exact

    launches = 0
    launch_lock = threading.Lock()

    @gate(compute_cost=1.0, name="gate1_slow_subprocess")
    def slow_command(key: str) -> str:
        nonlocal launches
        with launch_lock:
            launches += 1
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(0.35)"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return key.upper()

    rt = Runtime()
    barrier = threading.Barrier(8)
    actions: list[str] = []

    def worker() -> None:
        barrier.wait()
        out = rt.demand(slow_command("same"), exact())
        actions.append(out.action.value)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "demanders": 8,
        "subprocess_launches": launches,
        "wait_actions": actions.count("WAIT"),
        "run_actions": actions.count("RUN"),
        "wall_ms": elapsed_ms,
    }


def summarize(
    traces: list[SourceTrace], ratios: list[float], *, singleflight: bool = True
) -> dict[str, Any]:
    results: list[PolicyResult] = []
    for trace in traces:
        results.append(replay_always(trace))
        exact_base = replay_exact(trace, "remote_exact")
        k_exact = replay_exact(trace, "kynnys_exact")
        # Kynnys exact should be merely a programming-model expression of the
        # boring exact baseline, never a magic advantage.
        assert abs(exact_base.cost_ms - k_exact.cost_ms) < 1e-9
        assert exact_base.errors == k_exact.errors == 0
        results.extend([exact_base, k_exact])
        for ratio in ratios:
            results.append(replay_adaptive(trace, ratio))

    print("\nREAL TRACE SUMMARY")
    print("source                           changes  refresh_med  probe_med")
    for t in traces:
        pm = "-" if t.median_probe_ms is None else f"{t.median_probe_ms:.1f}"
        print(f"{t.name:32s} {t.changes:7d} {t.median_refresh_ms:11.1f} {pm:>10s}")

    print("\nPOLICY REPLAY (lower cost; errors must be read beside it)")
    print("source                           policy             ratio   cost_ms errors refresh probe reuse")
    for r in results:
        ratio = "-" if r.error_ratio is None else f"{r.error_ratio:g}"
        print(
            f"{r.source:32s} {r.policy:18s} {ratio:>6s} "
            f"{r.cost_ms:9.1f} {r.errors:6d} {r.refreshes:7d} {r.probes:5d} {r.reuses:5d}"
        )

    return {
        "traces": [
            {
                "name": t.name,
                "changes": t.changes,
                "has_exact_probe": t.has_exact_probe,
                "median_refresh_ms": t.median_refresh_ms,
                "median_probe_ms": t.median_probe_ms,
                "samples": [asdict(s) for s in t.samples],
            }
            for t in traces
        ],
        "results": [asdict(r) for r in results],
        "singleflight": run_singleflight_receipt() if singleflight else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--output", default="gate1_result.json")
    args = parser.parse_args()

    traces = fixture_traces() if args.fixture else collect_live(args.rounds, args.sleep)
    report = summarize(
        traces, ratios=[0.5, 2.0, 10.0, 100.0], singleflight=not args.fixture
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["singleflight"] is not None:
        print("\nSINGLEFLIGHT", json.dumps(report["singleflight"], sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
