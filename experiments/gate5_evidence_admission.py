from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


class Action(str, Enum):
    CLAIM = "CLAIM"
    REPEAT = "REPEAT"
    ROUTE = "ROUTE"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class MetricContract:
    name: str
    kind: str
    covariance_aware: bool = True
    shared_baseline: bool = False
    variance_reward_safe: bool = True
    null_centered: bool = True


@dataclass(frozen=True)
class Route:
    name: str
    observation: np.ndarray
    nuisance: np.ndarray


@dataclass(frozen=True)
class ClaimCase:
    name: str
    latent_a: np.ndarray
    latent_b: np.ndarray
    observation: np.ndarray
    nuisance: np.ndarray
    covariance: np.ndarray
    metric: MetricContract
    routes: tuple[Route, ...] = ()
    target_accuracy: float = 0.90
    claim_confidence: float = 0.95
    initial_repeats: int = 0
    max_repeats: int = 48
    expected_initial: Action = Action.REFUSE


@dataclass
class Capability:
    raw_contrast_norm: float
    whitened_residual_norm: float
    max_accuracy: float
    structurally_aliased: bool
    nuisance_confounded: bool


@dataclass
class Decision:
    action: Action
    reason: str
    obligations: list[str]
    capability: Capability
    route: str | None = None
    claim: str | None = None
    confidence: float | None = None
    requested_repeats: int = 0


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _inv_sqrt(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _orthogonal_residual(vector: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(-1)
    n = np.asarray(nuisance, dtype=float)
    if n.size == 0 or n.shape[1] == 0:
        return v.copy()
    u, singular, _vh = np.linalg.svd(n, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 0.0:
        return v.copy()
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    if rank == 0:
        return v.copy()
    basis = u[:, :rank]
    return v - basis @ (basis.T @ v)


def _capability_for(
    case: ClaimCase,
    observation: np.ndarray,
    nuisance: np.ndarray,
) -> Capability:
    contrast = np.asarray(observation, dtype=float) @ (case.latent_a - case.latent_b)
    raw_norm = float(np.linalg.norm(contrast))
    w = _inv_sqrt(case.covariance)
    whitened = w @ contrast
    nuisance_w = w @ np.asarray(nuisance, dtype=float)
    residual = _orthogonal_residual(whitened, nuisance_w)
    residual_norm = float(np.linalg.norm(residual))
    max_acc = _phi(residual_norm * math.sqrt(case.max_repeats) / 2.0)
    return Capability(
        raw_contrast_norm=raw_norm,
        whitened_residual_norm=residual_norm,
        max_accuracy=max_acc,
        structurally_aliased=raw_norm < 1e-10,
        nuisance_confounded=(raw_norm >= 1e-10 and residual_norm < 1e-10),
    )


def _metric_obligations(metric: MetricContract) -> list[str]:
    owed: list[str] = []
    if not metric.null_centered:
        owed.append("NULL_CENTERED")
    if not metric.variance_reward_safe:
        owed.append("NO_VARIANCE_REWARD")
    if metric.shared_baseline and not metric.covariance_aware:
        owed.append("CORRELATED_ESTIMATES")
    return owed


def _best_route(case: ClaimCase) -> tuple[Route | None, Capability | None]:
    best_route = None
    best_cap = None
    for route in case.routes:
        cap = _capability_for(case, route.observation, route.nuisance)
        if best_cap is None or cap.max_accuracy > best_cap.max_accuracy:
            best_route, best_cap = route, cap
    return best_route, best_cap


def _projected_templates(
    case: ClaimCase,
    observation: np.ndarray,
    nuisance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = _inv_sqrt(case.covariance)
    nuisance_w = w @ np.asarray(nuisance, dtype=float)
    ma = _orthogonal_residual(w @ (observation @ case.latent_a), nuisance_w)
    mb = _orthogonal_residual(w @ (observation @ case.latent_b), nuisance_w)
    return ma, mb, w, nuisance_w


def _posterior(
    case: ClaimCase,
    samples: np.ndarray,
    observation: np.ndarray,
    nuisance: np.ndarray,
) -> tuple[str, float]:
    samples = np.asarray(samples, dtype=float)
    ma, mb, w, nuisance_w = _projected_templates(case, observation, nuisance)
    ybar = np.mean(samples, axis=0)
    yp = _orthogonal_residual(w @ ybar, nuisance_w)
    delta = ma - mb
    midpoint = 0.5 * (ma + mb)
    llr = float(len(samples) * np.dot(delta, yp - midpoint))
    llr = max(-60.0, min(60.0, llr))
    pa = 1.0 / (1.0 + math.exp(-llr))
    return ("A", pa) if pa >= 0.5 else ("B", 1.0 - pa)


def decide(
    case: ClaimCase,
    samples: np.ndarray | None = None,
    *,
    observation: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
) -> Decision:
    obs = case.observation if observation is None else np.asarray(observation, dtype=float)
    nui = case.nuisance if nuisance is None else np.asarray(nuisance, dtype=float)
    capability = _capability_for(case, obs, nui)
    obligations = _metric_obligations(case.metric)

    if obligations:
        return Decision(
            action=Action.REFUSE,
            reason="statistic has unpaid evidence obligations",
            obligations=obligations,
            capability=capability,
        )

    route, route_cap = _best_route(case)

    if capability.structurally_aliased:
        if route is not None and route_cap is not None and route_cap.max_accuracy >= case.target_accuracy:
            return Decision(
                action=Action.ROUTE,
                reason="current observation map annihilates the cause contrast; alternate readout restores it",
                obligations=[],
                capability=capability,
                route=route.name,
            )
        return Decision(
            action=Action.REFUSE,
            reason="cause contrast lies in the kernel of the current observation map",
            obligations=[],
            capability=capability,
        )

    if capability.nuisance_confounded:
        if route is not None and route_cap is not None and route_cap.max_accuracy >= case.target_accuracy:
            return Decision(
                action=Action.ROUTE,
                reason="cause contrast is absorbed by declared nuisance; alternate readout leaves a residual direction",
                obligations=[],
                capability=capability,
                route=route.name,
            )
        return Decision(
            action=Action.REFUSE,
            reason="cause contrast is fully contained in the declared nuisance tangent space",
            obligations=[],
            capability=capability,
        )

    if capability.max_accuracy < case.target_accuracy:
        if route is not None and route_cap is not None and route_cap.max_accuracy >= case.target_accuracy:
            return Decision(
                action=Action.ROUTE,
                reason="more samples cannot raise the current readout above the required recovery ceiling",
                obligations=[],
                capability=capability,
                route=route.name,
            )
        return Decision(
            action=Action.REFUSE,
            reason="even the maximum repeat budget cannot reach the declared recovery threshold",
            obligations=[],
            capability=capability,
        )

    n = 0 if samples is None else int(len(samples))
    if n == 0:
        return Decision(
            action=Action.REPEAT,
            reason="structure and metric are admissible but no empirical evidence has been collected",
            obligations=[],
            capability=capability,
            requested_repeats=max(1, case.initial_repeats or 1),
        )

    label, confidence = _posterior(case, np.asarray(samples), obs, nui)
    if confidence >= case.claim_confidence:
        return Decision(
            action=Action.CLAIM,
            reason="held evidence crosses the declared posterior-confidence threshold",
            obligations=[],
            capability=capability,
            claim=label,
            confidence=confidence,
        )

    if n < case.max_repeats:
        return Decision(
            action=Action.REPEAT,
            reason="claim is not yet calibrated, but additional replication can still cross the modeled ceiling",
            obligations=[],
            capability=capability,
            confidence=confidence,
            requested_repeats=min(max(1, n), case.max_repeats - n),
        )

    return Decision(
        action=Action.REFUSE,
        reason="repeat budget exhausted without calibrated evidence",
        obligations=[],
        capability=capability,
        confidence=confidence,
    )


def _cases() -> list[ClaimCase]:
    a = np.asarray([1.0, 0.0])
    b = np.asarray([0.0, 1.0])
    i2 = np.eye(2)
    z2 = np.zeros((2, 0))
    good = MetricContract("noise-aware likelihood", "evidence", True, False, True, True)
    bad_gram = MetricContract(
        "raw shared-baseline Fisher Gram",
        "evidence_information",
        covariance_aware=False,
        shared_baseline=True,
        variance_reward_safe=False,
        null_centered=False,
    )

    return [
        ClaimCase(
            "exact_alias_refuse",
            a,
            b,
            np.asarray([[1.0, 1.0]]),
            np.zeros((1, 0)),
            np.eye(1),
            good,
            max_repeats=48,
            expected_initial=Action.REFUSE,
        ),
        ClaimCase(
            "exact_alias_route",
            a,
            b,
            np.asarray([[1.0, 1.0]]),
            np.zeros((1, 0)),
            np.eye(1),
            good,
            routes=(Route("distributed_receiver", i2, z2),),
            max_repeats=16,
            expected_initial=Action.ROUTE,
        ),
        ClaimCase(
            "low_ceiling_route",
            a,
            b,
            0.05 * i2,
            z2,
            i2,
            good,
            routes=(Route("higher-dimensional_receiver", i2, z2),),
            target_accuracy=0.90,
            max_repeats=48,
            expected_initial=Action.ROUTE,
        ),
        ClaimCase(
            "low_ceiling_refuse",
            a,
            b,
            0.05 * i2,
            z2,
            i2,
            good,
            target_accuracy=0.90,
            max_repeats=48,
            expected_initial=Action.REFUSE,
        ),
        ClaimCase(
            "nuisance_route",
            a,
            b,
            i2,
            np.asarray([[1.0], [-1.0]]),
            i2,
            good,
            routes=(
                Route(
                    "nuisance-breaking_receiver",
                    np.asarray([[1.0, 0.0], [0.0, 2.0]]),
                    np.asarray([[1.0], [-1.0]]),
                ),
            ),
            max_repeats=24,
            expected_initial=Action.ROUTE,
        ),
        ClaimCase(
            "wrong_metric_refuse",
            a,
            b,
            i2,
            z2,
            i2,
            bad_gram,
            max_repeats=48,
            expected_initial=Action.REFUSE,
        ),
        ClaimCase(
            "repeat_helpful",
            a,
            b,
            0.8 * i2,
            z2,
            i2,
            good,
            target_accuracy=0.90,
            claim_confidence=0.95,
            initial_repeats=2,
            max_repeats=32,
            expected_initial=Action.REPEAT,
        ),
        ClaimCase(
            "claim_now",
            a,
            b,
            2.0 * i2,
            z2,
            i2,
            good,
            target_accuracy=0.90,
            claim_confidence=0.95,
            initial_repeats=4,
            max_repeats=16,
            expected_initial=Action.CLAIM,
        ),
    ]


def _canonical_samples(case: ClaimCase) -> np.ndarray | None:
    if case.initial_repeats <= 0:
        return None
    # Canonical action receipt: evidence exactly at the A mean. Randomized trials
    # below are the real calibration test; this deterministic point only checks
    # the state-machine branch expected for a representative evidence state.
    mu = case.observation @ case.latent_a
    return np.repeat(mu.reshape(1, -1), case.initial_repeats, axis=0)


def _sample_stream(
    case: ClaimCase,
    true_label: str,
    rng: np.random.Generator,
    *,
    observation: np.ndarray,
    nuisance: np.ndarray,
    total: int,
) -> np.ndarray:
    latent = case.latent_a if true_label == "A" else case.latent_b
    mu = observation @ latent
    nuisance = np.asarray(nuisance, dtype=float)
    if nuisance.size and nuisance.shape[1]:
        beta = rng.normal(0.0, 0.8, size=nuisance.shape[1])
        mu = mu + nuisance @ beta
    chol = np.linalg.cholesky(case.covariance)
    return np.asarray([mu + chol @ rng.normal(size=len(mu)) for _ in range(total)])


def _run_recoverable_trials(
    case: ClaimCase,
    *,
    observation: np.ndarray,
    nuisance: np.ndarray,
    initial_repeats: int,
    episodes: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    claims = correct = immediate_claims = 0
    calls: list[int] = []
    refusals = 0

    for episode in range(episodes):
        true = "A" if episode % 2 == 0 else "B"
        stream = _sample_stream(
            case,
            true,
            rng,
            observation=observation,
            nuisance=nuisance,
            total=case.max_repeats,
        )
        n = initial_repeats
        samples = stream[:n] if n else np.empty((0, observation.shape[0]))
        first = True
        while True:
            decision = decide(
                case,
                samples if len(samples) else None,
                observation=observation,
                nuisance=nuisance,
            )
            if decision.action == Action.CLAIM:
                claims += 1
                correct += int(decision.claim == true)
                immediate_claims += int(first)
                calls.append(n)
                break
            if decision.action == Action.REPEAT:
                add = max(1, int(decision.requested_repeats))
                n2 = min(case.max_repeats, n + add)
                samples = stream[:n2]
                n = n2
                first = False
                if n >= case.max_repeats and add == 0:
                    raise RuntimeError("repeat policy made no progress")
                continue
            if decision.action == Action.REFUSE:
                refusals += 1
                calls.append(n)
                break
            raise RuntimeError(f"unexpected ROUTE inside post-route/recoverable trial: {case.name}")

    claim_accuracy = 0.0 if claims == 0 else correct / claims
    return {
        "episodes": episodes,
        "claims": claims,
        "refusals": refusals,
        "claim_coverage": claims / episodes,
        "claim_accuracy": claim_accuracy,
        "false_claim_rate": 0.0 if claims == 0 else 1.0 - claim_accuracy,
        "mean_calls": float(np.mean(calls)),
        "median_calls": float(np.median(calls)),
        "immediate_claim_fraction": immediate_claims / episodes,
    }


def _raw_gram_null_variance_receipt(seed: int = 55) -> dict[str, float]:
    """Vahti-shaped metamorphic check for Gate-4a-style raw Fisher geometry.

    True A and B effects are both zero.  They are estimated against the same
    noisy baseline.  The smallest eigenvalue of the raw two-column Gram matrix
    should represent zero information, but its mean rises as observation noise
    is increased.  This is exactly the obligation that an evidence-information
    statistic must discharge before being trusted.
    """
    rng = np.random.default_rng(seed)

    def score(noise_sd: float) -> float:
        values = []
        locations = 48
        repeats = 6
        dims = 5
        for _ in range(500):
            base = rng.normal(0.0, noise_sd, size=(locations, repeats, dims)).mean(axis=1)
            a = rng.normal(0.0, noise_sd, size=(locations, repeats, dims)).mean(axis=1)
            b = rng.normal(0.0, noise_sd, size=(locations, repeats, dims)).mean(axis=1)
            ea = (a - base).reshape(-1)
            eb = (b - base).reshape(-1)
            gram = np.asarray([[ea @ ea, ea @ eb], [ea @ eb, eb @ eb]])
            values.append(float(np.linalg.eigvalsh(gram)[0]))
        return float(np.mean(values))

    low = score(0.5)
    high = score(1.0)
    return {
        "null_true_information": 0.0,
        "mean_raw_smin_sd_0_5": low,
        "mean_raw_smin_sd_1_0": high,
        "variance_reward_ratio": high / max(low, 1e-12),
    }


def run(output: Path) -> dict[str, object]:
    cases = _cases()
    canonical = []
    pre_spend = {"REFUSE", "ROUTE"}
    all_initial_correct = True
    zero_spend_correct = True

    for case in cases:
        samples = _canonical_samples(case)
        decision = decide(case, samples)
        all_initial_correct &= decision.action == case.expected_initial
        requested = int(decision.requested_repeats)
        if case.expected_initial.value in pre_spend:
            zero_spend_correct &= requested == 0
        route, route_cap = _best_route(case)
        canonical.append(
            {
                "case": case.name,
                "expected": case.expected_initial.value,
                "decision": decision.action.value,
                "reason": decision.reason,
                "obligations": decision.obligations,
                "requested_repeats": requested,
                "current_capability": asdict(decision.capability),
                "recommended_route": decision.route,
                "best_route_capability": None if route_cap is None else asdict(route_cap),
                "confidence": decision.confidence,
            }
        )

    case_by_name = {c.name: c for c in cases}
    stochastic: dict[str, object] = {}

    repeat_case = case_by_name["repeat_helpful"]
    stochastic["repeat_helpful"] = _run_recoverable_trials(
        repeat_case,
        observation=repeat_case.observation,
        nuisance=repeat_case.nuisance,
        initial_repeats=repeat_case.initial_repeats,
        episodes=800,
        seed=5101,
    )

    claim_case = case_by_name["claim_now"]
    stochastic["claim_now"] = _run_recoverable_trials(
        claim_case,
        observation=claim_case.observation,
        nuisance=claim_case.nuisance,
        initial_repeats=claim_case.initial_repeats,
        episodes=800,
        seed=5102,
    )

    route_trials = {}
    for j, name in enumerate(("exact_alias_route", "low_ceiling_route", "nuisance_route")):
        case = case_by_name[name]
        route, route_cap = _best_route(case)
        assert route is not None and route_cap is not None
        route_trials[name] = {
            "route": route.name,
            "current_max_accuracy": _capability_for(case, case.observation, case.nuisance).max_accuracy,
            "routed_max_accuracy": route_cap.max_accuracy,
            "post_route": _run_recoverable_trials(
                case,
                observation=route.observation,
                nuisance=route.nuisance,
                initial_repeats=2,
                episodes=600,
                seed=5200 + j,
            ),
        }
    stochastic["routed_cases"] = route_trials

    metric_receipt = _raw_gram_null_variance_receipt()
    wrong_metric_row = next(row for row in canonical if row["case"] == "wrong_metric_refuse")

    # Counterfactual WAIT-only cost on the classes that our pre-spend audit
    # refuses/routes immediately.  It blindly spends every allowed repeat at the
    # current readout before giving up or choosing a label.
    pre_spend_cases = [
        case_by_name[n]
        for n in (
            "exact_alias_refuse",
            "exact_alias_route",
            "low_ceiling_route",
            "low_ceiling_refuse",
            "nuisance_route",
            "wrong_metric_refuse",
        )
    ]
    wait_only_calls = int(sum(c.max_repeats for c in pre_spend_cases))
    admission_calls = 0

    repeat_stats = stochastic["repeat_helpful"]
    claim_stats = stochastic["claim_now"]
    routed_stats = [v["post_route"] for v in route_trials.values()]

    gate = {
        "canonical_actions_all_correct": bool(all_initial_correct),
        "pre_spend_refuse_or_route_costs_zero": bool(zero_spend_correct),
        "wrong_metric_generates_vahti_style_obligations": {
            "NO_VARIANCE_REWARD",
            "CORRELATED_ESTIMATES",
            "NULL_CENTERED",
        }.issubset(set(wrong_metric_row["obligations"])),
        "raw_gram_null_rewards_variance": metric_receipt["variance_reward_ratio"] > 3.0,
        "repeat_helpful_claim_accuracy_ge_0_95": repeat_stats["claim_accuracy"] >= 0.95,
        "repeat_helpful_coverage_ge_0_90": repeat_stats["claim_coverage"] >= 0.90,
        "repeat_helpful_mean_calls_lt_half_max": repeat_stats["mean_calls"] < repeat_case.max_repeats / 2,
        "claim_now_immediate_ge_0_90": claim_stats["immediate_claim_fraction"] >= 0.90,
        "claim_now_false_claim_rate_le_0_05": claim_stats["false_claim_rate"] <= 0.05,
        "all_routes_raise_ceiling_by_0_10": all(
            row["routed_max_accuracy"] >= row["current_max_accuracy"] + 0.10
            for row in route_trials.values()
        ),
        "all_routes_reach_target_ceiling": all(
            row["routed_max_accuracy"] >= case_by_name[name].target_accuracy
            for name, row in route_trials.items()
        ),
        "post_route_claim_accuracy_ge_0_95": all(
            row["post_route"]["claim_accuracy"] >= 0.95 for row in route_trials.values()
        ),
        "wait_only_would_spend_positive_calls_before_same_precheck": wait_only_calls > admission_calls,
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
        "status": "Gate 5 evidence admission",
        "principle": "a demand creates obligations; it does not authorize a claim",
        "actions": [a.value for a in Action],
        "canonical_cases": canonical,
        "stochastic_validation": stochastic,
        "metric_variance_receipt": metric_receipt,
        "cost_counterfactual": {
            "pre_spend_cases": [c.name for c in pre_spend_cases],
            "admission_policy_new_observation_calls_before_REFUSE_or_ROUTE": admission_calls,
            "wait_only_max_repeat_calls": wait_only_calls,
        },
        "gate": gate,
    }

    print("\nGATE 5 — EVIDENCE ADMISSION")
    for row in canonical:
        print(
            f"{row['case']:24s} expected={row['expected']:6s} got={row['decision']:6s} "
            f"raw={row['current_capability']['raw_contrast_norm']:.4g} "
            f"resid={row['current_capability']['whitened_residual_norm']:.4g} "
            f"ceiling={row['current_capability']['max_accuracy']:.3f}"
        )
        if row["obligations"]:
            print("  owes: " + ", ".join(row["obligations"]))
        if row["recommended_route"]:
            print("  route: " + str(row["recommended_route"]))

    print("\nrandomized recoverability:")
    print(
        "repeat_helpful  "
        f"accuracy={repeat_stats['claim_accuracy']:.3f} coverage={repeat_stats['claim_coverage']:.3f} "
        f"mean_calls={repeat_stats['mean_calls']:.2f}/{repeat_case.max_repeats}"
    )
    print(
        "claim_now       "
        f"accuracy={claim_stats['claim_accuracy']:.3f} immediate={claim_stats['immediate_claim_fraction']:.3f} "
        f"mean_calls={claim_stats['mean_calls']:.2f}/{claim_case.max_repeats}"
    )
    for name, row in route_trials.items():
        ps = row["post_route"]
        print(
            f"{name:16s} ceiling {row['current_max_accuracy']:.3f}->{row['routed_max_accuracy']:.3f} "
            f"post-route accuracy={ps['claim_accuracy']:.3f} coverage={ps['claim_coverage']:.3f}"
        )
    print(
        "\nraw shared-baseline Gram null variance reward: "
        f"{metric_receipt['mean_raw_smin_sd_0_5']:.3f} -> {metric_receipt['mean_raw_smin_sd_1_0']:.3f} "
        f"({metric_receipt['variance_reward_ratio']:.2f}x)"
    )
    print(
        f"pre-spend calls: admission={admission_calls}, WAIT-only counterfactual={wait_only_calls}"
    )
    print("GATE:", json.dumps(gate, sort_keys=True))

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("gate5_result.json"))
    args = ap.parse_args()
    result = run(args.output)
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 5 did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
