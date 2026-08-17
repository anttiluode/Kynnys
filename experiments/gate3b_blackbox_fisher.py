from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from experiments import gate3_blackbox_probe as g3


@dataclass
class FisherResult:
    name: str
    indices: list[int]
    eta: float
    cosine: float
    sensitivity_a: float
    sensitivity_b: float
    smallest_singular: float
    largest_singular: float
    condition_number: float
    weak_direction_snr_at_sigma_0_01: float
    queries: int


def _p90(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return math.nan
    return xs[int(round(0.9 * (len(xs) - 1)))]


def _singular_values(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    design = np.column_stack([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    s = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    if len(s) == 1:
        return 0.0, float(s[0])
    return float(s[-1]), float(s[0])


def _result(name: str, indices: Sequence[int], a: np.ndarray, b: np.ndarray, queries: int) -> FisherResult:
    idx = np.asarray(indices, dtype=int)
    snap = g3._snapshot(a[idx], b[idx])
    smin, smax = _singular_values(a[idx], b[idx])
    cond = math.inf if smin <= 1e-18 else smax / smin
    return FisherResult(
        name=name,
        indices=[int(i) for i in idx],
        eta=snap.eta,
        cosine=snap.cosine,
        sensitivity_a=snap.sensitivity_a,
        sensitivity_b=snap.sensitivity_b,
        smallest_singular=smin,
        largest_singular=smax,
        condition_number=cond,
        weak_direction_snr_at_sigma_0_01=smin / 0.01,
        queries=int(queries),
    )


def _candidate_smin(
    g11: float,
    g12: float,
    g22: float,
    pa: np.ndarray,
    pb: np.ndarray,
) -> np.ndarray:
    # Smallest singular value after adding one predicted observation [pa,pb].
    a = g11 + pa * pa
    c = g12 + pa * pb
    d = g22 + pb * pb
    disc = np.sqrt(np.maximum(0.0, (a - d) ** 2 + 4.0 * c * c))
    lam_min = 0.5 * (a + d - disc)
    return np.sqrt(np.maximum(0.0, lam_min))


def _fisher_active(
    texts: Sequence[str],
    baseline_margin: np.ndarray,
    oracle: g3.PairedOracle,
    *,
    budget: int,
    random_seed_queries: int,
    boundary_seed_queries: int,
) -> list[int]:
    """Black-box D-optimal-ish sequential design.

    Inputs available to this function:
      * raw candidate text,
      * baseline black-box score for every candidate,
      * a budgeted callback returning paired A/B effects for selected probes.

    It receives no classifier vectorizer, coefficients, gradients, or hidden
    effect arrays.
    """

    x = g3._selector_features(texts, baseline_margin)
    n = len(texts)
    rng = random.Random(20260817)

    selected: list[int] = []
    oa: list[float] = []
    ob: list[float] = []

    def take(idx: int) -> None:
        if idx in selected:
            return
        a, b = oracle.query(idx)
        selected.append(int(idx))
        oa.append(float(a))
        ob.append(float(b))

    # A generic mixed warm start: some coverage of ordinary traffic, plus some
    # boundary probes because small baseline response is a common black-box
    # diagnostic heuristic. No counterfactual effect is used to choose seeds.
    for idx in rng.sample(range(n), random_seed_queries):
        take(idx)
    for idx in np.argsort(np.abs(baseline_margin)):
        if len(selected) >= random_seed_queries + boundary_seed_queries:
            break
        take(int(idx))

    while len(selected) < budget:
        idx_arr = np.asarray(selected, dtype=int)
        y = np.column_stack([oa, ob])
        a_obs = np.asarray(oa, dtype=float)
        b_obs = np.asarray(ob, dtype=float)
        g11 = float(np.dot(a_obs, a_obs))
        g12 = float(np.dot(a_obs, b_obs))
        g22 = float(np.dot(b_obs, b_obs))

        committee = []
        crng = np.random.default_rng(9000 + len(selected))
        for _ in range(12):
            boot = crng.integers(0, len(selected), size=len(selected))
            model = Ridge(alpha=2.0, solver="lsqr")
            model.fit(x[idx_arr[boot]], y[boot])
            committee.append(np.asarray(model.predict(x), dtype=float))
        pred = np.stack(committee, axis=0)

        fisher = []
        for member in pred:
            fisher.append(_candidate_smin(g11, g12, g22, member[:, 0], member[:, 1]))
        fisher = np.stack(fisher, axis=0)
        # Exploit predicted weak-direction information, but retain a moderate
        # committee uncertainty bonus so the surrogate can discover regions it
        # has not yet modeled well.
        score = np.mean(fisher, axis=0) + 0.75 * np.std(fisher, axis=0)
        score[idx_arr] = -np.inf
        take(int(np.argmax(score)))

    if oracle.queries != budget:
        raise AssertionError(f"oracle used {oracle.queries}; expected {budget}")
    return selected


def run(
    output: Path,
    *,
    budget: int = 48,
    random_seed_queries: int = 6,
    boundary_seed_queries: int = 6,
    random_trials: int = 400,
) -> dict:
    train, test, model_vec, baseline, regularized, chosen_c, scale, natural = g3._fit_pipeline()
    texts, meta = g3._candidate_texts(train.data, test.data)
    x_candidates = model_vec.transform(texts)

    # Hidden benchmark truth. Only the PairedOracle exposes selected rows to the
    # active design algorithm.
    base = np.asarray(baseline.decision_function(x_candidates), dtype=float)
    reg = np.asarray(regularized.decision_function(x_candidates), dtype=float)
    temp = scale * base
    effect_a = reg - base
    effect_b = temp - base

    oracle = g3.PairedOracle(effect_a, effect_b, budget)
    active_idx = _fisher_active(
        texts,
        base,
        oracle,
        budget=budget,
        random_seed_queries=random_seed_queries,
        boundary_seed_queries=boundary_seed_queries,
    )
    active = _result("fisher_active", active_idx, effect_a, effect_b, oracle.queries)

    uncertainty_idx = np.argsort(np.abs(base))[:budget]
    uncertainty = _result("uncertainty", uncertainty_idx, effect_a, effect_b, budget)

    # Static mixed attacker: same 50/50 idea as the active warm start, but no
    # learning from paid paired outcomes.
    rng_mix = random.Random(31337)
    mix = list(map(int, np.argsort(np.abs(base))[: budget // 2]))
    remaining = [i for i in range(len(texts)) if i not in set(mix)]
    mix.extend(rng_mix.sample(remaining, budget - len(mix)))
    mixed = _result("static_mixed", mix, effect_a, effect_b, budget)

    rng = random.Random(17)
    population = list(range(len(texts)))
    random_smin: list[float] = []
    random_eta: list[float] = []
    for _ in range(random_trials):
        idx = rng.sample(population, budget)
        row = _result("random", idx, effect_a, effect_b, budget)
        random_smin.append(row.smallest_singular)
        random_eta.append(row.eta)

    # Exact non-identifiability control on the active selected inputs.
    a_neg = effect_a[np.asarray(active_idx, dtype=int)]
    b_neg = 3.0 * a_neg
    neg_smin, neg_smax = _singular_values(a_neg, b_neg)
    neg_snap = g3._snapshot(a_neg, b_neg)

    active_kinds = [meta["kinds"][i] for i in active_idx]

    gate = {
        "natural_outputs_confounded": natural.eta < 0.05 and natural.cosine > 0.99,
        "absolute_weak_direction_signal": active.smallest_singular >= 0.05,
        "active_beats_uncertainty": active.smallest_singular > uncertainty.smallest_singular,
        "active_beats_static_mixed": active.smallest_singular > mixed.smallest_singular,
        "active_beats_random_p90": active.smallest_singular > _p90(random_smin),
        "negative_control_singular": neg_smin < 1e-9 and neg_snap.eta < 1e-9,
        "budget_respected": oracle.queries == budget,
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
        "definition": {
            "why_smallest_singular": (
                "For additive two-cause attribution under isotropic observation noise, "
                "the weakest Fisher-information direction is proportional to the square "
                "of the smallest singular value of the two-column effect matrix. It "
                "penalizes both collinearity and vanishing effect magnitude."
            ),
            "reference_noise_sigma": 0.01,
        },
        "dataset": {
            "name": "20 Newsgroups",
            "categories": list(g3.CATEGORIES),
            "candidate_pool": len(texts),
        },
        "hard_pair": {
            "regularization_C": chosen_c,
            "temperature_scale": scale,
            "natural": asdict(natural),
        },
        "budget": {
            "total": budget,
            "random_warm_start": random_seed_queries,
            "boundary_warm_start": boundary_seed_queries,
        },
        "strategies": {
            "active": asdict(active),
            "uncertainty": asdict(uncertainty),
            "static_mixed": asdict(mixed),
            "random": {
                "trials": random_trials,
                "smin_median": median(random_smin),
                "smin_p90": _p90(random_smin),
                "smin_max": max(random_smin),
                "eta_median": median(random_eta),
                "eta_p90": _p90(random_eta),
            },
        },
        "active_selected_kinds": {
            k: active_kinds.count(k) for k in sorted(set(active_kinds))
        },
        "active_examples": [texts[i][:160] for i in active_idx[:16]],
        "negative_control": {
            "eta": neg_snap.eta,
            "smallest_singular": neg_smin,
            "largest_singular": neg_smax,
        },
        "gate": gate,
    }

    print("\nGATE 3b — BLACK-BOX FISHER ATTRIBUTION")
    print(f"natural: eta={natural.eta:.6f} cos={natural.cosine:.6f}")
    print(f"pool={len(texts)} budget={budget}")
    for row in (active, uncertainty, mixed):
        print(
            f"{row.name:14s} smin={row.smallest_singular:.6f} "
            f"eta={row.eta:.6f} S=({row.sensitivity_a:.4f},{row.sensitivity_b:.4f}) "
            f"cond={row.condition_number:.1f} SNR@.01={row.weak_direction_snr_at_sigma_0_01:.2f}"
        )
    print(
        f"random smin: median={median(random_smin):.6f} "
        f"p90={_p90(random_smin):.6f} max={max(random_smin):.6f}"
    )
    print(f"active kinds: {result['active_selected_kinds']}")
    print(f"negative control: smin={neg_smin:.3e} eta={neg_snap.eta:.3e}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gate3b_result.json"))
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--random-seed-queries", type=int, default=6)
    parser.add_argument("--boundary-seed-queries", type=int, default=6)
    parser.add_argument("--random-trials", type=int, default=400)
    args = parser.parse_args()
    result = run(
        args.output,
        budget=args.budget,
        random_seed_queries=args.random_seed_queries,
        boundary_seed_queries=args.boundary_seed_queries,
        random_trials=args.random_trials,
    )
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 3b did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
