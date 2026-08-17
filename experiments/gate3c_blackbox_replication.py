from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from experiments import gate3_blackbox_probe as g3
from experiments import gate3b_blackbox_fisher as g3b


SEEDS = (101, 211, 307, 419, 503, 607, 709, 811)


def _p90(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    return xs[int(round(0.9 * (len(xs) - 1)))]


def _fisher_active_seed(
    x,
    baseline_margin: np.ndarray,
    oracle: g3.PairedOracle,
    *,
    budget: int,
    random_seed_queries: int,
    boundary_seed_queries: int,
    seed: int,
) -> list[int]:
    n = x.shape[0]
    rng = random.Random(seed)
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

        crng = np.random.default_rng(seed * 10000 + len(selected))
        fisher = []
        for _ in range(12):
            boot = crng.integers(0, len(selected), size=len(selected))
            model = Ridge(alpha=2.0, solver="lsqr")
            model.fit(x[idx_arr[boot]], y[boot])
            pred = np.asarray(model.predict(x), dtype=float)
            fisher.append(
                g3b._candidate_smin(g11, g12, g22, pred[:, 0], pred[:, 1])
            )
        fisher = np.stack(fisher, axis=0)
        score = np.mean(fisher, axis=0) + 0.75 * np.std(fisher, axis=0)
        score[idx_arr] = -np.inf
        take(int(np.argmax(score)))

    if oracle.queries != budget:
        raise AssertionError(f"seed {seed}: used {oracle.queries}, expected {budget}")
    return selected


def run(
    output: Path,
    *,
    budget: int = 48,
    random_seed_queries: int = 6,
    boundary_seed_queries: int = 6,
    baseline_trials: int = 400,
) -> dict:
    train, test, model_vec, baseline, regularized, chosen_c, scale, natural = g3._fit_pipeline()
    texts, meta = g3._candidate_texts(train.data, test.data)
    x_candidates = model_vec.transform(texts)
    base = np.asarray(baseline.decision_function(x_candidates), dtype=float)
    reg = np.asarray(regularized.decision_function(x_candidates), dtype=float)
    temp = scale * base
    effect_a = reg - base
    effect_b = temp - base

    selector_x = g3._selector_features(texts, base)

    active_rows = []
    selected_sets = []
    for seed in SEEDS:
        oracle = g3.PairedOracle(effect_a, effect_b, budget)
        idx = _fisher_active_seed(
            selector_x,
            base,
            oracle,
            budget=budget,
            random_seed_queries=random_seed_queries,
            boundary_seed_queries=boundary_seed_queries,
            seed=seed,
        )
        row = g3b._result(f"active_seed_{seed}", idx, effect_a, effect_b, oracle.queries)
        active_rows.append(row)
        selected_sets.append(idx)
        print(
            f"seed {seed}: smin={row.smallest_singular:.6f} eta={row.eta:.6f} "
            f"cond={row.condition_number:.1f}"
        )

    uncertainty_idx = np.argsort(np.abs(base))[:budget]
    uncertainty = g3b._result("uncertainty", uncertainty_idx, effect_a, effect_b, budget)

    rng = random.Random(1701)
    population = list(range(len(texts)))
    random_smin: list[float] = []
    mixed_smin: list[float] = []
    uncertainty_half = list(map(int, np.argsort(np.abs(base))[: budget // 2]))
    uncertainty_half_set = set(uncertainty_half)
    mixed_remaining = [i for i in population if i not in uncertainty_half_set]

    for trial in range(baseline_trials):
        ridx = rng.sample(population, budget)
        random_smin.append(
            g3b._result("random", ridx, effect_a, effect_b, budget).smallest_singular
        )

        mrng = random.Random(900000 + trial)
        midx = uncertainty_half + mrng.sample(
            mixed_remaining, budget - len(uncertainty_half)
        )
        mixed_smin.append(
            g3b._result("mixed", midx, effect_a, effect_b, budget).smallest_singular
        )

    active_smin = [r.smallest_singular for r in active_rows]
    active_eta = [r.eta for r in active_rows]
    random_p90 = _p90(random_smin)
    mixed_p90 = _p90(mixed_smin)

    # Exact non-identifiable control on the first replicated active set.
    neg_idx = np.asarray(selected_sets[0], dtype=int)
    neg_a = effect_a[neg_idx]
    neg_b = 3.0 * neg_a
    neg_smin, neg_smax = g3b._singular_values(neg_a, neg_b)
    neg_snap = g3._snapshot(neg_a, neg_b)

    fraction_above_random_p90 = sum(v > random_p90 for v in active_smin) / len(active_smin)
    fraction_above_mixed_p90 = sum(v > mixed_p90 for v in active_smin) / len(active_smin)

    gate = {
        "natural_outputs_confounded": natural.eta < 0.05 and natural.cosine > 0.99,
        "median_active_absolute_signal": median(active_smin) >= 0.05,
        "median_active_beats_random_p90": median(active_smin) > random_p90,
        "median_active_beats_mixed_p90": median(active_smin) > mixed_p90,
        "at_least_75pct_active_beats_random_p90": fraction_above_random_p90 >= 0.75,
        "all_active_beats_uncertainty": min(active_smin) > uncertainty.smallest_singular,
        "negative_control_singular": neg_smin < 1e-9 and neg_snap.eta < 1e-9,
        "all_budgets_respected": all(r.queries == budget for r in active_rows),
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
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
        "replication": {
            "seeds": list(SEEDS),
            "active": [asdict(r) for r in active_rows],
            "smin_median": median(active_smin),
            "smin_min": min(active_smin),
            "smin_max": max(active_smin),
            "eta_median": median(active_eta),
            "fraction_above_random_p90": fraction_above_random_p90,
            "fraction_above_mixed_p90": fraction_above_mixed_p90,
        },
        "baselines": {
            "uncertainty": asdict(uncertainty),
            "random": {
                "trials": baseline_trials,
                "smin_median": median(random_smin),
                "smin_p90": random_p90,
                "smin_max": max(random_smin),
            },
            "static_mixed": {
                "trials": baseline_trials,
                "smin_median": median(mixed_smin),
                "smin_p90": mixed_p90,
                "smin_max": max(mixed_smin),
            },
        },
        "negative_control": {
            "eta": neg_snap.eta,
            "smallest_singular": neg_smin,
            "largest_singular": neg_smax,
        },
        "gate": gate,
    }

    print("\nGATE 3c — REPLICATED BLACK-BOX FISHER ATTRIBUTION")
    print(f"active smin: min={min(active_smin):.6f} median={median(active_smin):.6f} max={max(active_smin):.6f}")
    print(f"active eta median={median(active_eta):.6f}")
    print(f"random: median={median(random_smin):.6f} p90={random_p90:.6f} max={max(random_smin):.6f}")
    print(f"mixed:  median={median(mixed_smin):.6f} p90={mixed_p90:.6f} max={max(mixed_smin):.6f}")
    print(f"uncertainty smin={uncertainty.smallest_singular:.6f}")
    print(f"fraction active > random p90={fraction_above_random_p90:.3f}")
    print(f"fraction active > mixed p90={fraction_above_mixed_p90:.3f}")
    print(f"negative control smin={neg_smin:.3e} eta={neg_snap.eta:.3e}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gate3c_result.json"))
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--random-seed-queries", type=int, default=6)
    parser.add_argument("--boundary-seed-queries", type=int, default=6)
    parser.add_argument("--baseline-trials", type=int, default=400)
    args = parser.parse_args()
    result = run(
        args.output,
        budget=args.budget,
        random_seed_queries=args.random_seed_queries,
        boundary_seed_queries=args.boundary_seed_queries,
        baseline_trials=args.baseline_trials,
    )
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 3c did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
