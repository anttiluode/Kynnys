from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path

import numpy as np

from experiments import gate3_blackbox_probe as g3
from experiments import gate4_stochastic_structured as g4
from experiments import gate4b_noise_corrected as g4b


@dataclass
class Row:
    repeats: int
    total_calls: int
    strategy: str
    truth_smin: float
    truth_eta: float
    recovery: float
    recovery_a: float
    recovery_b: float
    reliability: float


def _score_denoised(
    name: str,
    selected: list[int],
    batches,
    oracle: g4.ChargedOracle,
    scout_texts,
    truth_ea,
    truth_eb,
    truth_cov,
    truth_streams,
    *,
    repeats: int,
    recovery_seed: int,
) -> Row:
    da, db, metric_cov, reliability, _ = g4b._denoised_templates(
        scout_texts, oracle.scout, selected, batches, repeats
    )
    recovery, ra, rb = g4b._recover_with_templates(
        selected,
        da,
        db,
        metric_cov,
        truth_streams,
        repeats=repeats,
        trials=240,
        seed=recovery_seed,
    )
    sel = np.asarray(selected, dtype=int)
    truth_mean_cov = (2.0 / repeats) * truth_cov + 2e-3 * np.eye(g4.OBS_DIM)
    smin, _, _ = g4._design_metrics(truth_ea[sel], truth_eb[sel], truth_mean_cov)
    return Row(
        repeats=repeats,
        total_calls=oracle.calls,
        strategy=name,
        truth_smin=smin,
        truth_eta=g4._eta(truth_ea[sel], truth_eb[sel]),
        recovery=recovery,
        recovery_a=ra,
        recovery_b=rb,
        reliability=reliability,
    )


def run(output: Path) -> dict:
    scout_count = 512
    locations = 48
    warm_start = 12
    repeat_levels = (6, 12, 24, 48)

    train, test, model_vec, baseline, regularized, chosen_c, scale, natural = g3._fit_pipeline()
    texts_all, _ = g3._candidate_texts(train.data, test.data)
    x_all = model_vec.transform(texts_all)
    base_margin = np.asarray(baseline.decision_function(x_all), dtype=float)
    reg_margin = np.asarray(regularized.decision_function(x_all), dtype=float)
    margins = {"base": base_margin, "A": reg_margin, "B": scale * base_margin}

    rng = random.Random(4404)
    scout_global = sorted(rng.sample(range(len(texts_all)), scout_count))
    scout_texts = [texts_all[i] for i in scout_global]

    # One fixed stochastic world. Higher budgets reveal longer prefixes of the
    # same per-candidate streams instead of receiving a luckier world.
    streams = g4._candidate_streams(
        margins,
        scout_global,
        max_repeats=max(repeat_levels),
        world_seed=404,
    )
    truth_ea, truth_eb, truth_cov = g4._truth_for_scout(
        margins, scout_global, repeats=3500, seed=8_800_000
    )
    truth_streams = g4._candidate_streams(
        margins, scout_global, max_repeats=1200, world_seed=9_991
    )

    rows: list[Row] = []
    final_noise_bias = {}
    selected_by_level = {}

    for repeats in repeat_levels:
        call_budget = scout_count + locations * (3 * repeats - 1)

        active_oracle = g4.ChargedOracle(
            streams,
            repeats=repeats,
            scout_count=scout_count,
            call_budget=call_budget,
        )
        active_idx, active_batches, diagnostics = g4b._noise_corrected_active(
            scout_texts,
            active_oracle,
            locations=locations,
            warm_start=warm_start,
            seed=4040,
        )
        selected_by_level[str(repeats)] = list(map(int, active_idx))
        final_noise_bias[str(repeats)] = diagnostics[-1]
        rows.append(
            _score_denoised(
                "active",
                active_idx,
                active_batches,
                active_oracle,
                scout_texts,
                truth_ea,
                truth_eb,
                truth_cov,
                truth_streams,
                repeats=repeats,
                recovery_seed=10_000 + repeats,
            )
        )

        for j, name in enumerate(("uncertainty", "static_mixed", "random")):
            oracle = g4.ChargedOracle(
                streams,
                repeats=repeats,
                scout_count=scout_count,
                call_budget=call_budget,
            )
            idx = g4._static_indices(name, oracle.scout, locations, 990 + j)
            batches = g4._collect_static(idx, oracle)
            rows.append(
                _score_denoised(
                    name,
                    idx,
                    batches,
                    oracle,
                    scout_texts,
                    truth_ea,
                    truth_eb,
                    truth_cov,
                    truth_streams,
                    repeats=repeats,
                    recovery_seed=20_000 + 100 * j + repeats,
                )
            )

    by_repeat: dict[int, dict[str, Row]] = {}
    for row in rows:
        by_repeat.setdefault(row.repeats, {})[row.strategy] = row

    crossover = None
    for r in repeat_levels:
        if by_repeat[r]["active"].recovery >= 0.75:
            crossover = r
            break

    if crossover is None:
        active_advantage = False
        best_baseline = None
    else:
        baseline_rows = [
            by_repeat[crossover][n] for n in ("uncertainty", "static_mixed", "random")
        ]
        best_baseline = max(baseline_rows, key=lambda x: x.recovery)
        active_advantage = (
            by_repeat[crossover]["active"].recovery >= best_baseline.recovery + 0.05
        )

    bias_seq = [
        final_noise_bias[str(r)]["noise_bias_aa_per_location"] for r in repeat_levels
    ]
    bias_declines = all(bias_seq[i + 1] < bias_seq[i] for i in range(len(bias_seq) - 1))

    sel = np.asarray(selected_by_level[str(max(repeat_levels))], dtype=int)
    truth_mean_cov = (2.0 / max(repeat_levels)) * truth_cov + 2e-3 * np.eye(g4.OBS_DIM)
    neg_smin, _, _ = g4._design_metrics(
        truth_ea[sel], 3.0 * truth_ea[sel], truth_mean_cov
    )

    gate = {
        "all_costs_charged": all(
            row.total_calls == scout_count + locations * (3 * row.repeats - 1)
            for row in rows
        ),
        "noise_bias_declines_with_replication": bias_declines,
        "recovery_crossover_found_by_48_repeats": crossover is not None,
        "active_beats_best_baseline_by_0_05_at_crossover": active_advantage,
        "negative_control_singular": neg_smin < 1e-8,
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
        "purpose": (
            "Measure the sample-complexity boundary exposed by Gate 4a/4b. "
            "Selection algorithm, candidate pool, stochastic world and estimator "
            "family are fixed; only paid repeats increase."
        ),
        "hard_pair": {
            "regularization_C": chosen_c,
            "score_scale": scale,
            "latent_eta": natural.eta,
            "latent_cosine": natural.cosine,
        },
        "fixed_design": {
            "scout_count": scout_count,
            "selected_locations": locations,
            "warm_start": warm_start,
            "repeat_levels": list(repeat_levels),
        },
        "rows": [asdict(row) for row in rows],
        "noise_bias_aa_per_location": {
            str(r): final_noise_bias[str(r)]["noise_bias_aa_per_location"]
            for r in repeat_levels
        },
        "active_recovery_crossover_repeats": crossover,
        "active_recovery_crossover_calls": (
            None if crossover is None else scout_count + locations * (3 * crossover - 1)
        ),
        "best_baseline_at_crossover": None if best_baseline is None else asdict(best_baseline),
        "negative_control_smin": neg_smin,
        "gate": gate,
    }

    print("\nGATE 4c — STOCHASTIC ATTRIBUTION SAMPLE-COMPLEXITY SWEEP")
    print(
        f"hard pair latent eta={natural.eta:.6f} cos={natural.cosine:.6f}; "
        f"scout={scout_count}, locations={locations}"
    )
    for r in repeat_levels:
        print(f"\nrepeats={r} calls={scout_count + locations * (3*r - 1)} noise_bias={final_noise_bias[str(r)]['noise_bias_aa_per_location']:.3f}")
        for name in ("active", "uncertainty", "static_mixed", "random"):
            row = by_repeat[r][name]
            print(
                f"  {name:12s} smin={row.truth_smin:.4f} eta={row.truth_eta:.4f} "
                f"recovery={row.recovery:.3f} (A={row.recovery_a:.3f}, B={row.recovery_b:.3f}) "
                f"reliability={row.reliability:.3f}"
            )
    print(f"\nactive crossover repeats={crossover}")
    if best_baseline is not None:
        print(
            f"best baseline at crossover={best_baseline.strategy} recovery={best_baseline.recovery:.3f}"
        )
    print(f"negative control smin={neg_smin:.3e}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("gate4c_result.json"))
    args = p.parse_args()
    result = run(args.output)
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 4c did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
