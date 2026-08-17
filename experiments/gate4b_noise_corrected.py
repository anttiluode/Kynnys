from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from experiments import gate4_stochastic_structured as g4
from experiments import gate3_blackbox_probe as g3


def _project_psd_2x2(g11: float, g12: float, g22: float) -> tuple[float, float, float]:
    g = np.asarray([[g11, g12], [g12, g22]], dtype=float)
    vals, vecs = np.linalg.eigh(g)
    vals = np.maximum(vals, 0.0)
    gp = (vecs * vals) @ vecs.T
    return float(gp[0, 0]), float(gp[0, 1]), float(gp[1, 1])


def _noise_debiased_gram(
    ea: np.ndarray,
    eb: np.ndarray,
    sample_cov: np.ndarray,
    repeats: int,
) -> tuple[float, float, float, np.ndarray, dict]:
    # Mean-effect covariance from A-base and B-base with a shared baseline mean.
    caa = (2.0 / repeats) * sample_cov
    cbb = (2.0 / repeats) * sample_cov
    cab = (1.0 / repeats) * sample_cov
    metric_cov = caa + 2e-3 * np.eye(g4.OBS_DIM)
    w = g4._inv_sqrt(metric_cov)

    wa = ea @ w.T
    wb = eb @ w.T
    n = len(ea)
    obs11 = float(np.sum(wa * wa))
    obs12 = float(np.sum(wa * wb))
    obs22 = float(np.sum(wb * wb))

    baa = float(np.trace(w @ caa @ w.T))
    bab = float(np.trace(w @ cab @ w.T))
    bbb = float(np.trace(w @ cbb @ w.T))

    g11, g12, g22 = _project_psd_2x2(
        obs11 - n * baa,
        obs12 - n * bab,
        obs22 - n * bbb,
    )
    return g11, g12, g22, w, {
        "noise_bias_aa_per_location": baa,
        "noise_bias_ab_per_location": bab,
        "noise_bias_bb_per_location": bbb,
        "observed_gram": [obs11, obs12, obs22],
        "debiased_gram": [g11, g12, g22],
    }


def _effect_cov_parts(
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]], repeats: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ea, eb, _ = g4._effect_and_cov(batches, repeats=repeats)
    residuals = []
    for base, a, b in batches:
        for x in (base, a, b):
            residuals.append(x - np.mean(x, axis=0, keepdims=True))
    r = np.concatenate(residuals, axis=0)
    sample_cov = np.cov(r, rowvar=False) if len(r) > 1 else np.eye(g4.OBS_DIM)
    sample_cov = np.asarray(sample_cov, dtype=float) + 1e-4 * np.eye(g4.OBS_DIM)
    return ea, eb, sample_cov


def _candidate_min_eigen(
    g11: float,
    g12: float,
    g22: float,
    pa: np.ndarray,
    pb: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    wa = pa @ w.T
    wb = pb @ w.T
    aa = g11 + np.sum(wa * wa, axis=1)
    ab = g12 + np.sum(wa * wb, axis=1)
    bb = g22 + np.sum(wb * wb, axis=1)
    disc = np.sqrt(np.maximum(0.0, (aa - bb) ** 2 + 4.0 * ab * ab))
    return 0.5 * (aa + bb - disc)


def _noise_corrected_active(
    texts,
    oracle: g4.ChargedOracle,
    *,
    locations: int,
    warm_start: int,
    seed: int,
):
    x = g4._selector_features(texts, oracle.scout)
    rng = random.Random(seed)
    selected: list[int] = []
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    diagnostics: list[dict] = []

    def take(i: int) -> None:
        if i in selected:
            return
        selected.append(int(i))
        batches.append(oracle.batch(int(i)))

    for i in rng.sample(range(len(texts)), warm_start):
        take(i)

    while len(selected) < locations:
        ea, eb, sample_cov = _effect_cov_parts(batches, oracle.repeats)
        g11, g12, g22, w, diag = _noise_debiased_gram(
            ea, eb, sample_cov, oracle.repeats
        )
        diag["locations"] = len(selected)
        diagnostics.append(diag)

        y = np.hstack([ea, eb])
        idx = np.asarray(selected, dtype=int)
        score_members = []
        crng = np.random.default_rng(500_000 + seed + len(selected))
        for _ in range(14):
            boot = crng.integers(0, len(selected), size=len(selected))
            model = Ridge(alpha=12.0, solver="lsqr")
            model.fit(x[idx[boot]], y[boot])
            pred = np.asarray(model.predict(x), dtype=float)
            pa, pb = pred[:, : g4.OBS_DIM], pred[:, g4.OBS_DIM :]
            score_members.append(
                _candidate_min_eigen(g11, g12, g22, pa, pb, w)
            )
        scores = np.stack(score_members, axis=0)
        # Conservative acquisition: use a lower confidence bound. In 4a an
        # optimistic uncertainty bonus selected noise. Here uncertain predicted
        # gains are penalized rather than rewarded.
        score = np.mean(scores, axis=0) - 0.50 * np.std(scores, axis=0)
        score[idx] = -np.inf
        take(int(np.argmax(score)))

    return selected, batches, diagnostics


def _denoised_templates(texts, scout, selected, batches, repeats):
    x = g4._selector_features(texts, scout)
    ea, eb, sample_cov = _effect_cov_parts(batches, repeats)
    y = np.hstack([ea, eb])
    idx = np.asarray(selected, dtype=int)

    # High regularization intentionally treats the raw per-location means as
    # noisy labels. This is an errors-in-variables mitigation, not an attempt to
    # interpolate the 48 measured points.
    model = Ridge(alpha=18.0, solver="lsqr")
    model.fit(x[idx], y)
    pred = np.asarray(model.predict(x[idx]), dtype=float)
    pa, pb = pred[:, : g4.OBS_DIM], pred[:, g4.OBS_DIM :]

    # Empirical-Bayes reliability: subtract expected mean-estimation variance
    # from the across-location target variance. Low measured SNR puts more weight
    # on the smooth surrogate and less on the raw noisy mean.
    noise_var = float(np.trace((2.0 / repeats) * sample_cov) / g4.OBS_DIM)
    observed_var = float(np.var(y, axis=0).mean())
    signal_var = max(0.0, observed_var - noise_var)
    reliability = signal_var / max(signal_var + noise_var, 1e-12)
    reliability = float(np.clip(reliability, 0.05, 0.75))
    da = reliability * ea + (1.0 - reliability) * pa
    db = reliability * eb + (1.0 - reliability) * pb
    metric_cov = (2.0 / repeats) * sample_cov + 2e-3 * np.eye(g4.OBS_DIM)
    return da, db, metric_cov, reliability, sample_cov


def _recover_with_templates(
    selected,
    template_a,
    template_b,
    metric_cov,
    truth_streams,
    *,
    repeats,
    trials,
    seed,
):
    sel = np.asarray(selected, dtype=int)
    w = g4._inv_sqrt(metric_cov)
    j = np.column_stack([
        (template_a @ w.T).reshape(-1),
        (template_b @ w.T).reshape(-1),
    ])
    pinv = np.linalg.pinv(j)
    rng = np.random.default_rng(seed)
    ca = cb = 0
    max_available = truth_streams["base"].shape[1]
    for _ in range(trials):
        start = int(rng.integers(0, max_available - repeats + 1))
        base = np.mean(truth_streams["base"][sel, start : start + repeats, :], axis=1)
        aobs = np.mean(truth_streams["A"][sel, start : start + repeats, :], axis=1)
        bobs = np.mean(truth_streams["B"][sel, start : start + repeats, :], axis=1)
        ta = pinv @ (((aobs - base) @ w.T).reshape(-1))
        tb = pinv @ (((bobs - base) @ w.T).reshape(-1))
        ca += int(ta[0] > ta[1])
        cb += int(tb[1] > tb[0])
    aa, bb = ca / trials, cb / trials
    return float(0.5 * (aa + bb)), float(aa), float(bb)


def run(output: Path) -> dict:
    scout_count = 512
    locations = 48
    repeats = 6
    warm_start = 12
    recovery_trials = 300

    train, test, model_vec, baseline, regularized, chosen_c, scale, natural = g3._fit_pipeline()
    texts_all, meta = g3._candidate_texts(train.data, test.data)
    x_all = model_vec.transform(texts_all)
    base_margin = np.asarray(baseline.decision_function(x_all), dtype=float)
    reg_margin = np.asarray(regularized.decision_function(x_all), dtype=float)
    margins = {"base": base_margin, "A": reg_margin, "B": scale * base_margin}

    rng = random.Random(4404)
    scout_global = sorted(rng.sample(range(len(texts_all)), scout_count))
    scout_texts = [texts_all[i] for i in scout_global]
    streams = g4._candidate_streams(margins, scout_global, max_repeats=8, world_seed=404)
    truth_ea, truth_eb, truth_cov = g4._truth_for_scout(
        margins, scout_global, repeats=2500, seed=8_800_000
    )
    truth_streams = g4._candidate_streams(
        margins, scout_global, max_repeats=900, world_seed=9_991
    )

    call_budget = scout_count + locations * (3 * repeats - 1)
    oracle = g4.ChargedOracle(
        streams, repeats=repeats, scout_count=scout_count, call_budget=call_budget
    )
    selected, batches, diagnostics = _noise_corrected_active(
        scout_texts,
        oracle,
        locations=locations,
        warm_start=warm_start,
        seed=4040,
    )

    da, db, metric_cov, reliability, sample_cov = _denoised_templates(
        scout_texts, oracle.scout, selected, batches, repeats
    )
    recovery, rec_a, rec_b = _recover_with_templates(
        selected,
        da,
        db,
        metric_cov,
        truth_streams,
        repeats=repeats,
        trials=recovery_trials,
        seed=919,
    )

    sel = np.asarray(selected, dtype=int)
    truth_mean_cov = (2.0 / repeats) * truth_cov + 2e-3 * np.eye(g4.OBS_DIM)
    truth_smin, _, truth_cond = g4._design_metrics(
        truth_ea[sel], truth_eb[sel], truth_mean_cov
    )
    eta = g4._eta(truth_ea[sel], truth_eb[sel])

    # Rebuild Gate 4a baselines in the exact same world/budget, but recovery is
    # scored with their original raw templates; they are attackers, not granted
    # the new method.
    baselines = {}
    for j, name in enumerate(("uncertainty", "static_mixed", "random")):
        bo = g4.ChargedOracle(
            streams, repeats=repeats, scout_count=scout_count, call_budget=call_budget
        )
        idx = g4._static_indices(name, bo.scout, locations, 990 + j)
        batches_b = g4._collect_static(idx, bo)
        baselines[name] = g4._strategy_result(
            name,
            idx,
            batches_b,
            bo,
            truth_ea,
            truth_eb,
            truth_cov,
            truth_streams,
            scout_global,
            repeats=repeats,
            recovery_trials=recovery_trials,
            recovery_seed=222 + j,
        )

    rrng = random.Random(771)
    random_smin = []
    for _ in range(120):
        idx = rrng.sample(range(scout_count), locations)
        smin, _, _ = g4._design_metrics(
            truth_ea[np.asarray(idx)], truth_eb[np.asarray(idx)], truth_mean_cov
        )
        random_smin.append(smin)

    neg_smin, _, _ = g4._design_metrics(
        truth_ea[sel], 3.0 * truth_ea[sel], truth_mean_cov
    )

    gate = {
        "same_charged_budget_as_gate4a": oracle.calls == 1328,
        "active_truth_smin_beats_random_p90": truth_smin > g4._p90(random_smin),
        "active_truth_smin_beats_uncertainty": truth_smin > baselines["uncertainty"].truth_smin,
        "active_truth_smin_beats_static_mixed": truth_smin > baselines["static_mixed"].truth_smin,
        "recovery_at_least_0_75": recovery >= 0.75,
        "recovery_beats_uncertainty": recovery > baselines["uncertainty"].recovery_accuracy,
        "noise_bias_material": diagnostics[-1]["noise_bias_aa_per_location"] > 1.0,
        "negative_control_singular": neg_smin < 1e-8,
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
        "status": "noise-corrected follow-up to failed Gate 4a",
        "budget": {
            "scout": scout_count,
            "locations": locations,
            "repeats": repeats,
            "total_calls": call_budget,
        },
        "hard_pair": {
            "regularization_C": chosen_c,
            "score_scale": scale,
            "latent_eta": natural.eta,
        },
        "active": {
            "truth_smin": truth_smin,
            "truth_condition": truth_cond,
            "truth_eta": eta,
            "recovery": recovery,
            "recovery_a": rec_a,
            "recovery_b": rec_b,
            "template_reliability": reliability,
            "calls": oracle.calls,
        },
        "baselines": {k: asdict(v) for k, v in baselines.items()},
        "random_truth_smin": {
            "median": median(random_smin),
            "p90": g4._p90(random_smin),
            "max": max(random_smin),
        },
        "noise_diagnostics_final": diagnostics[-1],
        "estimated_sample_covariance": sample_cov.tolist(),
        "truth_sample_covariance": truth_cov.tolist(),
        "negative_control_smin": neg_smin,
        "selected_global": [int(scout_global[i]) for i in selected],
        "gate": gate,
    }

    print("\nGATE 4b — NOISE-CORRECTED STOCHASTIC ATTRIBUTION")
    print(f"same charged budget: {call_budget} calls")
    print(
        f"active truth_smin={truth_smin:.4f} eta={eta:.4f} cond={truth_cond:.1f} "
        f"recovery={recovery:.3f} (A={rec_a:.3f}, B={rec_b:.3f}) reliability={reliability:.3f}"
    )
    for name, row in baselines.items():
        print(
            f"{name:12s} truth_smin={row.truth_smin:.4f} recovery={row.recovery_accuracy:.3f}"
        )
    print(
        f"random smin median={median(random_smin):.4f} p90={g4._p90(random_smin):.4f} max={max(random_smin):.4f}"
    )
    print(f"final noise bias per location aa={diagnostics[-1]['noise_bias_aa_per_location']:.3f} ab={diagnostics[-1]['noise_bias_ab_per_location']:.3f}")
    print(f"negative control smin={neg_smin:.3e}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("gate4b_result.json"))
    args = p.parse_args()
    result = run(args.output)
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 4b did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
