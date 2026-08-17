from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

from experiments import gate3_blackbox_probe as g3


OBS_DIM = 5
VARIANTS = ("base", "A", "B")


@dataclass
class Strategy:
    name: str
    selected_local: list[int]
    selected_global: list[int]
    calls: int
    observed_smin: float
    truth_smin: float
    truth_condition: float
    eta_truth: float
    recovery_accuracy: float
    recovery_a: float
    recovery_b: float
    covariance_rel_error: float


def _p90(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return math.nan
    return xs[int(round(0.9 * (len(xs) - 1)))]


def _structured_batch(margins: np.ndarray, repeats: int, rng: np.random.Generator) -> np.ndarray:
    """Sample an LLM/tool-shaped structured output from scalar latent margins.

    Output vector fields:
      0: answer == sci.space
      1: tool == search
      2: confidence == low
      3: confidence == high
      4: hedge == true

    The fields share latent noise, so covariance is real and non-diagonal.
    """

    m = np.asarray(margins, dtype=float).reshape(-1, 1)
    n = len(margins)
    latent = m + rng.normal(0.0, 0.72, size=(n, repeats))
    absz = np.abs(latent)

    # Sample the answer from the noisy latent rather than thresholding it. This
    # adds run-to-run variability analogous to stochastic decoding.
    p_space = 1.0 / (1.0 + np.exp(-latent))
    answer = rng.random((n, repeats)) < p_space

    # Low confidence and tool-use are correlated through the same latent score.
    conf_jitter = rng.normal(0.0, 0.22, size=(n, repeats))
    conf_signal = absz + conf_jitter
    low = conf_signal < 0.72
    high = conf_signal > 1.55

    p_search = 1.0 / (1.0 + np.exp(2.25 * (absz - 0.78)))
    # A low-confidence output is more likely to route to a tool.
    p_search = np.clip(p_search + 0.16 * low, 0.0, 1.0)
    search = rng.random((n, repeats)) < p_search

    p_hedge = np.clip(0.10 + 0.46 * low + 0.22 * search, 0.0, 0.95)
    hedge = rng.random((n, repeats)) < p_hedge

    return np.stack([answer, search, low, high, hedge], axis=2).astype(float)


def _stream_seed(global_index: int, variant_code: int, world_seed: int) -> int:
    # Stable independent stream per candidate/variant.
    x = (world_seed * 1_000_003 + global_index * 97_409 + variant_code * 65_537) & 0xFFFFFFFF
    return int(x)


def _candidate_streams(
    margins: dict[str, np.ndarray],
    scout_global: Sequence[int],
    *,
    max_repeats: int,
    world_seed: int,
) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
    for local, global_idx in enumerate(scout_global):
        for code, variant in enumerate(VARIANTS):
            rng = np.random.default_rng(_stream_seed(int(global_idx), code, world_seed))
            arr = _structured_batch(
                np.asarray([margins[variant][global_idx]], dtype=float), max_repeats, rng
            )[0]
            out[variant].append(arr)
    return {k: np.stack(v, axis=0) for k, v in out.items()}


class ChargedOracle:
    """Every black-box structured sample consumes one call.

    Scout baseline sample zero is charged up front for every scout candidate.
    If a candidate is later selected, that already-paid sample is reused as the
    first baseline repeat; only the remaining baseline and A/B repeats cost more.
    """

    def __init__(
        self,
        streams: dict[str, np.ndarray],
        *,
        repeats: int,
        scout_count: int,
        call_budget: int,
    ) -> None:
        self.streams = streams
        self.repeats = int(repeats)
        self.call_budget = int(call_budget)
        self.calls = int(scout_count)
        self.scout = np.asarray(streams["base"][:, 0, :], dtype=float)
        self._seen: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if self.calls > self.call_budget:
            raise RuntimeError("scout stage exceeds call budget")

    def batch(self, local_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_idx = int(local_idx)
        if local_idx in self._seen:
            return self._seen[local_idx]
        added = (self.repeats - 1) + self.repeats + self.repeats
        if self.calls + added > self.call_budget:
            raise RuntimeError("structured-call budget exceeded")
        self.calls += added
        base = np.asarray(self.streams["base"][local_idx, : self.repeats, :], dtype=float)
        a = np.asarray(self.streams["A"][local_idx, : self.repeats, :], dtype=float)
        b = np.asarray(self.streams["B"][local_idx, : self.repeats, :], dtype=float)
        self._seen[local_idx] = (base, a, b)
        return base, a, b


def _effect_and_cov(
    batches: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effects_a = []
    effects_b = []
    residuals = []
    for base, a, b in batches:
        effects_a.append(np.mean(a, axis=0) - np.mean(base, axis=0))
        effects_b.append(np.mean(b, axis=0) - np.mean(base, axis=0))
        for x in (base, a, b):
            residuals.append(x - np.mean(x, axis=0, keepdims=True))
    ea = np.asarray(effects_a, dtype=float)
    eb = np.asarray(effects_b, dtype=float)
    r = np.concatenate(residuals, axis=0)
    if len(r) <= 1:
        cov = np.eye(OBS_DIM)
    else:
        cov = np.cov(r, rowvar=False)
    # Covariance of an A-base or B-base difference of means. The base is shared
    # in the empirical effect estimate; for design weighting the conservative
    # 2*Sigma/r approximation is sufficient.
    cov_mean = (2.0 / repeats) * cov + 2e-3 * np.eye(OBS_DIM)
    return ea, eb, cov_mean


def _inv_sqrt(cov: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.maximum(vals, 1e-8)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _design_metrics(ea: np.ndarray, eb: np.ndarray, cov_mean: np.ndarray) -> tuple[float, float, float]:
    w = _inv_sqrt(cov_mean)
    wa = np.asarray(ea, dtype=float) @ w.T
    wb = np.asarray(eb, dtype=float) @ w.T
    j = np.column_stack([wa.reshape(-1), wb.reshape(-1)])
    s = np.linalg.svd(j, full_matrices=False, compute_uv=False)
    smin, smax = float(s[-1]), float(s[0])
    cond = math.inf if smin <= 1e-15 else smax / smin
    return smin, smax, cond


def _eta(ea: np.ndarray, eb: np.ndarray) -> float:
    a = np.asarray(ea, dtype=float).reshape(-1)
    b = np.asarray(eb, dtype=float).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-18 or nb <= 1e-18:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.sqrt(max(0.0, 1.0 - cos * cos)))


def _selector_features(texts: Sequence[str], scout: np.ndarray):
    hv = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        ngram_range=(1, 2),
    )
    x = hv.transform(texts)
    lengths = np.asarray([max(1, len(t.split())) for t in texts], dtype=float)
    extra = sparse.csr_matrix(
        np.column_stack([scout, np.log1p(lengths)])
    )
    return sparse.hstack([x, extra], format="csr")


def _candidate_smin_from_vectors(
    g11: float,
    g12: float,
    g22: float,
    pa: np.ndarray,
    pb: np.ndarray,
    inv_sqrt: np.ndarray,
) -> np.ndarray:
    wa = pa @ inv_sqrt.T
    wb = pb @ inv_sqrt.T
    aa = g11 + np.sum(wa * wa, axis=1)
    ab = g12 + np.sum(wa * wb, axis=1)
    bb = g22 + np.sum(wb * wb, axis=1)
    disc = np.sqrt(np.maximum(0.0, (aa - bb) ** 2 + 4.0 * ab * ab))
    lam = 0.5 * (aa + bb - disc)
    return np.sqrt(np.maximum(0.0, lam))


def _fisher_active(
    texts: Sequence[str],
    oracle: ChargedOracle,
    *,
    locations: int,
    warm_start: int,
    seed: int,
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    x = _selector_features(texts, oracle.scout)
    rng = random.Random(seed)
    selected: list[int] = []
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def take(i: int) -> None:
        if i in selected:
            return
        selected.append(int(i))
        batches.append(oracle.batch(int(i)))

    for i in rng.sample(range(len(texts)), warm_start):
        take(i)

    while len(selected) < locations:
        ea, eb, cov_mean = _effect_and_cov(batches, repeats=oracle.repeats)
        w = _inv_sqrt(cov_mean)
        wa = ea @ w.T
        wb = eb @ w.T
        g11 = float(np.sum(wa * wa))
        g12 = float(np.sum(wa * wb))
        g22 = float(np.sum(wb * wb))
        y = np.hstack([ea, eb])
        idx = np.asarray(selected, dtype=int)

        committee_scores = []
        crng = np.random.default_rng(700_000 + seed + len(selected))
        for _ in range(10):
            boot = crng.integers(0, len(selected), size=len(selected))
            model = Ridge(alpha=2.5, solver="lsqr")
            model.fit(x[idx[boot]], y[boot])
            pred = np.asarray(model.predict(x), dtype=float)
            pa, pb = pred[:, :OBS_DIM], pred[:, OBS_DIM:]
            committee_scores.append(
                _candidate_smin_from_vectors(g11, g12, g22, pa, pb, w)
            )
        scores = np.stack(committee_scores, axis=0)
        score = np.mean(scores, axis=0) + 0.55 * np.std(scores, axis=0)
        score[idx] = -np.inf
        take(int(np.argmax(score)))

    return selected, batches


def _static_indices(
    name: str,
    scout: np.ndarray,
    locations: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    n = len(scout)
    if name == "random":
        return rng.sample(range(n), locations)
    if name == "uncertainty":
        # One paid scout sample per candidate. Prefer a search/low-confidence/
        # hedged decision; break ties deterministically with the answer bit.
        score = 2.0 * scout[:, 2] + 1.5 * scout[:, 1] + scout[:, 4] - 0.1 * scout[:, 3]
        jitter = np.asarray([rng.random() for _ in range(n)]) * 1e-6
        return list(map(int, np.argsort(-(score + jitter))[:locations]))
    if name == "static_mixed":
        uncertain = _static_indices("uncertainty", scout, locations // 2, seed + 1)
        used = set(uncertain)
        rest = [i for i in range(n) if i not in used]
        return uncertain + rng.sample(rest, locations - len(uncertain))
    raise ValueError(name)


def _collect_static(
    indices: Sequence[int], oracle: ChargedOracle
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [oracle.batch(int(i)) for i in indices]


def _truth_for_scout(
    margins: dict[str, np.ndarray],
    scout_global: Sequence[int],
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.asarray(scout_global, dtype=int)
    samples = {}
    for code, variant in enumerate(VARIANTS):
        rng = np.random.default_rng(seed + 10_000 * code)
        samples[variant] = _structured_batch(margins[variant][idx], repeats, rng)
    means = {k: np.mean(v, axis=1) for k, v in samples.items()}
    ea = means["A"] - means["base"]
    eb = means["B"] - means["base"]
    residuals = []
    for variant in VARIANTS:
        x = samples[variant]
        residuals.append((x - np.mean(x, axis=1, keepdims=True)).reshape(-1, OBS_DIM))
    cov = np.cov(np.concatenate(residuals, axis=0), rowvar=False)
    return ea, eb, cov


def _cov_rel_error(estimated_mean_cov: np.ndarray, truth_sample_cov: np.ndarray, repeats: int) -> float:
    target = (2.0 / repeats) * truth_sample_cov + 2e-3 * np.eye(OBS_DIM)
    return float(np.linalg.norm(estimated_mean_cov - target) / max(np.linalg.norm(target), 1e-12))


def _recover(
    selected: Sequence[int],
    train_ea: np.ndarray,
    train_eb: np.ndarray,
    train_cov_mean: np.ndarray,
    truth_streams: dict[str, np.ndarray],
    *,
    repeats: int,
    trials: int,
    seed: int,
) -> tuple[float, float, float]:
    """Fresh stochastic A-only/B-only attribution trials.

    The attribution design comes only from budgeted training observations. Fresh
    held-out stochastic streams score whether that design recovers the active
    cause under the same repeat count.
    """

    sel = np.asarray(selected, dtype=int)
    w = _inv_sqrt(train_cov_mean)
    wa = train_ea @ w.T
    wb = train_eb @ w.T
    j = np.column_stack([wa.reshape(-1), wb.reshape(-1)])
    pinv = np.linalg.pinv(j)
    rng = np.random.default_rng(seed)

    correct_a = 0
    correct_b = 0
    max_available = truth_streams["base"].shape[1]
    for t in range(trials):
        # Draw disjoint-ish repeat windows with replacement over a long scoring
        # stream. These calls are benchmark scoring only, never visible to the
        # selector.
        start = int(rng.integers(0, max_available - repeats + 1))
        base = np.mean(truth_streams["base"][sel, start : start + repeats, :], axis=1)
        aobs = np.mean(truth_streams["A"][sel, start : start + repeats, :], axis=1)
        bobs = np.mean(truth_streams["B"][sel, start : start + repeats, :], axis=1)
        ya = (aobs - base) @ w.T
        yb = (bobs - base) @ w.T
        theta_a = pinv @ ya.reshape(-1)
        theta_b = pinv @ yb.reshape(-1)
        correct_a += int(theta_a[0] > theta_a[1])
        correct_b += int(theta_b[1] > theta_b[0])
    acc_a = correct_a / trials
    acc_b = correct_b / trials
    return float(0.5 * (acc_a + acc_b)), float(acc_a), float(acc_b)


def _strategy_result(
    name: str,
    selected: Sequence[int],
    batches: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    oracle: ChargedOracle,
    truth_ea: np.ndarray,
    truth_eb: np.ndarray,
    truth_cov: np.ndarray,
    truth_streams: dict[str, np.ndarray],
    scout_global: Sequence[int],
    *,
    repeats: int,
    recovery_trials: int,
    recovery_seed: int,
) -> Strategy:
    observed_ea, observed_eb, observed_cov = _effect_and_cov(batches, repeats=repeats)
    obs_smin, _, _ = _design_metrics(observed_ea, observed_eb, observed_cov)
    sel = np.asarray(selected, dtype=int)
    truth_mean_cov = (2.0 / repeats) * truth_cov + 2e-3 * np.eye(OBS_DIM)
    true_smin, _, true_cond = _design_metrics(truth_ea[sel], truth_eb[sel], truth_mean_cov)
    rec, rec_a, rec_b = _recover(
        selected,
        observed_ea,
        observed_eb,
        observed_cov,
        truth_streams,
        repeats=repeats,
        trials=recovery_trials,
        seed=recovery_seed,
    )
    return Strategy(
        name=name,
        selected_local=list(map(int, selected)),
        selected_global=[int(scout_global[i]) for i in selected],
        calls=oracle.calls,
        observed_smin=obs_smin,
        truth_smin=true_smin,
        truth_condition=true_cond,
        eta_truth=_eta(truth_ea[sel], truth_eb[sel]),
        recovery_accuracy=rec,
        recovery_a=rec_a,
        recovery_b=rec_b,
        covariance_rel_error=_cov_rel_error(observed_cov, truth_cov, repeats),
    )


def run(
    output: Path,
    *,
    scout_count: int = 512,
    locations: int = 48,
    repeats: int = 6,
    warm_start: int = 12,
    recovery_trials: int = 250,
    random_trials: int = 120,
) -> dict:
    train, test, model_vec, baseline, regularized, chosen_c, scale, natural = g3._fit_pipeline()
    texts_all, meta = g3._candidate_texts(train.data, test.data)
    x_all = model_vec.transform(texts_all)
    base_margin = np.asarray(baseline.decision_function(x_all), dtype=float)
    reg_margin = np.asarray(regularized.decision_function(x_all), dtype=float)
    margins = {"base": base_margin, "A": reg_margin, "B": scale * base_margin}

    rng = random.Random(4404)
    scout_global = sorted(rng.sample(range(len(texts_all)), scout_count))
    scout_texts = [texts_all[i] for i in scout_global]
    scout_kinds = [meta["kinds"][i] for i in scout_global]

    streams = _candidate_streams(
        margins,
        scout_global,
        max_repeats=max(repeats, 8),
        world_seed=404,
    )
    # Hidden high-precision scoring distribution. It is never exposed to a
    # strategy and does not influence probe selection.
    truth_ea, truth_eb, truth_cov = _truth_for_scout(
        margins,
        scout_global,
        repeats=2500,
        seed=8_800_000,
    )
    truth_streams = _candidate_streams(
        margins,
        scout_global,
        max_repeats=900,
        world_seed=9_991,
    )

    calls_per_selected = 3 * repeats - 1
    call_budget = scout_count + locations * calls_per_selected

    # Active strategy.
    active_oracle = ChargedOracle(
        streams, repeats=repeats, scout_count=scout_count, call_budget=call_budget
    )
    active_idx, active_batches = _fisher_active(
        scout_texts,
        active_oracle,
        locations=locations,
        warm_start=warm_start,
        seed=4040,
    )
    active = _strategy_result(
        "active_fisher",
        active_idx,
        active_batches,
        active_oracle,
        truth_ea,
        truth_eb,
        truth_cov,
        truth_streams,
        scout_global,
        repeats=repeats,
        recovery_trials=recovery_trials,
        recovery_seed=121,
    )

    static_results: dict[str, Strategy] = {}
    for j, name in enumerate(("uncertainty", "static_mixed", "random")):
        oracle = ChargedOracle(
            streams, repeats=repeats, scout_count=scout_count, call_budget=call_budget
        )
        idx = _static_indices(name, oracle.scout, locations, 990 + j)
        batches = _collect_static(idx, oracle)
        static_results[name] = _strategy_result(
            name,
            idx,
            batches,
            oracle,
            truth_ea,
            truth_eb,
            truth_cov,
            truth_streams,
            scout_global,
            repeats=repeats,
            recovery_trials=recovery_trials,
            recovery_seed=222 + j,
        )

    # Random distribution scored from hidden truth geometry. All sets have the
    # same location count and would cost the same call budget.
    rrng = random.Random(771)
    random_truth_smin = []
    for _ in range(random_trials):
        idx = rrng.sample(range(scout_count), locations)
        truth_mean_cov = (2.0 / repeats) * truth_cov + 2e-3 * np.eye(OBS_DIM)
        smin, _, _ = _design_metrics(
            truth_ea[np.asarray(idx)], truth_eb[np.asarray(idx)], truth_mean_cov
        )
        random_truth_smin.append(smin)

    # Exact non-identifiability control in the same measured noise geometry.
    sel = np.asarray(active_idx, dtype=int)
    truth_mean_cov = (2.0 / repeats) * truth_cov + 2e-3 * np.eye(OBS_DIM)
    neg_smin, _, _ = _design_metrics(truth_ea[sel], 3.0 * truth_ea[sel], truth_mean_cov)

    gate = {
        "structured_output_is_stochastic": bool(np.trace(truth_cov) > 0.05),
        "all_selector_calls_charged": active.calls == call_budget,
        "active_truth_smin_absolute": active.truth_smin >= 1.00,
        "active_beats_uncertainty": active.truth_smin > static_results["uncertainty"].truth_smin,
        "active_beats_static_mixed": active.truth_smin > static_results["static_mixed"].truth_smin,
        "active_beats_random_p90": active.truth_smin > _p90(random_truth_smin),
        "active_recovery_at_least_0_75": active.recovery_accuracy >= 0.75,
        "active_recovery_beats_uncertainty": active.recovery_accuracy > static_results["uncertainty"].recovery_accuracy,
        "negative_control_singular": neg_smin < 1e-8,
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
        "definition": {
            "observable": [
                "answer_is_space",
                "tool_is_search",
                "confidence_low",
                "confidence_high",
                "hedge",
            ],
            "noise_model": "shared latent stochastic decoding plus correlated confidence/tool/hedge decisions",
            "information_metric": "smallest singular value after whitening by measured structured-output covariance",
            "all_policy_calls_charged": True,
            "scoring_streams_visible_to_selector": False,
        },
        "dataset": {
            "name": "20 Newsgroups",
            "categories": list(g3.CATEGORIES),
            "candidate_pool": len(texts_all),
            "scout_count": scout_count,
            "scout_kinds": dict(Counter(scout_kinds)),
        },
        "hard_pair": {
            "regularization_C": chosen_c,
            "score_scale": scale,
            "scalar_margin_natural_eta": natural.eta,
            "scalar_margin_natural_cosine": natural.cosine,
        },
        "budget": {
            "scout_calls": scout_count,
            "selected_locations": locations,
            "repeats_per_variant": repeats,
            "incremental_calls_per_selected_location": calls_per_selected,
            "total_calls": call_budget,
            "warm_start_locations": warm_start,
        },
        "truth_noise_covariance": truth_cov.tolist(),
        "strategies": {
            "active": asdict(active),
            **{k: asdict(v) for k, v in static_results.items()},
            "random_distribution": {
                "trials": random_trials,
                "truth_smin_median": median(random_truth_smin),
                "truth_smin_p90": _p90(random_truth_smin),
                "truth_smin_max": max(random_truth_smin),
            },
        },
        "negative_control": {"truth_smin": neg_smin},
        "active_kinds": dict(Counter(scout_kinds[i] for i in active_idx)),
        "active_examples": [scout_texts[i][:160] for i in active_idx[:12]],
        "gate": gate,
    }

    print("\nGATE 4 — STOCHASTIC STRUCTURED-OUTPUT ATTRIBUTION")
    print(
        f"pool={len(texts_all)} scout={scout_count} locations={locations} repeats={repeats} "
        f"charged calls/strategy={call_budget}"
    )
    print(
        f"latent hard pair: eta={natural.eta:.6f} cos={natural.cosine:.6f}; "
        f"C 1.0->{chosen_c}, scale={scale:.6f}"
    )
    print(f"truth structured covariance trace={np.trace(truth_cov):.4f}")
    for row in [active, static_results["uncertainty"], static_results["static_mixed"], static_results["random"]]:
        print(
            f"{row.name:14s} truth_smin={row.truth_smin:.4f} observed_smin={row.observed_smin:.4f} "
            f"eta={row.eta_truth:.4f} recovery={row.recovery_accuracy:.3f} "
            f"cov_err={row.covariance_rel_error:.3f} calls={row.calls}"
        )
    print(
        f"random truth smin median={median(random_truth_smin):.4f} "
        f"p90={_p90(random_truth_smin):.4f} max={max(random_truth_smin):.4f}"
    )
    print(f"negative scalar-copy smin={neg_smin:.3e}")
    print(f"active kinds={result['active_kinds']}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("gate4_result.json"))
    p.add_argument("--scout-count", type=int, default=512)
    p.add_argument("--locations", type=int, default=48)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--warm-start", type=int, default=12)
    p.add_argument("--recovery-trials", type=int, default=250)
    p.add_argument("--random-trials", type=int, default=120)
    args = p.parse_args()
    result = run(
        args.output,
        scout_count=args.scout_count,
        locations=args.locations,
        repeats=args.repeats,
        warm_start=args.warm_start,
        recovery_trials=args.recovery_trials,
        random_trials=args.random_trials,
    )
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 4 did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
