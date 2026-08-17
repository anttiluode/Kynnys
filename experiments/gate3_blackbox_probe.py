from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from kynnys import audit_effects


CATEGORIES = ("rec.autos", "sci.space")
REGULARIZATION_GRID = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1)


@dataclass
class Snapshot:
    eta: float
    cosine: float
    sensitivity_a: float
    sensitivity_b: float


@dataclass
class StrategyResult:
    name: str
    indices: list[int]
    eta: float
    cosine: float
    sensitivity_a: float
    sensitivity_b: float
    queries: int


def _norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return 0.0 if den == 0.0 else float(np.dot(aa, bb) / den)


def _snapshot(a: Sequence[float], b: Sequence[float]) -> Snapshot:
    report = {
        r.name: r
        for r in audit_effects({"A": list(map(float, a)), "B": list(map(float, b))})
    }
    eta = min(report["A"].identifiable_fraction, report["B"].identifiable_fraction)
    return Snapshot(
        eta=float(eta),
        cosine=float(max(report["A"].alias_cosine, report["B"].alias_cosine)),
        sensitivity_a=float(report["A"].sensitivity),
        sensitivity_b=float(report["B"].sensitivity),
    )


def _p90(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return math.nan
    return xs[int(round(0.9 * (len(xs) - 1)))]


def _fit_pipeline():
    train = fetch_20newsgroups(
        subset="train",
        categories=list(CATEGORIES),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=7,
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=list(CATEGORIES),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=11,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train.data)
    x_test = vectorizer.transform(test.data)
    y_train = np.asarray(train.target, dtype=int)

    baseline = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear",
        random_state=7,
    )
    baseline.fit(x_train, y_train)
    base_test = np.asarray(baseline.decision_function(x_test), dtype=float)

    # Fixed audit slice used only to construct the deliberately hard pair.
    rng = np.random.default_rng(20260817)
    perm = rng.permutation(x_test.shape[0])
    audit_idx = perm[:320]

    baseline_norm = max(_norm(base_test[audit_idx]), 1e-12)
    candidates = []
    for c in REGULARIZATION_GRID:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear",
            random_state=7,
        )
        model.fit(x_train, y_train)
        margin = np.asarray(model.decision_function(x_test[audit_idx]), dtype=float)
        effect = margin - base_test[audit_idx]
        rel = _norm(effect) / baseline_norm
        similarity = abs(_cosine(effect, base_test[audit_idx]))
        if rel >= 0.004:
            candidates.append((similarity, rel, c, model, effect))
    if not candidates:
        raise RuntimeError("no non-trivial regularization candidate")
    _, _, chosen_c, regularized, effect_audit = max(
        candidates, key=lambda row: (row[0], row[1])
    )

    denom = float(np.dot(base_test[audit_idx], base_test[audit_idx]))
    alpha = float(np.dot(effect_audit, base_test[audit_idx]) / denom)
    temperature_scale = 1.0 + alpha
    if not (0.05 < temperature_scale < 2.0):
        raise RuntimeError(f"pathological temperature scale {temperature_scale}")

    reg_test = np.asarray(regularized.decision_function(x_test), dtype=float)
    temp_test = temperature_scale * base_test
    natural = _snapshot(reg_test[audit_idx] - base_test[audit_idx], temp_test[audit_idx] - base_test[audit_idx])

    return train, test, vectorizer, baseline, regularized, chosen_c, temperature_scale, natural


def _candidate_texts(train_texts: Sequence[str], test_texts: Sequence[str]) -> tuple[list[str], dict]:
    # Candidate generation is intentionally independent of the model internals.
    # It sees corpus text only. We include ordinary held-out documents plus a
    # broad synthetic probe language built from frequent corpus tokens.
    cv = CountVectorizer(
        lowercase=True,
        binary=True,
        min_df=5,
        max_features=100,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    cv.fit(train_texts)
    words = list(cv.get_feature_names_out())

    out: list[str] = []
    kinds: list[str] = []
    seen: set[str] = set()

    def add(text: str, kind: str) -> None:
        text = " ".join(text.split())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
            kinds.append(kind)

    # Ordinary traffic first.
    for text in test_texts:
        add(text, "natural")

    # Black-box generated probes: no weights, gradients or model vocabulary.
    for w in words:
        add(w, "single")
        add(f"{w} {w} {w}", "repeat")

    for i, a in enumerate(words):
        for b in words[i + 1 :]:
            add(f"{a} {b}", "pair")

    rng = random.Random(20260817)
    for _ in range(1200):
        a, b, c = rng.sample(words, 3)
        add(f"{a} {b} {c}", "triple")

    meta = {
        "frequent_words": words,
        "kinds": kinds,
        "counts": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
    }
    return out, meta


class PairedOracle:
    """Budgeted black-box access to paired variant effects.

    The selector receives only this callable, raw candidate text, and baseline
    outputs. Hidden A/B arrays are not passed into the selection function.
    """

    def __init__(self, effect_a: np.ndarray, effect_b: np.ndarray, budget: int) -> None:
        self._a = np.asarray(effect_a, dtype=float)
        self._b = np.asarray(effect_b, dtype=float)
        self.budget = int(budget)
        self.queries = 0
        self._seen: set[int] = set()

    def query(self, index: int) -> tuple[float, float]:
        index = int(index)
        if index in self._seen:
            return float(self._a[index]), float(self._b[index])
        if self.queries >= self.budget:
            raise RuntimeError("probe budget exceeded")
        self._seen.add(index)
        self.queries += 1
        return float(self._a[index]), float(self._b[index])


def _selector_features(texts: Sequence[str], baseline_margin: np.ndarray):
    # Separate black-box-side representation. It is not the classifier's
    # vectorizer and never sees its coefficients.
    hv = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        ngram_range=(1, 2),
    )
    x = hv.transform(texts)
    bm = np.asarray(baseline_margin, dtype=float)
    length = np.asarray([max(1, len(t.split())) for t in texts], dtype=float)
    extra = sparse.csr_matrix(
        np.column_stack(
            [
                bm,
                np.abs(bm),
                np.log1p(length),
            ]
        )
    )
    return sparse.hstack([x, extra], format="csr")


def _active_blackbox(
    texts: Sequence[str],
    baseline_margin: np.ndarray,
    oracle: PairedOracle,
    *,
    budget: int,
    seed_queries: int,
) -> list[int]:
    x = _selector_features(texts, baseline_margin)
    n = len(texts)
    rng = random.Random(20260817)

    selected: list[int] = []
    observed_a: list[float] = []
    observed_b: list[float] = []

    # Fixed random warm start. Any later advantage must come from black-box
    # observations, not an internal-model seed heuristic.
    for idx in rng.sample(range(n), seed_queries):
        a, b = oracle.query(idx)
        selected.append(idx)
        observed_a.append(a)
        observed_b.append(b)

    while len(selected) < budget:
        idx_arr = np.asarray(selected, dtype=int)
        y = np.column_stack([observed_a, observed_b])

        # Bootstrap committee: uncertainty comes only from black-box samples.
        preds = []
        committee_rng = np.random.default_rng(1000 + len(selected))
        for _ in range(10):
            boot = committee_rng.integers(0, len(selected), size=len(selected))
            model = Ridge(alpha=3.0, solver="lsqr")
            model.fit(x[idx_arr[boot]], y[boot])
            preds.append(np.asarray(model.predict(x), dtype=float))
        pred = np.stack(preds, axis=0)  # committee, candidate, 2

        oa = np.asarray(observed_a, dtype=float)
        ob = np.asarray(observed_b, dtype=float)
        denom = float(np.dot(ob, ob))
        beta = 0.0 if denom <= 1e-18 else float(np.dot(oa, ob) / denom)

        residual_committee = pred[:, :, 0] - beta * pred[:, :, 1]
        mean_residual = np.mean(residual_committee, axis=0)
        sd_residual = np.std(residual_committee, axis=0)
        mean_strength = np.sqrt(np.mean(pred[:, :, 0], axis=0) ** 2 + np.mean(pred[:, :, 1], axis=0) ** 2)

        # Expected independent-direction signal + an exploration bonus. A small
        # strength factor avoids spending the whole budget on numerically empty
        # points that merely have unstable ratios.
        score = (np.abs(mean_residual) + 1.25 * sd_residual) * np.minimum(
            1.0, mean_strength / 0.05
        )
        score[idx_arr] = -np.inf
        idx = int(np.argmax(score))
        a, b = oracle.query(idx)
        selected.append(idx)
        observed_a.append(a)
        observed_b.append(b)

    if oracle.queries != budget:
        raise AssertionError(f"oracle used {oracle.queries}, expected {budget}")
    return selected


def _strategy_result(name: str, indices: Sequence[int], a: np.ndarray, b: np.ndarray, queries: int) -> StrategyResult:
    idx = np.asarray(indices, dtype=int)
    snap = _snapshot(a[idx], b[idx])
    return StrategyResult(
        name=name,
        indices=[int(i) for i in idx],
        eta=snap.eta,
        cosine=snap.cosine,
        sensitivity_a=snap.sensitivity_a,
        sensitivity_b=snap.sensitivity_b,
        queries=int(queries),
    )


def run(output: Path, *, budget: int = 48, seed_queries: int = 12, random_trials: int = 400) -> dict:
    train, test, model_vec, baseline, regularized, chosen_c, scale, natural = _fit_pipeline()
    texts, meta = _candidate_texts(train.data, test.data)
    x_candidates = model_vec.transform(texts)

    # Precomputation below is benchmark instrumentation. The active selector
    # never receives these arrays; it can access A/B only through PairedOracle.
    base = np.asarray(baseline.decision_function(x_candidates), dtype=float)
    reg = np.asarray(regularized.decision_function(x_candidates), dtype=float)
    temp = scale * base
    effect_a = reg - base
    effect_b = temp - base

    oracle = PairedOracle(effect_a, effect_b, budget)
    active_idx = _active_blackbox(
        texts,
        base,
        oracle,
        budget=budget,
        seed_queries=seed_queries,
    )
    active = _strategy_result("blackbox_active", active_idx, effect_a, effect_b, oracle.queries)

    # Ordinary uncertainty sampling: spend the same counterfactual budget on
    # examples nearest the baseline decision boundary.
    uncertainty_idx = np.argsort(np.abs(base))[:budget]
    uncertainty = _strategy_result("uncertainty", uncertainty_idx, effect_a, effect_b, budget)

    # Random matched-budget distribution.
    rng = random.Random(17)
    random_etas: list[float] = []
    random_cos: list[float] = []
    random_best: StrategyResult | None = None
    population = list(range(len(texts)))
    for trial in range(random_trials):
        idx = rng.sample(population, budget)
        row = _strategy_result(f"random_{trial}", idx, effect_a, effect_b, budget)
        random_etas.append(row.eta)
        random_cos.append(row.cosine)
        if random_best is None or row.eta > random_best.eta:
            random_best = row

    # Negative control: two literally scalar-copy hidden effects evaluated on
    # the exact probes chosen by the active strategy.
    neg_a = effect_a[np.asarray(active_idx)]
    neg_b = 3.0 * neg_a
    negative = _snapshot(neg_a, neg_b)

    # A small diagnostic of what the active strategy spent budget on.
    active_kinds = [meta["kinds"][i] for i in active_idx]
    uncertainty_kinds = [meta["kinds"][int(i)] for i in uncertainty_idx]

    gate = {
        "natural_outputs_confounded": natural.eta < 0.05 and natural.cosine > 0.99,
        "active_absolute_eta": active.eta >= 0.25,
        "active_beats_uncertainty_by_0_10": active.eta >= uncertainty.eta + 0.10,
        "active_beats_random_p90": active.eta > _p90(random_etas),
        "negative_control_refuses_separation": negative.eta < 1e-9,
        "budget_respected": oracle.queries == budget,
    }
    gate["pass"] = bool(all(gate.values()))

    result = {
        "dataset": {
            "name": "20 Newsgroups",
            "categories": list(CATEGORIES),
            "train_examples": len(train.data),
            "test_examples": len(test.data),
        },
        "hard_pair": {
            "regularization_C": chosen_c,
            "temperature_scale": scale,
            "natural": asdict(natural),
        },
        "candidate_pool": {
            "size": len(texts),
            "counts": meta["counts"],
            "frequent_word_count": len(meta["frequent_words"]),
        },
        "budget": {"total": budget, "warm_start": seed_queries},
        "strategies": {
            "active": asdict(active),
            "uncertainty": asdict(uncertainty),
            "random": {
                "trials": random_trials,
                "eta_median": median(random_etas),
                "eta_p90": _p90(random_etas),
                "eta_max": max(random_etas),
                "cosine_median": median(random_cos),
                "best": asdict(random_best) if random_best else None,
            },
        },
        "selected_kinds": {
            "active": {k: active_kinds.count(k) for k in sorted(set(active_kinds))},
            "uncertainty": {k: uncertainty_kinds.count(k) for k in sorted(set(uncertainty_kinds))},
        },
        "active_examples": [texts[i][:160] for i in active_idx[:16]],
        "negative_control": asdict(negative),
        "gate": gate,
    }

    print("\nGATE 3 — BLACK-BOX ATTRIBUTION")
    print(
        f"natural confounder: eta={natural.eta:.6f} cos={natural.cosine:.6f}; "
        f"C 1.0->{chosen_c}, score scale={scale:.6f}"
    )
    print(f"candidate pool={len(texts)}, query budget={budget}, warm start={seed_queries}")
    print(
        f"ACTIVE      eta={active.eta:.6f} cos={active.cosine:.6f} "
        f"S=({active.sensitivity_a:.4f},{active.sensitivity_b:.4f})"
    )
    print(
        f"UNCERTAINTY eta={uncertainty.eta:.6f} cos={uncertainty.cosine:.6f} "
        f"S=({uncertainty.sensitivity_a:.4f},{uncertainty.sensitivity_b:.4f})"
    )
    print(
        f"RANDOM      median eta={median(random_etas):.6f} "
        f"p90={_p90(random_etas):.6f} max={max(random_etas):.6f}"
    )
    print(f"active kinds: {result['selected_kinds']['active']}")
    print(f"uncertainty kinds: {result['selected_kinds']['uncertainty']}")
    print(f"negative scalar-copy eta={negative.eta:.3e}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gate3_result.json"))
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--seed-queries", type=int, default=12)
    parser.add_argument("--random-trials", type=int, default=400)
    args = parser.parse_args()
    result = run(
        args.output,
        budget=args.budget,
        seed_queries=args.seed_queries,
        random_trials=args.random_trials,
    )
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 3 did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
