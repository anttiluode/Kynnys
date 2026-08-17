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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from kynnys import audit_effects


CATEGORIES = ("rec.autos", "sci.space")
REGULARIZATION_GRID = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1)


@dataclass
class ModelMetrics:
    accuracy: float
    log_loss: float
    coverage: float


@dataclass
class AuditSnapshot:
    regularization_sensitivity: float
    temperature_sensitivity: float
    regularization_eta: float
    temperature_eta: float
    alias_cosine: float
    confounded_regularization: bool
    confounded_temperature: bool


@dataclass
class SeparationResult:
    k: int
    suggested_eta: float
    suggested_alias_cosine: float
    random_eta_median: float
    random_eta_p90: float
    random_alias_cosine_median: float
    residual_energy_fraction: float
    suggested_indices: list[int]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def _norm(a: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _prob_from_margin(margin: np.ndarray) -> np.ndarray:
    # Stable sigmoid without scipy.
    x = np.asarray(margin, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _metrics(margin: np.ndarray, y: np.ndarray, threshold: float) -> ModelMetrics:
    prob = _prob_from_margin(margin)
    pred = (margin >= 0.0).astype(int)
    return ModelMetrics(
        accuracy=float(accuracy_score(y, pred)),
        log_loss=float(log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
        coverage=float(np.mean(np.abs(margin) >= threshold)),
    )


def _snapshot(effect_a: np.ndarray, effect_b: np.ndarray) -> AuditSnapshot:
    report = {
        row.name: row
        for row in audit_effects(
            {
                "regularization": effect_a.tolist(),
                "temperature": effect_b.tolist(),
            }
        )
    }
    a = report["regularization"]
    b = report["temperature"]
    return AuditSnapshot(
        regularization_sensitivity=a.sensitivity,
        temperature_sensitivity=b.sensitivity,
        regularization_eta=a.identifiable_fraction,
        temperature_eta=b.identifiable_fraction,
        alias_cosine=max(a.alias_cosine, b.alias_cosine),
        confounded_regularization=a.confounded,
        confounded_temperature=b.confounded,
    )


def _choose_regularization(
    x_train,
    y_train: np.ndarray,
    x_audit,
    baseline_audit: np.ndarray,
) -> tuple[float, LogisticRegression, np.ndarray, float]:
    """Find a real regularization change whose output signature is scale-like.

    Temperature scaling can only multiply the baseline margin.  We therefore
    search the regularization path for the strongest *non-trivial* effect that
    most resembles such a scaling direction.  This is deliberate confounder
    construction, not hyperparameter tuning for classification accuracy.
    """

    candidates: list[tuple[float, float, float, LogisticRegression, np.ndarray]] = []
    baseline_norm = max(_norm(baseline_audit), 1e-12)
    for c in REGULARIZATION_GRID:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear",
            random_state=7,
        )
        model.fit(x_train, y_train)
        margin = np.asarray(model.decision_function(x_audit), dtype=float)
        effect = margin - baseline_audit
        rel = _norm(effect) / baseline_norm
        similarity = abs(_cosine(effect, baseline_audit))
        # Require an effect large enough to be measurable on the audit slice.
        if rel >= 0.004:
            candidates.append((similarity, rel, c, model, effect))

    if not candidates:
        raise RuntimeError("no non-trivial regularization candidate found")

    similarity, rel, c, model, effect = max(candidates, key=lambda row: (row[0], row[1]))
    return c, model, effect, similarity


def _fit_temperature(effect_regularization: np.ndarray, baseline_margin: np.ndarray) -> float:
    """Fit the post-hoc score multiplier that best mimics the first effect."""

    denom = _dot(baseline_margin, baseline_margin)
    if denom <= 1e-18:
        raise RuntimeError("baseline margin has no energy")
    alpha = _dot(effect_regularization, baseline_margin) / denom
    scale = 1.0 + alpha
    if not (0.05 < scale < 2.0):
        raise RuntimeError(f"fitted temperature scale {scale:.4f} is pathological")
    return scale


def _separator_scores(effect_a: np.ndarray, effect_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-observation residual after the best scalar alias fit.

    If B is merely a scaled copy of A, residual is zero and no existing
    observation can separate them.  Large residual coordinates identify cases
    where the two changes depart from their shared direction and are therefore
    useful paired-evaluation candidates.
    """

    denom = _dot(effect_b, effect_b)
    beta = 0.0 if denom <= 1e-18 else _dot(effect_a, effect_b) / denom
    residual = np.asarray(effect_a, dtype=float) - beta * np.asarray(effect_b, dtype=float)
    energy = _dot(residual, residual) / max(_dot(effect_a, effect_a), 1e-18)
    return np.abs(residual), float(energy)


def _p90(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return math.nan
    i = int(round(0.9 * (len(xs) - 1)))
    return xs[i]


def _separation_trial(
    effect_a: np.ndarray,
    effect_b: np.ndarray,
    absolute_indices: np.ndarray,
    *,
    k: int,
    random_trials: int = 250,
) -> SeparationResult:
    scores, residual_energy = _separator_scores(effect_a, effect_b)
    order = np.argsort(-scores)
    chosen_local = order[:k]
    suggested = _snapshot(effect_a[chosen_local], effect_b[chosen_local])

    rng = random.Random(20260817)
    n = len(effect_a)
    random_eta: list[float] = []
    random_cos: list[float] = []
    population = list(range(n))
    for _ in range(random_trials):
        sample = rng.sample(population, k)
        snap = _snapshot(effect_a[sample], effect_b[sample])
        random_eta.append(min(snap.regularization_eta, snap.temperature_eta))
        random_cos.append(snap.alias_cosine)

    return SeparationResult(
        k=k,
        suggested_eta=min(suggested.regularization_eta, suggested.temperature_eta),
        suggested_alias_cosine=suggested.alias_cosine,
        random_eta_median=median(random_eta),
        random_eta_p90=_p90(random_eta),
        random_alias_cosine_median=median(random_cos),
        residual_energy_fraction=residual_energy,
        suggested_indices=[int(absolute_indices[i]) for i in chosen_local],
    )


def run(output: Path, *, audit_size: int, candidate_size: int, separator_k: int) -> dict:
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
    y_test = np.asarray(test.target, dtype=int)

    if audit_size + candidate_size > x_test.shape[0]:
        raise ValueError(
            f"requested {audit_size + candidate_size} test examples but only {x_test.shape[0]} exist"
        )

    baseline = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear",
        random_state=7,
    )
    baseline.fit(x_train, y_train)
    base_margin_all = np.asarray(baseline.decision_function(x_test), dtype=float)

    rng = np.random.default_rng(20260817)
    perm = rng.permutation(x_test.shape[0])
    audit_idx = perm[:audit_size]
    candidate_idx = perm[audit_size : audit_size + candidate_size]

    chosen_c, regularized, reg_effect_audit, grid_similarity = _choose_regularization(
        x_train,
        y_train,
        x_test[audit_idx],
        base_margin_all[audit_idx],
    )
    reg_margin_all = np.asarray(regularized.decision_function(x_test), dtype=float)
    reg_effect_all = reg_margin_all - base_margin_all

    temperature_scale = _fit_temperature(reg_effect_audit, base_margin_all[audit_idx])
    temp_margin_all = temperature_scale * base_margin_all
    temp_effect_all = temp_margin_all - base_margin_all

    audit_before = _snapshot(reg_effect_all[audit_idx], temp_effect_all[audit_idx])

    separation = _separation_trial(
        reg_effect_all[candidate_idx],
        temp_effect_all[candidate_idx],
        candidate_idx,
        k=separator_k,
    )

    # Exact duplicate-direction negative control: there should be no separating
    # observation because the two effects are literally scalar copies.
    neg_a = reg_effect_all[audit_idx]
    neg_b = 3.0 * neg_a
    negative_control = _snapshot(neg_a, neg_b)
    neg_scores, neg_energy = _separator_scores(neg_a, neg_b)

    # Give the confidence changes a downstream interpretation.  Coverage is the
    # fraction of examples for which a caller would accept the prediction rather
    # than abstain. The threshold is fixed from the baseline test distribution.
    threshold = float(np.quantile(np.abs(base_margin_all), 0.25))
    combined_margin = temperature_scale * reg_margin_all
    metrics = {
        "baseline": _metrics(base_margin_all, y_test, threshold),
        "regularization": _metrics(reg_margin_all, y_test, threshold),
        "temperature": _metrics(temp_margin_all, y_test, threshold),
        "both": _metrics(combined_margin, y_test, threshold),
    }

    result = {
        "dataset": {
            "name": "20 Newsgroups",
            "categories": list(CATEGORIES),
            "train_examples": int(x_train.shape[0]),
            "test_examples": int(x_test.shape[0]),
            "features": int(x_train.shape[1]),
            "audit_size": int(audit_size),
            "candidate_size": int(candidate_size),
        },
        "changes": {
            "regularization": {"baseline_C": 1.0, "changed_C": chosen_c},
            "temperature": {"score_multiplier": temperature_scale},
            "regularization_vs_baseline_margin_cosine": grid_similarity,
        },
        "audit": asdict(audit_before),
        "separation": asdict(separation),
        "negative_control": {
            **asdict(negative_control),
            "separator_residual_energy_fraction": neg_energy,
            "max_separator_score": float(np.max(neg_scores)),
        },
        "downstream": {
            "abstention_margin_threshold": threshold,
            "metrics": {name: asdict(value) for name, value in metrics.items()},
        },
    }

    print("\nGATE 2 — REAL TEXT PIPELINE ATTRIBUTION")
    print(f"dataset: 20 Newsgroups {CATEGORIES}; train={x_train.shape[0]} test={x_test.shape[0]}")
    print(f"features: {x_train.shape[1]}")
    print(
        f"regularization: C 1.0 -> {chosen_c}; fitted score multiplier={temperature_scale:.6f}"
    )
    print(
        "audit slice: "
        f"cos={audit_before.alias_cosine:.6f} "
        f"eta(reg)={audit_before.regularization_eta:.6f} "
        f"eta(temp)={audit_before.temperature_eta:.6f} "
        f"confounded={audit_before.confounded_regularization and audit_before.confounded_temperature}"
    )
    print(
        f"separator k={separator_k}: eta={separation.suggested_eta:.6f}, "
        f"cos={separation.suggested_alias_cosine:.6f}; "
        f"random median eta={separation.random_eta_median:.6f}, "
        f"random p90 eta={separation.random_eta_p90:.6f}"
    )
    print(
        "negative control scalar duplicate: "
        f"eta={negative_control.regularization_eta:.6f}, "
        f"residual_energy={neg_energy:.3e}"
    )
    print("\ndownstream metrics")
    for name, value in metrics.items():
        print(
            f"  {name:15s} accuracy={value.accuracy:.4f} "
            f"logloss={value.log_loss:.4f} coverage={value.coverage:.4f}"
        )

    # Gate logic. We want both a real confounder and evidence that the targeted
    # slice improves identifiability beyond ordinary random evaluation.
    gate = {
        "confounder_detected": audit_before.alias_cosine >= 0.97,
        "strict_confounded_flag": (
            audit_before.confounded_regularization and audit_before.confounded_temperature
        ),
        "targeted_beats_random_median": separation.suggested_eta > separation.random_eta_median,
        "targeted_beats_random_p90": separation.suggested_eta > separation.random_eta_p90,
        "negative_control_refuses_separation": (
            negative_control.regularization_eta < 1e-9 and neg_energy < 1e-18
        ),
    }
    gate["pass"] = bool(
        gate["confounder_detected"]
        and gate["targeted_beats_random_median"]
        and gate["negative_control_refuses_separation"]
    )
    result["gate"] = gate

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nGATE: {json.dumps(gate, sort_keys=True)}")
    print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gate2_result.json"))
    parser.add_argument("--audit-size", type=int, default=320)
    parser.add_argument("--candidate-size", type=int, default=320)
    parser.add_argument("--separator-k", type=int, default=24)
    args = parser.parse_args()
    result = run(
        args.output,
        audit_size=args.audit_size,
        candidate_size=args.candidate_size,
        separator_k=args.separator_k,
    )
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 2 did not pass its preregistered minimum criteria")


if __name__ == "__main__":
    main()
