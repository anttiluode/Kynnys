from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from experiments import gate2_audit_text_pipeline as g2


@dataclass
class OrthogonalProbeResult:
    k: int
    regularization_eta: float
    temperature_eta: float
    alias_cosine: float
    regularization_sensitivity: float
    temperature_sensitivity: float
    random_regularization_eta_median: float
    random_regularization_eta_p90: float
    random_alias_cosine_median: float
    chosen_median_abs_baseline_margin: float
    pool_median_abs_baseline_margin: float
    chosen_docs: list[str]


def _p90(values: list[float]) -> float:
    xs = sorted(values)
    return xs[int(round(0.9 * (len(xs) - 1)))]


def _setup():
    train = fetch_20newsgroups(
        subset="train",
        categories=list(g2.CATEGORIES),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=7,
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=list(g2.CATEGORIES),
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
    base_margin = np.asarray(baseline.decision_function(x_test), dtype=float)

    rng = np.random.default_rng(20260817)
    audit_idx = rng.permutation(x_test.shape[0])[:320]
    chosen_c, regularized, reg_effect_audit, similarity = g2._choose_regularization(
        x_train,
        y_train,
        x_test[audit_idx],
        base_margin[audit_idx],
    )
    reg_margin = np.asarray(regularized.decision_function(x_test), dtype=float)
    temp_scale = g2._fit_temperature(reg_effect_audit, base_margin[audit_idx])
    temp_margin = temp_scale * base_margin
    natural = g2._snapshot(
        reg_margin[audit_idx] - base_margin[audit_idx],
        temp_margin[audit_idx] - base_margin[audit_idx],
    )
    return (
        vectorizer,
        baseline,
        regularized,
        temp_scale,
        chosen_c,
        similarity,
        natural,
        reg_effect_audit,
        (temp_margin - base_margin)[audit_idx],
    )


def _generate_pair_docs(
    vectorizer: TfidfVectorizer,
    baseline: LogisticRegression,
    regularized: LogisticRegression,
    temp_scale: float,
    *,
    feature_budget: int = 90,
) -> list[str]:
    names = np.asarray(vectorizer.get_feature_names_out())
    eligible = g2._eligible_unigram_indices(names)

    w0 = np.asarray(baseline.coef_[0], dtype=float)
    w1 = np.asarray(regularized.coef_[0], dtype=float)
    b0 = float(baseline.intercept_[0])
    b1 = float(regularized.intercept_[0])

    # Feature-level effect signatures. Use residual from the best pure-scaling
    # explanation only to choose a compact pool; the actual probe is evaluated
    # by executing real vectorizer/model calls below.
    a = (w1[eligible] - w0[eligible]) + (b1 - b0)
    b = (temp_scale - 1.0) * (w0[eligible] + b0)
    scores, _ = g2._separator_scores(a, b)
    top = eligible[np.argsort(-scores)[:feature_budget]]

    docs: list[str] = []
    ratios = ((1, 1), (1, 3), (3, 1), (1, 6), (6, 1))
    for ii in range(len(top)):
        ti = str(names[top[ii]])
        for jj in range(ii + 1, len(top)):
            tj = str(names[top[jj]])
            for ci, cj in ratios:
                docs.append(" ".join([ti] * ci + [tj] * cj))
    return docs


def run(output: Path, *, k: int, random_trials: int) -> dict:
    (
        vectorizer,
        baseline,
        regularized,
        temp_scale,
        chosen_c,
        similarity,
        natural,
        natural_a,
        natural_b,
    ) = _setup()

    docs = _generate_pair_docs(vectorizer, baseline, regularized, temp_scale)
    x = vectorizer.transform(docs)
    base = np.asarray(baseline.decision_function(x), dtype=float)
    reg = np.asarray(regularized.decision_function(x), dtype=float)
    temp = temp_scale * base
    effect_a = reg - base
    effect_b = temp - base

    # Use the natural audit to define the shared alias direction, then design
    # probes that suppress it.  First keep pair-documents with unusually small
    # baseline margin: a positive score scaling then has little leverage.
    denom = g2._dot(natural_b, natural_b)
    beta = 0.0 if denom <= 1e-18 else g2._dot(natural_a, natural_b) / denom
    residual = effect_a - beta * effect_b

    abs_base = np.abs(base)
    null_cut = float(np.quantile(abs_base, 0.15))
    pool = np.flatnonzero(abs_base <= null_cut)
    if len(pool) < k * 4:
        raise RuntimeError("orthogonal probe candidate pool is unexpectedly small")

    # A separating set should contain both sides of the departure from the
    # shared direction rather than merely the largest absolute residuals.
    half = k // 2
    pos = pool[np.argsort(-residual[pool])[:half]]
    neg = pool[np.argsort(residual[pool])[:half]]
    chosen = np.unique(np.concatenate([pos, neg]))
    if len(chosen) < k:
        ranked = pool[np.argsort(-np.abs(residual[pool]))]
        fill = [i for i in ranked if i not in set(chosen.tolist())]
        chosen = np.concatenate([chosen, np.asarray(fill[: k - len(chosen)], dtype=int)])
    chosen = chosen[:k]

    snap = g2._snapshot(effect_a[chosen], effect_b[chosen])

    rng = random.Random(20260817)
    pool_list = pool.tolist()
    random_eta: list[float] = []
    random_cos: list[float] = []
    for _ in range(random_trials):
        idx = rng.sample(pool_list, k)
        rs = g2._snapshot(effect_a[idx], effect_b[idx])
        random_eta.append(rs.regularization_eta)
        random_cos.append(rs.alias_cosine)

    probe = OrthogonalProbeResult(
        k=k,
        regularization_eta=snap.regularization_eta,
        temperature_eta=snap.temperature_eta,
        alias_cosine=snap.alias_cosine,
        regularization_sensitivity=snap.regularization_sensitivity,
        temperature_sensitivity=snap.temperature_sensitivity,
        random_regularization_eta_median=median(random_eta),
        random_regularization_eta_p90=_p90(random_eta),
        random_alias_cosine_median=median(random_cos),
        chosen_median_abs_baseline_margin=float(median(abs_base[chosen].tolist())),
        pool_median_abs_baseline_margin=float(median(abs_base[pool].tolist())),
        chosen_docs=[docs[i] for i in chosen],
    )

    # Exact scalar duplicate control remains impossible to separate.
    neg_a = np.asarray(natural_a, dtype=float)
    neg_b = 2.5 * neg_a
    neg = g2._snapshot(neg_a, neg_b)
    _, neg_energy = g2._separator_scores(neg_a, neg_b)

    gate = {
        "natural_outputs_strictly_confounded": (
            natural.alias_cosine >= 0.995
            and natural.confounded_regularization
            and natural.confounded_temperature
        ),
        "orthogonal_probe_meaningfully_separates_regularization": (
            probe.regularization_eta >= 0.25
        ),
        "orthogonal_probe_beats_random_p90": (
            probe.regularization_eta > probe.random_regularization_eta_p90
        ),
        "shared_direction_was_actually_nulled": (
            probe.chosen_median_abs_baseline_margin <= probe.pool_median_abs_baseline_margin
        ),
        "negative_control_refuses_separation": (
            neg.regularization_eta < 1e-9 and neg_energy < 1e-18
        ),
    }
    gate["pass"] = all(gate.values())

    result = {
        "changes": {
            "regularization_C": chosen_c,
            "temperature_score_multiplier": temp_scale,
            "regularization_vs_baseline_margin_cosine": similarity,
        },
        "natural_audit": asdict(natural),
        "orthogonal_probe": asdict(probe),
        "negative_control": {
            **asdict(neg),
            "residual_energy_fraction": neg_energy,
        },
        "candidate_documents": len(docs),
        "near_null_pool": len(pool),
        "gate": gate,
    }

    print("\nGATE 2c — ORTHOGONALIZED ATTRIBUTION PROBE")
    print(
        f"natural: cos={natural.alias_cosine:.6f} "
        f"eta(reg)={natural.regularization_eta:.6f}"
    )
    print(
        f"constructed pair docs={len(docs)}, near-null pool={len(pool)}, k={k}"
    )
    print(
        f"orthogonal probe: eta(reg)={probe.regularization_eta:.6f}, "
        f"eta(temp)={probe.temperature_eta:.6f}, cos={probe.alias_cosine:.6f}"
    )
    print(
        f"random near-null controls: median eta={probe.random_regularization_eta_median:.6f}, "
        f"p90={probe.random_regularization_eta_p90:.6f}"
    )
    print(
        f"sensitivities: reg={probe.regularization_sensitivity:.6f}, "
        f"temp={probe.temperature_sensitivity:.6f}"
    )
    print("probe examples:")
    for doc in probe.chosen_docs[:10]:
        print(f"  {doc}")
    print(f"negative scalar control eta={neg.regularization_eta:.3e}")
    print(f"GATE: {json.dumps(gate, sort_keys=True)}")

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gate2c_result.json"))
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=250)
    args = parser.parse_args()
    result = run(args.output, k=args.k, random_trials=args.random_trials)
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 2c did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
