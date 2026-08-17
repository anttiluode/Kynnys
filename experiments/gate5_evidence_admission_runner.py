"""Mechanical runner for Gate 5.

The original experiment allowed routes to change output dimension but stored the
noise covariance on the case.  For the two exact-alias cases, keep the policy and
all thresholds unchanged and embed the one-dimensional sum readout in two output
dimensions so current and alternate receivers share the same covariance shape.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments import gate5_evidence_admission as g5


_original_cases = g5._cases


def _fixed_cases():
    out = []
    for case in _original_cases():
        if case.name in {"exact_alias_refuse", "exact_alias_route"}:
            case = replace(
                case,
                observation=np.asarray([[1.0, 1.0], [0.0, 0.0]]),
                nuisance=np.zeros((2, 0)),
                covariance=np.eye(2),
            )
        out.append(case)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("gate5_result.json"))
    args = ap.parse_args()
    g5._cases = _fixed_cases
    result = g5.run(args.output)
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 5 did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
