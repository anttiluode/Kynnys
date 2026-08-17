from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


Vector = Sequence[float]


@dataclass(frozen=True)
class EffectAudit:
    name: str
    sensitivity: float
    identifiable_fraction: float
    strongest_alias: str | None
    alias_cosine: float

    @property
    def confounded(self) -> bool:
        return self.sensitivity > 0.0 and self.identifiable_fraction < 0.1


def _dot(a: Vector, b: Vector) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(v: Vector) -> float:
    return math.sqrt(_dot(v, v))


def _scale(v: Vector, s: float) -> list[float]:
    return [float(x) * s for x in v]


def _sub(a: Vector, b: Vector) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _orthonormal_basis(vectors: Sequence[Vector], *, tol: float = 1e-12) -> list[list[float]]:
    basis: list[list[float]] = []
    for raw in vectors:
        v = [float(x) for x in raw]
        for q in basis:
            v = _sub(v, _scale(q, _dot(v, q)))
        n = _norm(v)
        if n > tol:
            basis.append(_scale(v, 1.0 / n))
    return basis


def _residual(v: Vector, basis_vectors: Sequence[Vector]) -> list[float]:
    residual = [float(x) for x in v]
    for q in _orthonormal_basis(basis_vectors):
        residual = _sub(residual, _scale(q, _dot(residual, q)))
    return residual


def _cosine(a: Vector, b: Vector) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def audit_effects(
    effects: Mapping[str, Vector],
    *,
    nuisance: Mapping[str, Vector] | None = None,
) -> list[EffectAudit]:
    """Audit impact and independent identifiability of gate-effect vectors.

    This is a small domain-independent transplant of the epistemic lesson from
    TransientWaveCompiler: a large effect can matter while still being
    impossible to attribute uniquely from the observed output.

    For effect g_i, `identifiable_fraction` is

        ||(I - P_Jminus_i) g_i|| / ||g_i||

    where Jminus_i contains all other gate effects plus declared nuisance
    directions.  A value near zero means "important perhaps, but confounded",
    not "harmless".
    """

    names = list(effects)
    if not names:
        return []
    dim = len(effects[names[0]])
    if dim == 0:
        raise ValueError("effect vectors must be non-empty")
    for name, v in effects.items():
        if len(v) != dim:
            raise ValueError(f"effect {name!r} has dimension {len(v)}; expected {dim}")
    nuisance = nuisance or {}
    for name, v in nuisance.items():
        if len(v) != dim:
            raise ValueError(f"nuisance {name!r} has dimension {len(v)}; expected {dim}")

    out: list[EffectAudit] = []
    nuisance_vectors = list(nuisance.values())
    for name in names:
        g = effects[name]
        sensitivity = _norm(g)
        others = [effects[other] for other in names if other != name] + nuisance_vectors
        residual = _residual(g, others)
        eta = 0.0 if sensitivity == 0.0 else min(1.0, _norm(residual) / sensitivity)

        alias_name: str | None = None
        alias_cosine = 0.0
        candidates = {other: effects[other] for other in names if other != name}
        candidates.update({f"nuisance:{k}": v for k, v in nuisance.items()})
        for other, vector in candidates.items():
            c = abs(_cosine(g, vector))
            if c > alias_cosine:
                alias_name = other
                alias_cosine = c

        out.append(
            EffectAudit(
                name=name,
                sensitivity=sensitivity,
                identifiable_fraction=eta,
                strongest_alias=alias_name,
                alias_cosine=alias_cosine,
            )
        )
    return out
