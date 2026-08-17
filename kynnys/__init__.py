"""Kynnys: demand-driven admission for expensive persistent computation."""

from .audit import EffectAudit, audit_effects
from .core import (
    Action,
    Demand,
    EgressViolation,
    GateCall,
    GateContext,
    GateError,
    GateSpec,
    Outcome,
    Private,
    PrivateEscapeError,
    Runtime,
    current_context,
    demand,
    exact,
    gate,
    risk,
)

__all__ = [
    "Action",
    "Demand",
    "EffectAudit",
    "EgressViolation",
    "GateCall",
    "GateContext",
    "GateError",
    "GateSpec",
    "Outcome",
    "Private",
    "PrivateEscapeError",
    "Runtime",
    "audit_effects",
    "current_context",
    "demand",
    "exact",
    "gate",
    "risk",
]

__version__ = "0.1.0"
