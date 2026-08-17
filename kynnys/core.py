from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
import contextvars
import hashlib
import math
import pickle
import threading
import time
from typing import Any, Callable, Generic, Mapping, TypeVar


T = TypeVar("T")


class Action(str, Enum):
    """Observable admission outcome for one demand."""

    REUSE = "REUSE"
    PROBE_REUSE = "PROBE_REUSE"
    PROBE_RUN = "PROBE_RUN"
    RUN = "RUN"
    WAIT = "WAIT"
    HOLD = "HOLD"


class GateError(RuntimeError):
    pass


class EgressViolation(GateError):
    pass


class PrivateEscapeError(GateError):
    pass


@dataclass(frozen=True)
class Private(Generic[T]):
    """Marks a value as gate-private for dynamic boundary checks.

    This is a runtime guard, not a Python capability-security mechanism.  A
    future compiler/IR can make the boundary static; Python cannot prevent a
    gate body from deliberately leaking an object through unrelated globals.
    """

    value: T


ProbeFn = Callable[[Any, "GateCall[Any]", float], bool]
KeyFn = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]
GateFn = Callable[..., T]


@dataclass(frozen=True)
class GateSpec(Generic[T]):
    name: str
    fn: GateFn[T]
    compute_cost: float = 1.0
    probe: ProbeFn | None = None
    probe_cost: float = 0.0
    hazard_rate: float = 0.0
    max_egress_bytes: int | None = None
    key_fn: KeyFn | None = None
    pass_context: bool = False

    def __post_init__(self) -> None:
        if self.compute_cost < 0 or self.probe_cost < 0 or self.hazard_rate < 0:
            raise ValueError("costs and hazard_rate must be non-negative")
        if self.max_egress_bytes is not None and self.max_egress_bytes < 0:
            raise ValueError("max_egress_bytes must be non-negative")
        if self.probe is None and self.probe_cost:
            raise ValueError("probe_cost requires a probe")


@dataclass(frozen=True)
class GateCall(Generic[T]):
    """A possible computation. Constructing one never executes its gate body."""

    spec: GateSpec[T]
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]

    def cache_key(self) -> str:
        if self.spec.key_fn is not None:
            payload = self.spec.key_fn(self.args, self.kwargs)
        else:
            payload = (self.args, tuple(sorted(self.kwargs.items())))
        try:
            raw = pickle.dumps(payload, protocol=5)
        except Exception as exc:  # pragma: no cover - message path is tested indirectly
            raise TypeError(
                f"arguments for gate {self.spec.name!r} are not stably serializable; "
                "provide key_fn=..."
            ) from exc
        digest = hashlib.blake2b(raw, digest_size=20).hexdigest()
        return f"{self.spec.name}:{digest}"


@dataclass(frozen=True)
class Demand:
    """Requirements attached to a demanded consequence.

    error_cost is the cost of reusing a cached result that has become invalid.
    math.inf means that uncertain reuse is not permitted. max_spend bounds the
    actual control+compute spend for this demand.
    """

    error_cost: float = math.inf
    max_spend: float = math.inf
    allow_probe: bool = True

    def __post_init__(self) -> None:
        if self.error_cost < 0 or self.max_spend < 0:
            raise ValueError("error_cost and max_spend must be non-negative")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    validated_at: float
    egress_bytes: int
    generation: int


@dataclass(frozen=True)
class Outcome(Generic[T]):
    value: T | None
    action: Action
    cost: float
    p_invalid: float | None
    reason: str
    gate: str
    generation: int | None
    waited: bool = False


@dataclass
class _InFlight(Generic[T]):
    future: Future[CacheEntry[T]]


@dataclass
class GateContext:
    """Execution-only access to persistent gate-local private state."""

    runtime: "Runtime"
    spec: GateSpec[Any]
    call_key: str

    def local(self, name: str, factory: Callable[[], T]) -> T:
        if not name:
            raise ValueError("local state name must be non-empty")
        return self.runtime._get_local(self.spec.name, name, factory)

    def private(self, value: T) -> Private[T]:
        return Private(value)


_current_context: contextvars.ContextVar[GateContext | None] = contextvars.ContextVar(
    "kynnys_gate_context", default=None
)


def current_context() -> GateContext:
    ctx = _current_context.get()
    if ctx is None:
        raise GateError("current_context() is only available while a gate body is executing")
    return ctx


def exact(*, max_spend: float = math.inf, allow_probe: bool = True) -> Demand:
    return Demand(error_cost=math.inf, max_spend=max_spend, allow_probe=allow_probe)


def risk(error_cost: float, *, max_spend: float = math.inf, allow_probe: bool = True) -> Demand:
    return Demand(error_cost=error_cost, max_spend=max_spend, allow_probe=allow_probe)


def gate(
    *,
    compute_cost: float = 1.0,
    probe: ProbeFn | None = None,
    probe_cost: float = 0.0,
    hazard_rate: float = 0.0,
    max_egress_bytes: int | None = None,
    key_fn: KeyFn | None = None,
    pass_context: bool = False,
    name: str | None = None,
) -> Callable[[GateFn[T]], Callable[..., GateCall[T]]]:
    """Turn a Python function into a lazy Kynnys gate.

    Calling the decorated function creates a GateCall. The function body runs
    only when a Runtime admits a demand for that call.
    """

    def decorate(fn: GateFn[T]) -> Callable[..., GateCall[T]]:
        spec = GateSpec(
            name=name or fn.__qualname__,
            fn=fn,
            compute_cost=float(compute_cost),
            probe=probe,
            probe_cost=float(probe_cost),
            hazard_rate=float(hazard_rate),
            max_egress_bytes=max_egress_bytes,
            key_fn=key_fn,
            pass_context=pass_context,
        )

        def make_call(*args: Any, **kwargs: Any) -> GateCall[T]:
            return GateCall(spec=spec, args=args, kwargs=dict(kwargs))

        make_call.__name__ = fn.__name__
        make_call.__qualname__ = fn.__qualname__
        make_call.__doc__ = fn.__doc__
        setattr(make_call, "gate_spec", spec)
        return make_call

    return decorate


class Runtime:
    """Reference interpreter for Kynnys demand/admission semantics.

    The runtime deliberately stays ordinary: cache + explicit uncertainty +
    value-of-information probe choice + in-flight de-duplication.  The research
    object is the programming contract, not a claim that these algorithms are
    individually new.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.cache: dict[str, CacheEntry[Any]] = {}
        self.events: list[Outcome[Any]] = []
        self._local_state: dict[tuple[str, str], Any] = {}
        self._inflight: dict[str, _InFlight[Any]] = {}
        self._lock = threading.RLock()
        self._generation = 0

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.events.clear()
            self._local_state.clear()
            self._inflight.clear()
            self._generation = 0

    def _get_local(self, gate_name: str, name: str, factory: Callable[[], T]) -> T:
        slot = (gate_name, name)
        with self._lock:
            if slot not in self._local_state:
                self._local_state[slot] = factory()
            return self._local_state[slot]

    @staticmethod
    def _p_invalid(hazard_rate: float, age: float) -> float:
        if hazard_rate <= 0 or age <= 0:
            return 0.0
        # hazard_rate is explicitly a continuous-time rate, so exp(-lambda t)
        # is the intended model. Discrete Bernoulli hazards should be converted
        # by the caller instead of silently mixing the two parameterizations.
        return 1.0 - math.exp(-hazard_rate * age)

    @staticmethod
    def _contains_private(value: Any, seen: set[int] | None = None) -> bool:
        if isinstance(value, Private):
            return True
        if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool, complex)):
            return False
        seen = set() if seen is None else seen
        marker = id(value)
        if marker in seen:
            return False
        seen.add(marker)
        if isinstance(value, Mapping):
            return any(
                Runtime._contains_private(k, seen) or Runtime._contains_private(v, seen)
                for k, v in value.items()
            )
        if isinstance(value, (tuple, list, set, frozenset)):
            return any(Runtime._contains_private(v, seen) for v in value)
        if hasattr(value, "__dict__"):
            return any(Runtime._contains_private(v, seen) for v in vars(value).values())
        return False

    @staticmethod
    def _egress_size(value: Any) -> int:
        try:
            return len(pickle.dumps(value, protocol=5))
        except Exception as exc:
            raise EgressViolation(
                "gate output cannot be serialized for the egress contract"
            ) from exc

    def _check_egress(self, spec: GateSpec[Any], value: Any) -> int:
        if self._contains_private(value):
            raise PrivateEscapeError(
                f"gate {spec.name!r} attempted to return a Private value across its boundary"
            )
        size = self._egress_size(value)
        if spec.max_egress_bytes is not None and size > spec.max_egress_bytes:
            raise EgressViolation(
                f"gate {spec.name!r} emitted {size} bytes; "
                f"contract permits at most {spec.max_egress_bytes}"
            )
        return size

    def _record(self, outcome: Outcome[T]) -> Outcome[T]:
        with self._lock:
            self.events.append(outcome)
        return outcome

    def _execute(self, call: GateCall[T], key: str, now: float) -> tuple[CacheEntry[T], bool]:
        """Execute once; concurrent equivalent demanders wait on the same work."""

        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                future = existing.future
                owner = False
            else:
                future = Future()
                self._inflight[key] = _InFlight(future=future)
                owner = True

        if not owner:
            return future.result(), True

        ctx = GateContext(runtime=self, spec=call.spec, call_key=key)
        token = _current_context.set(ctx)
        try:
            if call.spec.pass_context:
                value = call.spec.fn(ctx, *call.args, **dict(call.kwargs))
            else:
                value = call.spec.fn(*call.args, **dict(call.kwargs))
            egress_bytes = self._check_egress(call.spec, value)
            with self._lock:
                self._generation += 1
                entry = CacheEntry(
                    value=value,
                    validated_at=now,
                    egress_bytes=egress_bytes,
                    generation=self._generation,
                )
                self.cache[key] = entry
            future.set_result(entry)
            return entry, False
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            _current_context.reset(token)
            with self._lock:
                self._inflight.pop(key, None)

    def demand(
        self,
        call: GateCall[T],
        requirements: Demand | None = None,
        *,
        now: float | None = None,
    ) -> Outcome[T]:
        if not isinstance(call, GateCall):
            raise TypeError("Runtime.demand expects a GateCall; decorate the function with @gate")
        req = requirements or exact()
        now = self.clock() if now is None else float(now)
        key = call.cache_key()
        spec = call.spec

        with self._lock:
            in_flight = self._inflight.get(key)
            cached = self.cache.get(key)

        if in_flight is not None:
            entry = in_flight.future.result()
            return self._record(
                Outcome(
                    value=entry.value,
                    action=Action.WAIT,
                    cost=0.0,
                    p_invalid=None,
                    reason="equivalent_work_already_in_flight",
                    gate=spec.name,
                    generation=entry.generation,
                    waited=True,
                )
            )

        if cached is None:
            if spec.compute_cost > req.max_spend:
                return self._record(
                    Outcome(None, Action.HOLD, 0.0, None, "compute_exceeds_max_spend", spec.name, None)
                )
            entry, waited = self._execute(call, key, now)
            return self._record(
                Outcome(
                    value=entry.value,
                    action=Action.WAIT if waited else Action.RUN,
                    cost=0.0 if waited else spec.compute_cost,
                    p_invalid=0.0,
                    reason="cache_miss",
                    gate=spec.name,
                    generation=entry.generation,
                    waited=waited,
                )
            )

        age = max(0.0, now - cached.validated_at)
        p_invalid = self._p_invalid(spec.hazard_rate, age)
        reuse_loss = p_invalid * req.error_cost
        run_cost = spec.compute_cost
        probe_expected = math.inf
        if req.allow_probe and spec.probe is not None:
            probe_expected = spec.probe_cost + p_invalid * spec.compute_cost

        # No hidden-world uncertainty: exact input identity is enough.
        if p_invalid == 0.0:
            return self._record(
                Outcome(
                    cached.value,
                    Action.REUSE,
                    0.0,
                    0.0,
                    "cached_consequence_exact_for_declared_model",
                    spec.name,
                    cached.generation,
                )
            )

        choice = min(
            ((reuse_loss, "reuse"), (run_cost, "run"), (probe_expected, "probe")),
            key=lambda pair: (pair[0], {"reuse": 0, "probe": 1, "run": 2}[pair[1]]),
        )[1]

        if choice == "reuse":
            return self._record(
                Outcome(
                    cached.value,
                    Action.REUSE,
                    0.0,
                    p_invalid,
                    "risk_budget_prefers_uncertain_reuse",
                    spec.name,
                    cached.generation,
                )
            )

        if choice == "probe":
            if spec.probe_cost > req.max_spend:
                return self._record(
                    Outcome(cached.value, Action.HOLD, 0.0, p_invalid, "probe_exceeds_max_spend", spec.name, cached.generation)
                )
            valid = bool(spec.probe(cached.value, call, now))
            if valid:
                with self._lock:
                    cached.validated_at = now
                return self._record(
                    Outcome(
                        cached.value,
                        Action.PROBE_REUSE,
                        spec.probe_cost,
                        0.0,
                        "probe_confirmed_cached_consequence",
                        spec.name,
                        cached.generation,
                    )
                )
            total = spec.probe_cost + spec.compute_cost
            if total > req.max_spend:
                return self._record(
                    Outcome(
                        cached.value,
                        Action.HOLD,
                        spec.probe_cost,
                        1.0,
                        "probe_rejected_cache_but_refresh_exceeds_remaining_contract",
                        spec.name,
                        cached.generation,
                    )
                )
            entry, waited = self._execute(call, key, now)
            return self._record(
                Outcome(
                    entry.value,
                    Action.WAIT if waited else Action.PROBE_RUN,
                    0.0 if waited else total,
                    1.0,
                    "probe_rejected_cache_then_ran",
                    spec.name,
                    entry.generation,
                    waited=waited,
                )
            )

        if spec.compute_cost > req.max_spend:
            return self._record(
                Outcome(cached.value, Action.HOLD, 0.0, p_invalid, "compute_exceeds_max_spend", spec.name, cached.generation)
            )
        entry, waited = self._execute(call, key, now)
        return self._record(
            Outcome(
                entry.value,
                Action.WAIT if waited else Action.RUN,
                0.0 if waited else spec.compute_cost,
                p_invalid,
                "refresh_cheaper_than_reuse_or_probe",
                spec.name,
                entry.generation,
                waited=waited,
            )
        )

    def metrics(self) -> dict[str, Any]:
        counts = {a.value: 0 for a in Action}
        total_cost = 0.0
        for event in self.events:
            counts[event.action.value] += 1
            total_cost += event.cost
        return {
            "demands": len(self.events),
            "actions": counts,
            "total_cost": total_cost,
            "cache_entries": len(self.cache),
            "private_slots": len(self._local_state),
            "in_flight": len(self._inflight),
        }


_default_runtime = Runtime()


def demand(
    call: GateCall[T],
    requirements: Demand | None = None,
    *,
    runtime: Runtime | None = None,
    now: float | None = None,
) -> Outcome[T]:
    """Demand a gate consequence from the selected runtime."""

    return (runtime or _default_runtime).demand(call, requirements, now=now)
