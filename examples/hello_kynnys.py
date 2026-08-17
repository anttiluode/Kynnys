"""Small executable demo: demand is not execution."""

import math

from kynnys import Runtime, gate, risk


world = {"version": 1}


def still_same(cached_value, call, now):
    # A cheap opaque-world check. In a real program this might be a HEAD request,
    # CI status query, metadata lookup, sensor check, etc.
    return cached_value["world_version"] == world["version"]


@gate(
    compute_cost=12,
    probe=still_same,
    probe_cost=2,
    hazard_rate=math.log(2) / 10.0,  # 50% invalidity belief after 10 time units
    max_egress_bytes=128,
    pass_context=True,
)
def expensive_interpretation(ctx, prompt):
    stats = ctx.local("stats", lambda: {"runs": 0})
    stats["runs"] += 1
    return {
        "answer": prompt.upper(),
        "world_version": world["version"],
        "runs": stats["runs"],
    }


rt = Runtime()

possible = expensive_interpretation("is the service healthy?")
print("after call construction:", rt.metrics())  # zero computation

print(rt.demand(possible, risk(60), now=0.0))
print(rt.demand(possible, risk(60), now=10.0))  # probe band: cheap validation

world["version"] = 2
print(rt.demand(possible, risk(60), now=20.0))  # probe fails -> run
print(rt.metrics())
