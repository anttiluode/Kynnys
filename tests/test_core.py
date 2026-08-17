import math
import threading
import time
import unittest

from kynnys import (
    Action,
    EgressViolation,
    PrivateEscapeError,
    Runtime,
    exact,
    gate,
    risk,
)


class KynnysCoreTests(unittest.TestCase):
    def test_call_is_description_not_execution(self):
        calls = []

        @gate()
        def work(x):
            calls.append(x)
            return x * 2

        rt = Runtime()
        pending = work(3)
        self.assertEqual(calls, [])
        out = rt.demand(pending, now=0.0)
        self.assertEqual(out.value, 6)
        self.assertEqual(out.action, Action.RUN)
        self.assertEqual(calls, [3])

    def test_exact_declared_input_reuses_without_hidden_hazard(self):
        count = 0

        @gate(compute_cost=10)
        def work(x):
            nonlocal count
            count += 1
            return x + count

        rt = Runtime()
        first = rt.demand(work(1), exact(), now=0.0)
        second = rt.demand(work(1), exact(), now=100.0)
        self.assertEqual(first.action, Action.RUN)
        self.assertEqual(second.action, Action.REUSE)
        self.assertEqual(count, 1)

    def test_risk_can_prefer_reuse(self):
        count = 0

        @gate(compute_cost=10, hazard_rate=1.0)
        def work():
            nonlocal count
            count += 1
            return count

        rt = Runtime()
        rt.demand(work(), now=0.0)
        out = rt.demand(work(), risk(0.1), now=10.0)
        self.assertEqual(out.action, Action.REUSE)
        self.assertEqual(count, 1)
        self.assertGreater(out.p_invalid, 0.99)

    def test_probe_has_value_of_information_band(self):
        probe_calls = 0
        run_calls = 0

        def probe(value, call, now):
            nonlocal probe_calls
            probe_calls += 1
            return True

        @gate(compute_cost=12, probe=probe, probe_cost=5, hazard_rate=math.log(2))
        def work():
            nonlocal run_calls
            run_calls += 1
            return run_calls

        rt = Runtime()
        rt.demand(work(), now=0.0)
        # At age 1, p_invalid=.5. Reuse=30, Run=12, Probe=11.
        out = rt.demand(work(), risk(60), now=1.0)
        self.assertEqual(out.action, Action.PROBE_REUSE)
        self.assertEqual(out.cost, 5)
        self.assertEqual(probe_calls, 1)
        self.assertEqual(run_calls, 1)

    def test_probe_cannot_strand_affordable_direct_run_under_max_spend(self):
        probe_calls = 0
        run_calls = 0

        def probe(value, call, now):
            nonlocal probe_calls
            probe_calls += 1
            return False

        @gate(compute_cost=12, probe=probe, probe_cost=5, hazard_rate=math.log(2))
        def work():
            nonlocal run_calls
            run_calls += 1
            return run_calls

        rt = Runtime()
        rt.demand(work(), now=0.0)
        # At p=.5 the unconstrained expected costs are probe=11, run=12.
        # But max_spend=12 cannot cover the probe-fails-then-run path (17),
        # while direct execution is admissible. The runtime must RUN directly.
        out = rt.demand(work(), risk(60, max_spend=12), now=1.0)
        self.assertEqual(out.action, Action.RUN)
        self.assertEqual(out.cost, 12)
        self.assertEqual(probe_calls, 0)
        self.assertEqual(run_calls, 2)

    def test_failed_probe_runs_and_charges_both(self):
        def probe(value, call, now):
            return False

        calls = 0

        @gate(compute_cost=12, probe=probe, probe_cost=5, hazard_rate=math.log(2))
        def work():
            nonlocal calls
            calls += 1
            return calls

        rt = Runtime()
        rt.demand(work(), now=0.0)
        out = rt.demand(work(), risk(60), now=1.0)
        self.assertEqual(out.action, Action.PROBE_RUN)
        self.assertEqual(out.cost, 17)
        self.assertEqual(out.value, 2)

    def test_exact_volatile_result_will_not_gamble(self):
        calls = 0

        @gate(compute_cost=3, hazard_rate=1.0)
        def work():
            nonlocal calls
            calls += 1
            return calls

        rt = Runtime()
        rt.demand(work(), exact(), now=0.0)
        out = rt.demand(work(), exact(), now=1.0)
        self.assertEqual(out.action, Action.RUN)
        self.assertEqual(calls, 2)

    def test_max_spend_can_hold(self):
        @gate(compute_cost=10)
        def work():
            return 1

        out = Runtime().demand(work(), exact(max_spend=2), now=0.0)
        self.assertEqual(out.action, Action.HOLD)
        self.assertIsNone(out.value)

    def test_egress_bound_is_enforced(self):
        @gate(max_egress_bytes=8)
        def too_wide():
            return "x" * 100

        with self.assertRaises(EgressViolation):
            Runtime().demand(too_wide(), now=0.0)

    def test_private_value_cannot_be_returned(self):
        @gate(pass_context=True)
        def leaks(ctx):
            return {"oops": ctx.private([1, 2, 3])}

        with self.assertRaises(PrivateEscapeError):
            Runtime().demand(leaks(), now=0.0)

    def test_gate_local_state_persists_across_different_calls(self):
        @gate(pass_context=True)
        def counter(ctx, x):
            box = ctx.local("counter", lambda: {"n": 0})
            box["n"] += 1
            return x, box["n"]

        rt = Runtime()
        a = rt.demand(counter("a"), now=0.0)
        b = rt.demand(counter("b"), now=0.0)
        self.assertEqual(a.value, ("a", 1))
        self.assertEqual(b.value, ("b", 2))
        self.assertEqual(rt.metrics()["private_slots"], 1)

    def test_equivalent_concurrent_demands_do_not_duplicate_execution(self):
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        @gate(compute_cost=7)
        def slow(x):
            nonlocal calls
            calls += 1
            entered.set()
            release.wait(timeout=2)
            return x * 10

        rt = Runtime()
        outputs = []

        def run_one():
            outputs.append(rt.demand(slow(2), now=0.0))

        t1 = threading.Thread(target=run_one)
        t2 = threading.Thread(target=run_one)
        t1.start()
        self.assertTrue(entered.wait(timeout=1))
        t2.start()
        time.sleep(0.02)
        release.set()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertEqual(calls, 1)
        self.assertEqual(sorted(o.action.value for o in outputs), ["RUN", "WAIT"])
        self.assertEqual([o.value for o in outputs], [20, 20])


if __name__ == "__main__":
    unittest.main()
