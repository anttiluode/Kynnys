import unittest

from kynnys import audit_effects


class AuditTests(unittest.TestCase):
    def test_orthogonal_effects_are_identifiable(self):
        report = {r.name: r for r in audit_effects({"a": [1, 0], "b": [0, 2]})}
        self.assertAlmostEqual(report["a"].identifiable_fraction, 1.0)
        self.assertAlmostEqual(report["b"].identifiable_fraction, 1.0)
        self.assertAlmostEqual(report["b"].sensitivity, 2.0)

    def test_collinear_effects_are_confounded_not_harmless(self):
        report = {r.name: r for r in audit_effects({"a": [10, 0], "b": [2, 0]})}
        self.assertGreater(report["a"].sensitivity, 0)
        self.assertAlmostEqual(report["a"].identifiable_fraction, 0.0)
        self.assertTrue(report["a"].confounded)
        self.assertEqual(report["a"].strongest_alias, "b")

    def test_nuisance_direction_can_explain_effect(self):
        report = audit_effects({"gate": [1, 1]}, nuisance={"warmup": [2, 2]})[0]
        self.assertAlmostEqual(report.identifiable_fraction, 0.0)
        self.assertEqual(report.strongest_alias, "nuisance:warmup")


if __name__ == "__main__":
    unittest.main()
