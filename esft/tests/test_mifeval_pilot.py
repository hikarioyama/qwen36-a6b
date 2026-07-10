from __future__ import annotations

import sys
import unittest
from pathlib import Path

ESFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ESFT))

import mifeval_pilot


class MifevalPilotTests(unittest.TestCase):
    def test_binary_seed_agreement(self):
        self.assertEqual(mifeval_pilot.agreement([True] * 5), 1.0)
        self.assertEqual(mifeval_pilot.agreement([False] * 5), 1.0)
        self.assertEqual(mifeval_pilot.agreement([True, True, False, False, False]), 0.4)

    def test_paired_bootstrap_is_item_clustered_and_seeded(self):
        base = {"a": 0.0, "b": 1.0}
        b2 = {"a": 1.0, "b": 1.0}
        out = mifeval_pilot.paired_bootstrap(base, b2)
        self.assertEqual(out["delta_b2_minus_base"], 0.5)
        self.assertEqual(out["replicates"], mifeval_pilot.BOOTSTRAP_REPS)
        self.assertIn("clustered within item", out["method"])

    def test_prepare_requires_all_source_items_and_stable_keys(self):
        with mifeval_pilot.INPUT.open(encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f if line.strip()), 172)
        self.assertEqual(len(mifeval_pilot.sha256_file(mifeval_pilot.INPUT)), 64)


if __name__ == "__main__":
    unittest.main()
