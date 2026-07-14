#!/usr/bin/env python3
"""Unit tests for the deterministic paraphrase quality gates."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "esft"))
import paraphrase_gates as gates  # noqa: E402


def seed() -> dict:
    return {
        "seed_id": "seed-0007",
        "tier": "T2",
        "style_card": {"name": "neutral_direct"},
        "tools": [{
            "name": "fulfill-order",
            "description": "Complete a customer fulfilment operation.",
            "parameters": {"type": "object", "required": ["tracking_code", "rate"], "properties": {
                "tracking_code": {"type": "string", "description": "The customer tracking code."},
                "rate": {"type": "number", "description": "The approved service rate."},
            }},
        }],
        "expected_stages": [[{"name": "fulfill-order", "arguments": {"tracking_code": "PK-7", "rate": 0.75}}]],
        "derived_values": [],
    }


class ParaphraseGateTests(unittest.TestCase):
    def test_positive_typed_values_accept_numeric_equivalence(self):
        result = gates.check_request(seed(), 'Please process "PK-7" at .75 when you can.')
        self.assertTrue(result.passed, result.failures)

    def test_typed_value_fidelity_rejects_missing_string(self):
        result = gates.check_request(seed(), 'Please process "not-PK-7" at .75 when you can.')
        self.assertIn("typed_value_fidelity", result.failures)

    def test_schema_semantics_rejects_missing_property_description(self):
        broken = seed()
        del broken["tools"][0]["parameters"]["properties"]["rate"]["description"]
        result = gates.check_request(broken, 'Please process "PK-7" at .75 when you can.')
        self.assertIn("schema_semantics", result.failures)

    def test_schema_leak_rejects_normalized_and_exact_stopword_names(self):
        result = gates.check_request(seed(), 'Please process "PK-7" at .75 with the tracking code.')
        self.assertIn("schema_derived_leak", result.failures)
        stopword_seed = seed()
        stopword_seed["tools"][0]["parameters"]["properties"]["status"] = {
            "type": "string", "description": "The requested public status."
        }
        result = gates.check_schema_leaks(stopword_seed, 'Could you share the status for "PK-7"?')
        self.assertIn("schema_derived_leak", result.failures)

    def test_operational_invention_rejects_unknown_number_and_quote(self):
        result = gates.check_request(seed(), 'Please process "PK-7" at .75, cap it at 12, and use "rush".')
        self.assertIn("operational_invention", result.failures)

    def test_batch_prefix_and_style_quota_rejects_violating_group(self):
        rows = [{"natural_request": f"Please make item {index} distinct from all prior examples.", "style_card": "neutral_direct"}
                for index in range(20)]
        results = [gates.GateResult(True) for _ in rows]
        gates.apply_batch_gates(rows, results)
        self.assertTrue(all("batch_prefix_2" in result.failures for result in results))
        self.assertTrue(all("batch_style" in result.failures for result in results))

    def test_near_duplicate_rejects_later_row(self):
        rows = [
            {"natural_request": 'Please process "PK-7" at .75 right away.', "style_card": "neutral_direct"},
            {"natural_request": 'Please process "PK-7" at .75 right away!', "style_card": "conversational"},
        ]
        results = [gates.GateResult(True), gates.GateResult(True)]
        gates.apply_batch_gates(rows, results)
        self.assertTrue(results[0].passed)
        self.assertIn("skeleton_similarity", results[1].failures)


if __name__ == "__main__":
    unittest.main()


class ContainerElementTest(unittest.TestCase):
    def test_list_elements_are_not_inventions(self) -> None:
        seed = {"expected_stages": [[{"name": "convert",
                                      "arguments": {"a": 2344.75, "b": ["tag-2346", "tag-2347"]}}]]}
        text = 'Convert the setup using 2344.75 and ["tag-2346","tag-2347"].'
        result = gates.check_operational_invention(seed, text)
        self.assertTrue(result.passed, result.failures)
