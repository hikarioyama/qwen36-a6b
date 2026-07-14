#!/usr/bin/env python3
"""Offline tests for style-card prompting and local driver gates."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "esft"))
import paraphrase_glm52_fireworks as driver  # noqa: E402


class ParaphraseDriverTests(unittest.TestCase):
    def row(self) -> dict:
        return {
            "seed_id": "seed-0001",
            "tier": "T2",
            "style_card": {"name": "terse_mobile", "instruction": "terse mobile request; compact but complete"},
            "transcription_request": "Formal instruction.",
            "value_literals": ['"PK-7"', "0.75"],
            "gate_seed": {
                "tools": [{"name": "fulfill-order", "parameters": {"properties": {
                    "tracking_code": {"type": "string", "description": "The customer tracking code."},
                    "rate": {"type": "number", "description": "The approved service rate."},
                }}}],
                "expected_stages": [[{"name": "fulfill-order", "arguments": {"tracking_code": "PK-7", "rate": 0.75}}]],
                "derived_values": [],
                "request_values": ["PK-7", 0.75],
            },
        }

    def test_style_card_is_prompted_without_forcing_first_person(self):
        prompt = driver.build_user_prompt(self.row(), None)
        self.assertIn("STYLE CARD (terse_mobile)", prompt)
        self.assertNotIn("first person", driver.SYSTEM_PROMPT.lower())

    def test_paraphrase_row_runs_gates_without_network(self):
        usage = driver.Usage()
        response = '{"natural_request":"Handle \\"PK-7\\" at .75."}'
        with mock.patch.object(driver, "call_api", return_value=response):
            result = driver.paraphrase_row(self.row(), "unused", "offline", usage, retries=0, temperature=0.0)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["style_card"]["name"], "terse_mobile")


if __name__ == "__main__":
    unittest.main()
