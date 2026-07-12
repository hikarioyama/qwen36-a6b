#!/usr/bin/env python3
"""CPU regression tests for the intent-level selfgen seed fork."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "esft"))
SPEC = importlib.util.spec_from_file_location("selfgen_intent", ROOT / "esft" / "selfgen_toolcall_intent_v1.py")
assert SPEC and SPEC.loader
selfgen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selfgen
SPEC.loader.exec_module(selfgen)


class SelfgenToolcallIntentTests(unittest.TestCase):
    def test_t1_preserves_v1_core_seed_exactly(self):
        baseline = selfgen.v1.make_seed(17, random.Random(123))
        intent = selfgen.make_seed(17, random.Random(123), "T1")
        for key in ("seed_id", "schema_id", "domain", "pattern", "tools", "user_request", "expected_stages"):
            self.assertEqual(intent[key], baseline[key], key)
        self.assertEqual(intent["natural_request"], baseline["user_request"])
        self.assertEqual(intent["distractor_tools"], [])

    def test_intent_tiers_surface_every_static_value_without_call_structure(self):
        for tier in ("T2", "T3", "T4"):
            with self.subTest(tier=tier):
                seed = selfgen.make_seed(24, random.Random(27), tier)
                passed, missing = selfgen.validate_value_occurrences(seed)
                self.assertTrue(passed, missing)
                request = seed["natural_request"]
                for stage in seed["expected_stages"]:
                    for call in stage:
                        self.assertNotIn(call["name"], request)
                        for field in call["arguments"]:
                            self.assertNotIn(field, request)

    def test_distractor_call_cannot_be_selected_as_expected_trace(self):
        seed = selfgen.make_seed(31, random.Random(8), "T3")
        expected = seed["expected_stages"][0]
        distractor = seed["distractor_tools"][0]
        invalid = {"name": distractor["name"], "arguments": expected[0]["arguments"]}
        tool_map = {tool["name"]: tool for tool in seed["tools"]}
        calls, reasons, chosen = selfgen.select_candidate(
            [selfgen.canonical({"calls": [invalid]})], expected, tool_map)
        self.assertIsNone(calls)
        self.assertIsNone(chosen)
        self.assertNotEqual(reasons, [])

    def test_default_tier_mix_has_exact_largest_remainder_allocation(self):
        mix = selfgen.parse_tier_mix(selfgen.DEFAULT_TIER_MIX)
        self.assertEqual(selfgen.tier_counts(10, mix), {"T1": 1, "T2": 4, "T3": 3, "T4": 2})
        seeds, rejected = selfgen.build_seeds(10, 44, {}, set(), set(), mix)
        self.assertFalse(rejected)
        self.assertEqual({tier: sum(seed["tier"] == tier for seed in seeds) for tier in selfgen.TIERS},
                         {"T1": 1, "T2": 4, "T3": 3, "T4": 2})

    def test_long_chain_fixture_remains_machine_verifiable(self):
        seed = selfgen.make_seed(9, random.Random(5), "T4")
        self.assertEqual(len(seed["expected_stages"]), 4)
        self.assertEqual(len(seed["expected_stages"][0]), 2)
        accepted, rejected, reasons = selfgen.evaluate_records(selfgen.cpu_fixture_records([seed]), set(), set())
        self.assertEqual(len(accepted), 1, reasons)
        self.assertFalse(rejected)

    def test_paraphrase_ingest_falls_back_and_records_tier_downgrade(self):
        original_root = selfgen.OUT_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                selfgen.OUT_ROOT = Path(tmp)
                target = selfgen.run_dir("intent-test")
                target.mkdir()
                seed = selfgen.make_seed(7, random.Random(4), "T2")
                selfgen.atomic_json(target / "manifest.json", {"state": "prepared"})
                selfgen.atomic_json(target / "seeds.json", {"seeds": [seed]})
                batch = target / "batch.jsonl"
                selfgen.emit_paraphrase_batch(type("Args", (), {"run_id": "intent-test", "output": str(batch)})())
                emitted = json.loads(batch.read_text().strip())
                self.assertEqual(emitted["seed_id"], seed["seed_id"])
                writeback = target / "writeback.jsonl"
                writeback.write_text(json.dumps({"seed_id": seed["seed_id"], "natural_request": "missing values"}) + "\n")
                selfgen.ingest_paraphrase(type("Args", (), {"run_id": "intent-test", "input": str(writeback)})())
                ingested = json.loads((target / "seeds.json").read_text())["seeds"][0]
                self.assertEqual(ingested["tier"], "T1")
                self.assertEqual(ingested["tier_original"], "T2")
                self.assertEqual(ingested["user_request"], ingested["transcription_request"])
        finally:
            selfgen.OUT_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
