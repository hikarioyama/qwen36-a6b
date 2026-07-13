#!/usr/bin/env python3
"""CPU-only tests for INC-0 short-thinking rejection selection and SFT output."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "esft"))
SPEC = importlib.util.spec_from_file_location("inc0_shortthink_v1", ROOT / "esft" / "rl" / "inc0_shortthink_v1.py")
assert SPEC and SPEC.loader
inc0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inc0
SPEC.loader.exec_module(inc0)


def _seed() -> dict:
    return {"seed_id": "seed-a", "tier": "T2", "user_request": "Call the mock tool.",
            "tools": [{"type": "function", "name": "mock.call", "description": "mock",
                       "parameters": {"type": "object", "properties": {}, "required": []}}],
            "expected_stages": [[{"name": "mock.call", "arguments": {}}]]}


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        pass


class Inc0ShortThinkTests(unittest.TestCase):
    def test_shortest_closed_pass_wins_and_index_breaks_tie(self):
        rows = [
            {"rollout_idx": 0, "pass": True, "think_closed": False, "think_chars": 0},
            {"rollout_idx": 3, "pass": True, "think_closed": True, "think_chars": 5},
            {"rollout_idx": 2, "pass": True, "think_closed": True, "think_chars": 5},
            {"rollout_idx": 1, "pass": True, "think_closed": True, "think_chars": 9},
        ]
        self.assertEqual(inc0.select_shortest_closed(rows)["rollout_idx"], 2)
        self.assertIsNone(inc0.select_shortest_closed([rows[0], {"rollout_idx": 4, "pass": False,
                                                                  "think_closed": True, "think_chars": 1}]))

    def test_trainer_conversion_retains_closed_think_in_assistant_turn(self):
        raw = '<think>brief reasoning</think>{"calls":[{"name":"mock.call","arguments":{}}]}'
        record = {"seed_key": "intent_r1:seed-a", "seed": _seed(),
                  "selected": {"rollout_idx": 0, "pass": True, "think_closed": True,
                               "think_chars": 15, "raw": raw}}
        trainer = inc0.make_trainer_record(record)
        self.assertEqual(trainer["_source"], "inc0_shortthink_20260714")
        self.assertEqual(trainer["_domain"], "toolcall")
        assistant = next(message for message in trainer["messages"] if message["role"] == "assistant")
        self.assertEqual(assistant["content"], raw)
        self.assertIn("# Tools", trainer["messages"][0]["content"])

    def test_summary_reports_per_tier_adoption_pass_and_zero_rate(self):
        selected = {"rollout_idx": 1, "pass": True, "think_closed": True, "think_chars": 0, "raw": "<think></think>{}"}
        record = {"tier": "T1", "rollouts": [
            {"pass": True, "think_closed": True, "think_chars": 0},
            {"pass": False, "think_closed": False, "think_chars": 8}], "selected": selected}
        summary = inc0.summarize([record], tiers=("T1", "T2"))
        self.assertEqual(summary["by_tier"]["T1"]["adopted"], 1)
        self.assertEqual(summary["by_tier"]["T1"]["pass_rate"], 0.5)
        self.assertEqual(summary["by_tier"]["T1"]["adopted_zero_think_rate"], 1.0)
        self.assertIsNone(summary["by_tier"]["T2"]["adopted_zero_think_rate"])

    def test_checkpoint_resume_skips_completed_seed_groups(self):
        seeds = [{**_seed(), "seed_id": "t1", "tier": "T1"}, {**_seed(), "seed_id": "t2", "tier": "T2"}]
        calls = []
        valid = '<think></think>{"calls":[{"name":"mock.call","arguments":{}}]}'

        def transport(request, timeout):
            calls.append(request.full_url)
            if request.full_url.endswith("/v1/models"):
                return _Response({"data": [{"id": "mock-model"}]})
            return _Response({"choices": [{"text": valid}] * 8})

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(inc0.intent, "_load_frozen_seeds", return_value=(Path(tmp), {}, {"seeds": seeds})), \
             mock.patch.dict(os.environ, {"SELFGEN_NOTHINK": "", "SELFGEN_STAGE_HINT": ""}, clear=False):
            output = Path(tmp) / "rollouts.jsonl"
            first = inc0.run_campaign(run_ids=("intent_r1",), tiers=("T1", "T2"), n_per_tier=1,
                                      sample_seed=7, endpoints="http://one,http://two", rollout_output=output,
                                      transport=transport, sleep=lambda _seconds: None)
            seen_after_first = len(calls)
            second = inc0.run_campaign(run_ids=("intent_r1",), tiers=("T1", "T2"), n_per_tier=1,
                                       sample_seed=7, endpoints="http://one,http://two", rollout_output=output,
                                       transport=transport, sleep=lambda _seconds: None)
            checkpoint_rows = len(output.read_text(encoding="utf-8").splitlines())
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(seen_after_first, len(calls))
        self.assertEqual(checkpoint_rows, 2)


if __name__ == "__main__":
    unittest.main()
