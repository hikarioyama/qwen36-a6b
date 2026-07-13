#!/usr/bin/env python3
"""CPU-only regression tests for :mod:`think_length_probe`."""
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
SPEC = importlib.util.spec_from_file_location("think_length_probe", ROOT / "esft" / "rl" / "think_length_probe.py")
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.status = 200

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        pass


def _seed(seed_id: str, tier: str) -> dict:
    return {"seed_id": seed_id, "tier": tier, "user_request": f"request {seed_id}",
            "tools": [{"type": "function", "name": "mock.call", "description": "mock",
                       "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                                      "required": ["value"], "additionalProperties": False}}],
            "expected_stages": [[{"name": "mock.call", "arguments": {"value": "ok"}}]]}


class ThinkLengthProbeTests(unittest.TestCase):
    def test_split_thinking_and_stage0_selection(self):
        seed = _seed("seed-a", "T2")
        valid = '{"calls":[{"name":"mock.call","arguments":{"value":"ok"}}]}'
        row = probe.score_rollout(seed, 3, "<think>abc</think>\n" + valid)
        self.assertEqual(row, {"seed_id": "seed-a", "tier": "T2", "rollout_idx": 3,
                               "think_chars": 3, "think_closed": True,
                               "total_chars": len("<think>abc</think>\n" + valid), "pass": True})
        chars, closed, payload = probe.split_thinking("<think>unfinished")
        self.assertEqual((chars, closed, payload), (10, False, "<think>unfinished"))

    def test_summary_reports_correct_distribution_and_per_seed_headroom(self):
        rows = [
            {"seed_id": "a", "tier": "T2", "rollout_idx": 0, "think_chars": 10, "think_closed": True,
             "total_chars": 20, "pass": True},
            {"seed_id": "a", "tier": "T2", "rollout_idx": 1, "think_chars": 100, "think_closed": True,
             "total_chars": 110, "pass": True},
            {"seed_id": "b", "tier": "T3", "rollout_idx": 0, "think_chars": 20, "think_closed": True,
             "total_chars": 30, "pass": True},
            {"seed_id": "b", "tier": "T3", "rollout_idx": 1, "think_chars": 20, "think_closed": True,
             "total_chars": 30, "pass": True},
            {"seed_id": "c", "tier": "T3", "rollout_idx": 0, "think_chars": 999, "think_closed": False,
             "total_chars": 999, "pass": False},
        ]
        summary = probe.summarize(rows)
        self.assertEqual((summary["rollouts"], summary["passes"]), (5, 4))
        self.assertEqual(summary["correct_think_chars"], {"count": 4, "p10": 13.0, "p50": 20.0, "p90": 76.0})
        self.assertEqual(summary["seed_correct_min_max_ratio"],
                         {"count": 2, "p10": 0.19, "p50": 0.55, "p90": 0.91})
        self.assertEqual(summary["direct_pruning_opportunity"],
                         {"eligible_seeds": 2, "seeds_with_shortest_below_half_median": 1, "rate": 0.5})

    def test_mock_transport_uses_thinking_prompt_and_fixed_completion_parameters(self):
        seeds = [_seed("t2", "T2"), _seed("t3", "T3")]
        requests = []

        def transport(request, timeout):
            self.assertEqual(timeout, 300)
            requests.append(request)
            if request.full_url.endswith("/v1/models"):
                return _Response({"data": [{"id": "mock-model"}]})
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["n"], 8)
            self.assertEqual(payload["temperature"], 1.0)
            self.assertEqual(payload["top_p"], 0.95)
            self.assertEqual(payload["max_tokens"], 4096)
            prompt = json.loads(payload["prompt"])
            self.assertEqual(prompt["turn"], 1)
            self.assertEqual(prompt["prior_tool_results"], [])
            valid = '{"calls":[{"name":"mock.call","arguments":{"value":"ok"}}]}'
            return _Response({"choices": [{"text": "<think>x</think>" + valid}] * 8})

        with mock.patch.object(probe.intent, "_load_frozen_seeds", return_value=(Path("."), {}, {"seeds": seeds})), \
             mock.patch.dict(os.environ, {"SELFGEN_NOTHINK": "", "SELFGEN_STAGE_HINT": ""}, clear=False):
            rows, summary = probe.run_probe(run_id="intent_r1", n=2, tiers=("T2", "T3"), sample_seed=7,
                                            endpoints="http://one,http://two", transport=transport,
                                            sleep=lambda _seconds: None)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["pass"] and row["think_chars"] == 1 for row in rows))
        self.assertEqual(summary["passes"], 16)
        self.assertEqual([request.full_url for request in requests],
                         ["http://one/v1/models", "http://two/v1/models",
                          "http://one/v1/completions", "http://two/v1/completions"])

    def test_cli_writes_jsonl_and_summary_without_gpu(self):
        seed = _seed("t2", "T2")
        valid = '{"calls":[{"name":"mock.call","arguments":{"value":"ok"}}]}'

        def transport(request, timeout):
            self.assertEqual(timeout, 300)
            if request.full_url.endswith("/v1/models"):
                return _Response({"data": [{"id": "mock-model"}]})
            return _Response({"choices": [{"text": valid}] * 8})

        real_run_probe = probe.run_probe
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(probe.intent, "_load_frozen_seeds", return_value=(Path(tmp), {}, {"seeds": [seed]})), \
             mock.patch.object(probe, "run_probe", wraps=lambda **kwargs: real_run_probe(
                 transport=transport, sleep=lambda _seconds: None, **kwargs)):
            output = Path(tmp) / "rows.jsonl"
            with mock.patch.dict(os.environ, {"SELFGEN_NOTHINK": "", "SELFGEN_STAGE_HINT": ""}, clear=False):
                # Restrict to a real frozen tier and call main; the mocked transport
                # is the only generation path exercised by this test.
                rc = probe.main(["--n", "1", "--tiers", "T2", "--endpoints", "http://one,http://two",
                                 "--output", str(output)])
            self.assertEqual(rc, 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 8)
            self.assertEqual(json.loads(output.with_suffix(".summary.json").read_text())["passes"], 8)


if __name__ == "__main__":
    unittest.main()
