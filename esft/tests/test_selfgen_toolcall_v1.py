#!/usr/bin/env python3
"""CPU tests for selfgen v1's seed, parser, executor and scaffold stripping."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import multiprocessing as mp
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("selfgen", ROOT / "esft" / "selfgen_toolcall_v1.py")
assert SPEC and SPEC.loader
selfgen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selfgen
SPEC.loader.exec_module(selfgen)


class SelfgenToolcallTests(unittest.TestCase):
    def setUp(self):
        self.seed = selfgen.make_seed(2, __import__("random").Random(9))
        self.tool_map = {x["name"]: x for x in self.seed["tools"]}

    def test_expected_calls_validate_and_executor_is_deterministic(self):
        for stage, calls in enumerate(self.seed["expected_stages"]):
            for call in calls:
                self.assertEqual(selfgen.validate_call(call, self.tool_map)[0], None)
                self.assertEqual(selfgen.mock_execute(call, stage, self.seed["pattern"]),
                                 selfgen.mock_execute(call, stage, self.seed["pattern"]))

    def test_bad_json_and_undeclared_call_are_rejected(self):
        self.assertEqual(selfgen.parse_model_turn("not json", self.tool_map)[1], "json_parse")
        self.assertEqual(selfgen.parse_model_turn('{"calls":[{"name":"stolen","arguments":{}}]}', self.tool_map)[1],
                         "undeclared_function")

    def test_plan_alignment_and_scaffold_distillation(self):
        expected = self.seed["expected_stages"][0]
        raw = selfgen.canonical({"calls": expected})
        calls, reasons, selected = selfgen.select_candidate(["{}", raw], expected, self.tool_map)
        self.assertEqual(calls, expected)
        self.assertEqual(selected, 1)
        self.assertEqual(reasons, ["turn_shape"])
        record = selfgen.render_training(self.seed, [expected], [[selfgen.mock_execute(expected[0], 0, self.seed["pattern"])]] )
        dumped = selfgen.canonical(record)
        self.assertNotIn("few_shots", dumped)
        self.assertNotIn("You are a precise", dumped)
        self.assertEqual(record["messages"][0]["role"], "user")

    def test_fixture_validation_never_writes_training_output(self):
        seeds = [selfgen.make_seed(i, __import__("random").Random(i)) for i in range(4)]
        records = selfgen.cpu_fixture_records(seeds)
        accepted, rejected, reasons = selfgen.evaluate_records(records, set(), set())
        self.assertEqual(len(accepted), 4)
        self.assertFalse(rejected)
        self.assertFalse(reasons)

    def test_schema_pool_is_approximately_200_and_patterns_are_balanced(self):
        seeds = [selfgen.make_seed(i, __import__("random").Random(i)) for i in range(500)]
        self.assertEqual(len({seed["schema_id"] for seed in seeds}), 197)
        self.assertEqual({pattern: sum(seed["pattern"] == pattern for seed in seeds)
                          for pattern in selfgen.PATTERNS}, {pattern: 125 for pattern in selfgen.PATTERNS})

    def test_checkpoint_resume_skips_only_durably_completed_task_ids(self):
        seeds = [selfgen.make_seed(i, __import__("random").Random(i)) for i in range(4)]
        records = selfgen.cpu_fixture_records(seeds)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            selfgen.append_checkpoint_record(selfgen.checkpoint_path(target, 0), records[0])
            selfgen.append_checkpoint_record(selfgen.checkpoint_path(target, 1), records[3])
            completed = selfgen.load_checkpoint_records(target, seeds)
        self.assertEqual(set(completed), {seeds[0]["seed_id"], seeds[3]["seed_id"]})
        self.assertEqual([seed["seed_id"] for seed in selfgen.pending_seeds(seeds, completed)],
                         [seeds[1]["seed_id"], seeds[2]["seed_id"]])

    def test_worker_exception_is_returned_as_an_error_sentinel(self):
        output: mp.Queue = mp.Queue()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(selfgen, "load_stock_model", side_effect=RuntimeError("test boom")):
                    selfgen.generation_worker(
                        0, [], selfgen.GenerationSpec(1, 0.7, 32, 4, 0),
                        Path(tmp) / "checkpoint.jsonl", output)
            item = output.get(timeout=5)
        finally:
            output.close()
            output.join_thread()
        self.assertEqual(item["kind"], "error")
        self.assertEqual(item["gpu"], 0)
        self.assertIn("RuntimeError: test boom", item["error"])
        self.assertIn("test boom", item["traceback"])

    def test_wait_loop_accepts_terminal_sentinels_without_a_fixed_deadline(self):
        class FakeWorker:
            def __init__(self, pid):
                self.pid, self.exitcode = pid, None

            def is_alive(self):
                return True

        class FakeOutput:
            def __init__(self):
                self.items = [{"kind": "complete", "gpu": 0, "records_written": 1},
                              {"kind": "complete", "gpu": 1, "records_written": 1}]

            def get_nowait(self):
                return self.items.pop(0)

        selfgen.wait_for_worker_completion([FakeWorker(10), FakeWorker(11)], FakeOutput())

    def test_wait_loop_fails_when_workers_die_without_sentinels(self):
        class DeadWorker:
            def __init__(self, pid, exitcode):
                self.pid, self.exitcode = pid, exitcode

            def is_alive(self):
                return False

        class EmptyOutput:
            def get_nowait(self):
                raise selfgen.queue.Empty

        with mock.patch.object(selfgen, "WORKER_SENTINEL_GRACE_SECONDS", 0), \
                mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "gpu0 exitcode=17, gpu1 exitcode=18"):
                selfgen.wait_for_worker_completion(
                    [DeadWorker(10, 17), DeadWorker(11, 18)], EmptyOutput())


if __name__ == "__main__":
    unittest.main()
