#!/usr/bin/env python3
"""CPU regression tests for the intent-level selfgen seed fork."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import URLError
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "esft"))
SPEC = importlib.util.spec_from_file_location("selfgen_intent", ROOT / "esft" / "selfgen_toolcall_intent_v1.py")
assert SPEC and SPEC.loader
selfgen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selfgen
SPEC.loader.exec_module(selfgen)


class _FakeRows(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"
        self.prompt_batches: list[list[str]] = []

    def __call__(self, prompts, **_kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.prompt_batches.append(list(prompts))
        # This fixed batch width models left padding: decode must use the common
        # width rather than any individual prompt's unpadded length.
        return _FakeEncoding({"input_ids": _FakeRows([[0, 0, 0] for _ in prompts]),
                              "attention_mask": _FakeRows([[1, 1, 1] for _ in prompts])})

    def decode(self, tokens, **_kwargs):
        return "".join(tokens)


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    def __init__(self):
        self.empty_cache_calls = 0
        self.cuda = type("Cuda", (), {"empty_cache": self._empty_cache})()

    def _empty_cache(self):
        self.empty_cache_calls += 1

    def no_grad(self):
        return _FakeNoGrad()


class _FakeModel:
    def __init__(self, tokenizer, seeds, *, fail_seed_ids=(), fail_stages_by_seed=None, oom_attempts=0):
        self.tokenizer = tokenizer
        self.seeds = {seed["seed_id"]: seed for seed in seeds}
        self.fail_seed_ids = set(fail_seed_ids)
        self.fail_stages_by_seed = fail_stages_by_seed or {}
        self.oom_attempts = oom_attempts
        self.stage_by_seed = {seed["seed_id"]: 0 for seed in seeds}
        self.batch_sizes = []
        self.generate_kwargs = []

    def generate(self, **kwargs):
        self.batch_sizes.append(len(kwargs["input_ids"]))
        self.generate_kwargs.append(kwargs)
        if self.oom_attempts:
            self.oom_attempts -= 1
            raise RuntimeError("CUDA out of memory")
        rows = _FakeRows()
        for prompt in self.tokenizer.prompt_batches[-1]:
            request = json.loads(prompt)["user_request"]
            seed = next(seed for seed in self.seeds.values() if seed["user_request"] == request)
            stage = self.stage_by_seed[seed["seed_id"]]
            if (seed["seed_id"] in self.fail_seed_ids or
                    self.fail_stages_by_seed.get(seed["seed_id"]) == stage):
                candidates = ["not-json"] * kwargs["num_return_sequences"]
            else:
                candidates = [selfgen.canonical({"calls": seed["expected_stages"][stage]})]
                candidates.extend(["not-json"] * (kwargs["num_return_sequences"] - 1))
                self.stage_by_seed[seed["seed_id"]] += 1
            rows.extend([[0, 0, 0, candidate] for candidate in candidates])
        return rows


class _FakeOutput:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _HTTPResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        pass


class SelfgenToolcallIntentTests(unittest.TestCase):
    @staticmethod
    def _seeds(*numbers):
        return [selfgen.make_seed(number, random.Random(number), "T1") for number in numbers]

    def _run_worker(self, seeds, *, wave, fail_seed_ids=(), fail_stages_by_seed=None, oom_attempts=0):
        tokenizer, torch = _FakeTokenizer(), _FakeTorch()
        model = _FakeModel(tokenizer, seeds, fail_seed_ids=fail_seed_ids,
                           fail_stages_by_seed=fail_stages_by_seed, oom_attempts=oom_attempts)
        output = _FakeOutput()
        spec = selfgen.GenerationSpec(1, 0.7, 64, 4, 0)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(selfgen.v1, "load_stock_model", return_value=(torch, tokenizer, model)), \
             mock.patch.dict(os.environ, {"SELFGEN_WAVE": str(wave)}, clear=False):
            checkpoint = Path(tmp) / "generation_records_gpu0.jsonl"
            selfgen.generation_worker(0, seeds, spec, checkpoint, output)
            if output.items and output.items[-1]["kind"] == "error":
                self.fail(output.items[-1]["traceback"])
            records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
        return tokenizer, torch, model, output, records

    def test_vllm_choices_are_grouped_per_seed_and_stages_stay_serial(self):
        seeds = [selfgen.make_seed(0, random.Random(0), "T1"),
                 selfgen.make_seed(9, random.Random(9), "T4")]  # one stage and a four-stage long chain
        by_request = {seed["user_request"]: seed for seed in seeds}
        calls, active, max_active = [], 0, 0
        guard = threading.Lock()

        def transport(request, timeout):
            nonlocal active, max_active
            self.assertEqual(timeout, 300)
            if request.full_url.endswith("/v1/models"):
                return _HTTPResponse({"data": [{"id": "selfgen-test"}]})
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], "selfgen-test")
            self.assertEqual(payload["n"], 4)
            self.assertEqual(payload["top_p"], 0.95)
            prompt = json.loads(payload["prompt"])
            seed = by_request[prompt["user_request"]]
            stage = prompt["turn"] - 1
            with guard:
                active += 1
                max_active = max(max_active, active)
                calls.append((seed["seed_id"], stage))
            time.sleep(0.01)
            with guard:
                active -= 1
            valid = selfgen.canonical({"calls": seed["expected_stages"][stage]})
            return _HTTPResponse({"choices": [{"text": valid}] + [{"text": "not-json"}] * 3})

        append_active, concurrent_append = 0, False
        original_append = selfgen.v1.append_checkpoint_record

        def checked_append(path, record):
            nonlocal append_active, concurrent_append
            with guard:
                append_active += 1
                concurrent_append |= append_active > 1
            time.sleep(0.01)
            original_append(path, record)
            with guard:
                append_active -= 1

        output = _FakeOutput()
        spec = selfgen.GenerationSpec(1, 0.7, 64, 4, 0)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(selfgen.v1, "append_checkpoint_record", side_effect=checked_append), \
             mock.patch.object(selfgen.v1, "load_stock_model", side_effect=AssertionError("HF must stay unloaded")), \
             mock.patch.dict(os.environ, {"SELFGEN_INFLIGHT": "2"}, clear=False):
            checkpoint = Path(tmp) / "generation_records_gpu0.jsonl"
            selfgen.vllm_generation_worker(0, seeds, spec, checkpoint, output,
                                           "http://vllm.test", transport=transport, sleep=lambda _seconds: None)
            self.assertEqual(output.items, [{"kind": "complete", "gpu": 0, "records_written": 2}])
            records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
        self.assertEqual({record["seed"]["seed_id"] for record in records}, {seed["seed_id"] for seed in seeds})
        self.assertTrue(all(set(record) == {"seed", "selected", "results", "selection", "failures"}
                            for record in records))
        self.assertGreaterEqual(max_active, 2)  # independent seeds reached the endpoint concurrently
        self.assertFalse(concurrent_append)  # JSONL appends were held under the worker lock
        long_seed = seeds[1]
        self.assertEqual([stage for seed_id, stage in calls if seed_id == long_seed["seed_id"]], [0, 1, 2, 3])
        self.assertEqual(len(next(record for record in records
                                  if record["seed"]["seed_id"] == long_seed["seed_id"])["selected"]), 4)

    def test_vllm_retries_connection_failure_before_completion_success(self):
        attempts, sleeps = 0, []

        def transport(request, timeout):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise URLError("connection refused")
            self.assertTrue(request.full_url.endswith("/v1/completions"))
            return _HTTPResponse({"choices": [{"text": "a"}] * 4})

        client = selfgen._VLLMClient("http://vllm.test", model="selfgen-test",
                                     transport=transport, sleep=sleeps.append)
        raws = client.generate("prompt", selfgen.GenerationSpec(1, 0.7, 64, 4, 0))
        self.assertEqual(raws, ["a"] * 4)
        self.assertEqual(attempts, 2)
        self.assertEqual(sleeps, [1])

    def test_vllm_preflight_checks_endpoints_without_nvidia_smi(self):
        identity = {"revision": "test-stock"}
        seen = []

        def transport(request, timeout):
            seen.append(request.full_url)
            return _HTTPResponse({"data": [{"id": "selfgen-test"}]})

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(selfgen, "_load_frozen_seeds", return_value=(Path(tmp), {
                 "state": "prepared", "model": {"identity": identity, "topk": 8, "patch": None}}, {})), \
             mock.patch.object(selfgen.v1, "verified_stock_identity", return_value=identity), \
             mock.patch.object(selfgen.v1.subprocess, "run", side_effect=AssertionError("nvidia-smi must be skipped")):
            result = selfgen.gpu_preflight("intent-test", backend="vllm",
                                           endpoints="http://vllm0,http://vllm1", transport=transport,
                                           sleep=lambda _seconds: None)
        self.assertEqual(result["served_models"], ["selfgen-test", "selfgen-test"])
        self.assertEqual(seen, ["http://vllm0/v1/models", "http://vllm1/v1/models"])

    def test_wave_batch_maps_best_of_groups_in_prompt_order(self):
        seeds = self._seeds(0, 4)
        tokenizer, _torch, model, output, records = self._run_worker(seeds, wave=2)
        self.assertEqual(model.batch_sizes, [2])
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual([seed["seed_id"] for seed in seeds], [
            next(seed["seed_id"] for seed in seeds
                 if seed["user_request"] == json.loads(prompt)["user_request"])
            for prompt in tokenizer.prompt_batches[0]
        ])
        self.assertEqual([record["seed"]["seed_id"] for record in records], [seed["seed_id"] for seed in seeds])
        self.assertTrue(all(record["selection"][0]["candidate_index"] == 0 for record in records))
        self.assertIn("attention_mask", model.generate_kwargs[0])
        self.assertEqual(model.generate_kwargs[0]["num_return_sequences"], 4)
        self.assertEqual(output.items, [{"kind": "complete", "gpu": 0, "records_written": 2}])

    def test_failed_seed_checkpoint_record_keeps_v1_shape(self):
        failed, passed = self._seeds(2, 4)
        _tokenizer, _torch, _model, _output, records = self._run_worker(
            [failed, passed], wave=2, fail_stages_by_seed={failed["seed_id"]: 1})
        by_id = {record["seed"]["seed_id"]: record for record in records}
        expected_failures = selfgen.select_candidate(
            ["not-json"] * 4, failed["expected_stages"][1],
            {tool["name"]: tool for tool in failed["tools"]})[1]
        first_calls = failed["expected_stages"][0]
        first_results = [selfgen.mock_execute(call, 0, failed["pattern"]) for call in first_calls]
        self.assertEqual(by_id[failed["seed_id"]], {"seed": failed, "selected": [first_calls],
                                                       "results": [first_results],
                                                       "selection": [{"stage": 0, "candidate_index": 0,
                                                                      "candidate_rejections": []}],
                                                       "failures": expected_failures})
        self.assertEqual(set(by_id[passed["seed_id"]]), {"seed", "selected", "results", "selection", "failures"})

    def test_cuda_oom_halves_wave_and_retries(self):
        seeds = self._seeds(0, 4, 8, 12)
        _tokenizer, torch, model, _output, records = self._run_worker(seeds, wave=4, oom_attempts=1)
        self.assertEqual(model.batch_sizes[:2], [4, 2])
        self.assertEqual(torch.empty_cache_calls, 1)
        self.assertEqual(len(records), 4)

    def test_t1_preserves_v1_core_seed_exactly(self):
        baseline = selfgen.v1.make_seed(17, random.Random(123))
        intent = selfgen.make_seed(17, random.Random(123), "T1")
        # P0 intentionally adds semantic property descriptions; all executable
        # schema/tracing fields remain v1-compatible after removing that new
        # public metadata.
        baseline_tools = copy.deepcopy(baseline["tools"])
        intent_tools = copy.deepcopy(intent["tools"])
        for tool in intent_tools:
            for spec in tool["parameters"]["properties"].values():
                spec.pop("description", None)
        for key in ("seed_id", "schema_id", "domain", "pattern", "user_request", "expected_stages"):
            self.assertEqual(intent[key], baseline[key], key)
        self.assertEqual(intent_tools, baseline_tools, "tools")
        self.assertEqual(intent["natural_request"], baseline["user_request"])
        self.assertEqual(intent["distractor_tools"], [])

    def test_mock_name_style_is_byte_identical_to_existing_schema_names(self):
        baseline = selfgen.v1.make_seed(17, random.Random(123))
        styled = selfgen.make_seed(17, random.Random(123), "T1", name_style="mock")
        baseline_names = [(tool["name"], list(tool["parameters"]["properties"])) for tool in baseline["tools"]]
        styled_names = [(tool["name"], list(tool["parameters"]["properties"])) for tool in styled["tools"]]
        self.assertEqual(styled_names, baseline_names)

    def test_diverse_names_are_deterministic_and_all_styles_are_present(self):
        first = selfgen.make_seed(9, random.Random(5), "T4", name_style="diverse", eval_names=set())
        second = selfgen.make_seed(9, random.Random(5), "T4", name_style="diverse", eval_names=set())
        self.assertEqual(first["tools"], second["tools"])
        names = [tool["name"] for tool in first["tools"][:5]]
        # Short verb form is verb_noun — never a bare English word, which would
        # collide with natural request phrasing at the schema-leak gate.
        verb, _, noun = names[0].partition("_")
        self.assertIn(verb, selfgen.DIVERSE_VERBS)
        self.assertIn(noun, selfgen.DIVERSE_NOUNS)
        self.assertIn("_", names[1])  # snake_case
        self.assertNotRegex(names[2], r"[_.-]")  # camelCase (possibly with a numeric suffix)
        self.assertIn(".", names[3])  # dotted namespace
        self.assertIn("-server-", names[4])  # server-prefix form
        self.assertTrue(all(not name.startswith("field_") for tool in first["tools"]
                            for name in tool["parameters"]["properties"]))
        accepted, rejected, reasons = selfgen.evaluate_records(
            selfgen.cpu_fixture_records([first]), set(), set())
        self.assertEqual(len(accepted), 1, reasons)
        self.assertFalse(rejected)

    def test_diverse_name_rejects_injected_normalized_bfcl_name(self):
        # ``math.factorial`` is representative of the real-function spellings
        # guarded against by BFCL-v4 name loading; punctuation must not evade it.
        with mock.patch.object(selfgen, "_function_name_candidate",
                               side_effect=["math.factorial", "lookup_route"]):
            chosen = selfgen._choose_diverse_function_name(
                2, "math", random.Random(1),
                {selfgen.normalize_identifier("math.factorial")}, set())
        self.assertEqual(chosen, "lookup_route")

    def test_prepare_records_requested_name_style_in_manifest(self):
        original_root = selfgen.OUT_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(selfgen, "bfcl_function_names", return_value=frozenset()), \
                 mock.patch.object(selfgen.v1, "verified_stock_identity", return_value={"revision": "test"}), \
                 mock.patch.object(selfgen.v1, "bfcl_structural_profile", return_value={}), \
                 mock.patch.object(selfgen.v1, "contamination_corpus", return_value=({}, set(), set())):
                selfgen.OUT_ROOT = Path(tmp)
                args = selfgen.parser().parse_args([
                    "prepare", "--run-id", "diverse-test", "--n", "1", "--name-style", "diverse"])
                args.func(args)
                target = selfgen.run_dir("diverse-test")
                self.assertEqual(json.loads((target / "manifest.json").read_text())["name_style"], "diverse")
                self.assertEqual(json.loads((target / "seeds.json").read_text())["name_style"], "diverse")
        finally:
            selfgen.OUT_ROOT = original_root

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
        self.assertEqual(selfgen.tier_counts(10, mix), {"T1": 0, "T2": 5, "T3": 3, "T4": 2})
        seeds, rejected = selfgen.build_seeds(10, 44, {}, set(), set(), mix)
        self.assertFalse(rejected)
        self.assertEqual({tier: sum(seed["tier"] == tier for seed in seeds) for tier in selfgen.TIERS},
                         {"T1": 0, "T2": 5, "T3": 3, "T4": 2})

    def test_long_chain_fixture_remains_machine_verifiable(self):
        seed = selfgen.make_seed(9, random.Random(5), "T4")
        self.assertEqual(len(seed["expected_stages"]), 4)
        self.assertEqual(len(seed["expected_stages"][0]), 2)
        accepted, rejected, reasons = selfgen.evaluate_records(selfgen.cpu_fixture_records([seed]), set(), set())
        self.assertEqual(len(accepted), 1, reasons)
        self.assertFalse(rejected)

    def test_parallel_reverse_order_is_normalized_and_preserves_long_chain_links(self):
        seed = selfgen.make_seed(9, random.Random(5), "T4")
        expected = seed["expected_stages"][0]
        reversed_calls = list(reversed(copy.deepcopy(expected)))
        tool_map = {tool["name"]: tool for tool in seed["tools"]}
        detailed_calls, detailed_reasons, detailed_chosen, reordered = selfgen._select_candidate_details(
            [selfgen.canonical({"calls": reversed_calls})], expected, tool_map)
        self.assertEqual(detailed_calls, expected)
        self.assertEqual(detailed_reasons, [])
        self.assertEqual(detailed_chosen, 0)
        self.assertTrue(reordered)
        selected, reasons, chosen = selfgen.select_candidate(
            [selfgen.canonical({"calls": reversed_calls})], expected, tool_map)
        self.assertEqual(selected, expected)
        self.assertEqual(reasons, [])
        self.assertEqual(chosen, 0)

        all_selected = [selected] + copy.deepcopy(seed["expected_stages"][1:])
        results = [[selfgen.mock_execute(call, stage, seed["pattern"]) for call in calls]
                   for stage, calls in enumerate(all_selected)]
        self.assertTrue(selfgen._long_chain_links_match(seed, all_selected, results))
        record = {"seed": seed, "selected": all_selected, "results": results,
                  "selection": [{"stage": 0, "candidate_index": 0, "candidate_rejections": [],
                                 "parallel_reordered": True}] +
                               [{"stage": stage, "candidate_index": 0, "candidate_rejections": []}
                                for stage in range(1, len(all_selected))],
                  "failures": []}
        accepted, rejected, reasons = selfgen.evaluate_records([record], set(), set())
        self.assertEqual(len(accepted), 1, reasons)
        self.assertFalse(rejected)
        self.assertTrue(accepted[0]["metadata"]["selection"][0]["parallel_reordered"])

    def test_parallel_different_call_remains_plan_alignment_rejection(self):
        seed = selfgen.make_seed(9, random.Random(5), "T4")
        expected = seed["expected_stages"][0]
        wrong = list(reversed(copy.deepcopy(expected)))
        key, value = next(iter(wrong[0]["arguments"].items()))
        if isinstance(value, str):
            wrong[0]["arguments"][key] = value + "-different"
        elif isinstance(value, bool):
            wrong[0]["arguments"][key] = not value
        elif isinstance(value, int):
            wrong[0]["arguments"][key] = value + 1
        elif isinstance(value, float):
            wrong[0]["arguments"][key] = value + 1.0
        else:
            self.fail(f"unexpected fixture argument type: {type(value)!r}")
        tool_map = {tool["name"]: tool for tool in seed["tools"]}
        calls, reasons, chosen = selfgen.select_candidate(
            [selfgen.canonical({"calls": wrong})], expected, tool_map)
        self.assertIsNone(calls)
        self.assertIsNone(chosen)
        self.assertEqual(reasons, ["plan_alignment"])

    def test_retry_failed_requeues_only_failed_checkpoint_records(self):
        failed, succeeded = self._seeds(0, 4)
        records = selfgen.cpu_fixture_records([failed, succeeded])
        records[0]["failures"] = ["plan_alignment"]
        completed = {record["seed"]["seed_id"]: record for record in records}
        self.assertEqual(selfgen.pending_seeds([failed, succeeded], completed), [])
        self.assertEqual(selfgen.pending_seeds([failed, succeeded], completed, retry_failed=True), [failed])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            selfgen.v1.append_checkpoint_record(selfgen.v1.checkpoint_path(target, 1), records[0])
            selfgen.v1.append_checkpoint_record(selfgen.v1.checkpoint_path(target, 0), records[1])
            self.assertEqual(selfgen._retry_partitions(target, [failed], completed),
                             {0: [], 1: [failed]})
        args = selfgen.parser().parse_args(["execute", "--run-id", "intent-test", "--retry-failed"])
        self.assertTrue(args.retry_failed)
        self.assertEqual(args.backend, "hf")
        self.assertIsNone(args.endpoints)

    def test_checkpoint_retry_deduplicates_last_record_before_completeness_check(self):
        seeds = self._seeds(0, 4)
        failed, succeeded = selfgen.cpu_fixture_records(seeds)
        failed["failures"] = ["plan_alignment"]
        retried = copy.deepcopy(failed)
        retried["failures"] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            checkpoint = selfgen.v1.checkpoint_path(target, 0)
            selfgen.v1.append_checkpoint_record(checkpoint, failed)
            selfgen.v1.append_checkpoint_record(checkpoint, retried)
            selfgen.v1.append_checkpoint_record(checkpoint, succeeded)
            loaded = selfgen.load_checkpoint_records(target, seeds)
            self.assertEqual(loaded[failed["seed"]["seed_id"]], retried)
            records = selfgen._complete_records_or_raise(target, seeds)
        self.assertEqual(len(records), len(seeds))
        self.assertFalse(next(record for record in records
                              if record["seed"]["seed_id"] == failed["seed"]["seed_id"])["failures"])

    def test_paraphrase_ingest_quarantines_failed_writeback_without_fallback(self):
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
                frozen = json.loads((target / "seeds.json").read_text())
                self.assertEqual(frozen["seeds"], [])
                self.assertEqual(frozen["paraphrase_result"], {"excluded": 1})
                quarantined = json.loads((target / "paraphrase_excluded.jsonl").read_text())
                self.assertEqual(quarantined["seed_id"], seed["seed_id"])
                self.assertNotIn("fallback", str(quarantined))
        finally:
            selfgen.OUT_ROOT = original_root

    def test_paraphrase_ingest_excludes_unparaphrased_t1_seed(self):
        original_root = selfgen.OUT_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                selfgen.OUT_ROOT = Path(tmp)
                target = selfgen.run_dir("intent-test")
                target.mkdir()
                seed = selfgen.make_seed(8, random.Random(5), "T1")
                selfgen.atomic_json(target / "manifest.json", {"state": "prepared"})
                selfgen.atomic_json(target / "seeds.json", {"seeds": [seed]})
                writeback = target / "writeback.jsonl"
                writeback.write_text("")
                selfgen.ingest_paraphrase(type("Args", (), {"run_id": "intent-test", "input": str(writeback)})())
                self.assertEqual(json.loads((target / "seeds.json").read_text())["seeds"], [])
                quarantined = json.loads((target / "paraphrase_excluded.jsonl").read_text())
                self.assertIn("request", quarantined["failures"])
        finally:
            selfgen.OUT_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
