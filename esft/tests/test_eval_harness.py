from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ESFT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ESFT_ROOT))

import codex_harness
import eval_harness


class EvalHarnessTests(unittest.TestCase):
    def test_item_key_is_content_derived_and_ignores_runtime_fields(self):
        bench = eval_harness.Benchmark()
        first = {"question": "q", "gold": "A", "_id": 0}
        second = {"gold": "A", "question": "q", "_id": 99}
        changed = {"question": "different", "gold": "A", "_id": 0}

        self.assertEqual(bench.item_key(first), bench.item_key(second))
        self.assertNotEqual(bench.item_key(first), bench.item_key(changed))

    def test_paired_verdict_uses_stable_keys_not_position_ids(self):
        items_a = [
            {"id": 0, "item_key": "alpha", "correct": True},
            {"id": 1, "item_key": "beta", "correct": False},
        ]
        items_b = [
            {"id": 0, "item_key": "beta", "correct": True},
            {"id": 1, "item_key": "alpha", "correct": True},
        ]

        verdict = eval_harness.paired_verdict(items_a, items_b)

        self.assertEqual(verdict["n"], 2)
        self.assertEqual(verdict["acc_a"], 0.5)
        self.assertEqual(verdict["acc_b"], 1.0)
        self.assertEqual(verdict["n01"], 1)
        self.assertEqual(verdict["n10"], 0)

    def test_paired_verdict_rejects_different_content_even_when_ids_match(self):
        items_a = [{"id": 0, "item_key": "alpha", "correct": True}]
        items_b = [{"id": 0, "item_key": "different", "correct": True}]

        with self.assertRaisesRegex(ValueError, "item id sets differ"):
            eval_harness.paired_verdict(items_a, items_b)

    def test_paired_verdict_rejects_stable_to_legacy_mix(self):
        stable = [{"id": 0, "item_key": "alpha", "correct": True}]
        legacy = [{"id": 0, "correct": True}]

        with self.assertRaisesRegex(ValueError, "one result uses stable item keys"):
            eval_harness.paired_verdict(stable, legacy)

    def test_protocol_validation_rejects_shuffle_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shuffle"):
            eval_harness._validate_paired_protocol(
                {"seed": 0, "shuffle": False},
                {"seed": 0, "shuffle": True},
            )

    def test_complete_protocol_metadata_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            eval_harness._validate_paired_protocol({}, {})

    def test_file_verdict_checks_protocol_before_legacy_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write(name: str, shuffle: bool) -> Path:
                path = root / name
                path.write_text(json.dumps({
                    "_meta": {"seed": 0, "shuffle": shuffle,
                              "n_per_benchmark": 1},
                    "items": {"mmlu": [{"id": 0, "correct": True}]},
                }))
                return path

            with self.assertRaisesRegex(ValueError, "identical evaluation protocols"):
                eval_harness.run_paired_verdict(
                    write("a.json", False),
                    write("b.json", True),
                )

    def test_codex_campaign_commands_are_serial_arm_equivalents(self):
        cfg = codex_harness.load_config(codex_harness.DEFAULT_CONFIG)
        run_dir = Path("/tmp/example-run")
        base, base_tag = codex_harness.eval_command(
            cfg, "humaneval", "base_k8", run_dir)
        b2, b2_tag = codex_harness.eval_command(
            cfg, "humaneval", "b2_k32", run_dir)

        self.assertEqual(base_tag, "base_k8_humaneval")
        self.assertEqual(b2_tag, "b2_k32_humaneval")
        self.assertNotIn("--patch", base)
        self.assertEqual(
            b2[b2.index("--patch") + 1], cfg["patches"]["b2"]["path"])
        self.assertEqual(base[base.index("--topk") + 1], "8")
        self.assertEqual(b2[b2.index("--topk") + 1], "32")
        for flag in ("--n", "--seed", "--shuffle", "--batch-size", "--max-new"):
            self.assertIn(flag, base)
            self.assertIn(flag, b2)

    def test_mmlu_protocol_forces_choice_logprob(self):
        cfg = codex_harness.load_config(codex_harness.DEFAULT_CONFIG)
        args = codex_harness.protocol_args(cfg["protocols"]["mmlu"])

        self.assertIn("--choice-logprob", args)
        self.assertIn("--shuffle", args)
        self.assertIn("--no-think", args)
        self.assertNotIn("--max-new", args)

    def test_generation_cap_requires_missing_eos(self):
        eos = eval_harness.EOS_IDS[0]

        self.assertTrue(eval_harness._hit_generation_cap([1, 2, 3, 4], 4))
        self.assertFalse(eval_harness._hit_generation_cap([1, eos, 0, 0], 4))
        self.assertFalse(eval_harness._hit_generation_cap([1, 2, 3], 4))

    def test_noninferiority_is_three_way(self):
        # With 1,000 identical outcomes, exact bounds rule out a one-point loss.
        self.assertEqual(eval_harness.noninferiority_verdict(
            {"n": 1000, "n10": 0, "n01": 0, "delta": 0.0}, 0.01)["status"],
            "PASS")
        # A clear 100-item one-way discordance is a definite regression.
        self.assertEqual(eval_harness.noninferiority_verdict(
            {"n": 1000, "n10": 100, "n01": 0, "delta": -0.1}, 0.01)["status"],
            "FAIL")
        # One observation with no discordance must not produce a zero-width PASS.
        self.assertEqual(eval_harness.noninferiority_verdict(
            {"n": 1, "n10": 0, "n01": 0, "delta": 0.0}, 0.01)["status"],
            "INCONCLUSIVE")

    def test_paired_exact_bounds_cover_observed_mmlu_uncertainty(self):
        lower, upper = eval_harness.paired_difference_exact_bounds(600, 20, 17)

        self.assertLess(lower, -0.01)
        self.assertGreater(upper, 0.0)

    def test_noninferiority_rejects_invalid_margins_and_counts(self):
        verdict = {"n": 10, "n10": 0, "n01": 0, "delta": 0.0}
        for margin in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                eval_harness.noninferiority_verdict(verdict, margin)
        with self.assertRaisesRegex(ValueError, "paired counts"):
            eval_harness.paired_difference_exact_bounds(10.5, 0, 0)

    def test_physical_gpu_selection_is_remapped_to_logical_pair(self):
        cfg = copy.deepcopy(codex_harness.load_config(codex_harness.DEFAULT_CONFIG))
        cfg["runtime"]["gpus"] = "1,2"

        command, _ = codex_harness.eval_command(
            cfg, "mmlu", "base_k8", Path("/tmp/example-run"))
        env = codex_harness.command_env(cfg)

        self.assertEqual(command[command.index("--gpus") + 1], "0,1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,2")

    def test_arm_output_validation_checks_counts_and_identity(self):
        cfg = copy.deepcopy(codex_harness.load_config(codex_harness.DEFAULT_CONFIG))
        cfg["protocols"]["mmlu"]["n"] = 2
        source_hashes = codex_harness.evaluation_source_hashes()
        meta = {
            "model": "base",
            "model_path": cfg["stock"]["path"],
            "patch": None,
            "topk": 8,
            "benchmarks": ["mmlu"],
            "n_per_benchmark": 2,
            "batch_size": 16,
            "max_new": None,
            "seed": 0,
            "shuffle": True,
            "gpus": [0, 1],
            "no_think": True,
            "choice_logprob": True,
            "effective_prompt_modes": {"mmlu": "choice_logprob_no_think"},
            "split": "first-N shuffled",
            "harness_sha256": source_hashes["eval_harness.py"],
            "source_sha256": source_hashes,
            "python_executable": cfg["runtime"]["python"],
            "python_version": sys.version,
            "package_versions": codex_harness.package_versions(),
        }
        items = [
            {"id": 0, "item_key": "a", "correct": True, "truncated": False},
            {"id": 1, "item_key": "b", "correct": False, "truncated": False},
        ]
        report = {
            "_meta": meta,
            "results": {"mmlu": {
                "model": "base", "topk": 8, "benchmark": "mmlu", "n": 2,
                "max_new": None, "correct": 1, "truncated_n": 0,
            }},
        }
        item_report = {"_meta": meta, "items": {"mmlu": items}}
        audit = {
            "eval_harness": {
                "sha256": source_hashes["eval_harness.py"],
                "source_sha256": source_hashes,
            },
            "python": {
                "version": sys.version,
                "packages": codex_harness.package_versions(),
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            items_path = Path(tmp) / "items.json"
            result_path.write_text(json.dumps(report))
            items_path.write_text(json.dumps(item_report))
            _, validation = codex_harness.validate_arm_output(
                cfg, audit, "mmlu", "base_k8", result_path, items_path)

        self.assertEqual(validation["item_count"], 2)
        self.assertEqual(validation["unique_item_keys"], 2)


if __name__ == "__main__":
    unittest.main()
