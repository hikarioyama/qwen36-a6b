from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "remote" / "run_b2_1000_eval.py"
SPEC = importlib.util.spec_from_file_location("run_b2_1000_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def fake_config(root: Path) -> dict:
    return {
        "run": {"root": str(root)},
        "runtime": {
            "python": "/runtime/python",
            "hf_home": "/cache",
            "python_path": "/deps/python",
            "bubblewrap_bin_dir": "/deps/bin",
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "thread_count": 8,
            "expected_gpu_uuids": copy.deepcopy(campaign.EXPECTED_GPU_UUIDS),
        },
        "stock": {"path": "/models/stock"},
        "patches": {"b2": {"path": "/models/b2.safetensors"}},
        "gpu_pairs": copy.deepcopy(campaign.EXPECTED_GPU_PAIRS),
        "protocols": copy.deepcopy(campaign.EXPECTED_PROTOCOLS),
        "dataset_hashes": copy.deepcopy(campaign.EXPECTED_DATASET_HASHES),
    }


class B21000CampaignTests(unittest.TestCase):
    def test_commands_use_logical_gpus_and_frozen_physical_env(self):
        cfg = fake_config(Path("/tmp/example"))

        command = campaign.eval_command(cfg, "b2_1000_k32", "gsm8k")
        env = campaign.child_environment(cfg, "gsm8k")

        self.assertEqual(command[command.index("--gpus") + 1], "0,1")
        self.assertEqual(command[command.index("--topk") + 1], "32")
        self.assertEqual(command[command.index("--patch") + 1], "/models/b2.safetensors")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(
            campaign.EXPECTED_GPU_UUIDS[str(index)] for index in (2, 3)))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], "8")

    def test_base_commands_never_receive_patch(self):
        cfg = fake_config(Path("/tmp/example"))
        for benchmark in campaign.BENCHMARKS:
            with self.subTest(benchmark=benchmark):
                command = campaign.eval_command(cfg, "base_k8", benchmark)
                self.assertNotIn("--patch", command)
                self.assertEqual(command[command.index("--topk") + 1], "8")

    def test_humaneval_uses_all_four_assigned_gpus_logically(self):
        cfg = fake_config(Path("/tmp/example"))
        command = campaign.eval_command(cfg, "base_k8", "humaneval")
        env = campaign.child_environment(cfg, "humaneval")

        self.assertEqual(command[command.index("--gpus") + 1], "0,1,2,3")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(
            campaign.EXPECTED_GPU_UUIDS[str(index)] for index in (4, 5, 6, 7)))

    def test_strict_item_validation_checks_types_order_and_dataset_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fake_config(root)
            cfg["protocols"]["mmlu"]["n"] = 2
            keys = ["a" * 64, "b" * 64]
            items = [
                {"id": 0, "item_key": keys[0], "correct": True,
                 "truncated": False, "sandbox_timeout": False, "gen_len": 0},
                {"id": 1, "item_key": keys[1], "correct": False,
                 "truncated": False, "sandbox_timeout": False, "gen_len": 0},
            ]
            result_path = root / "result.json"
            items_path = root / "items.json"
            result_path.write_text(json.dumps({"results": {"mmlu": {
                "correct": 1, "truncated_n": 0, "sandbox_timeout_n": 0,
                "acc": 0.5,
            }}}))
            items_path.write_text(json.dumps({"items": {"mmlu": items}}))
            digest = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()

            validation = campaign.validate_item_details(
                cfg, "mmlu", result_path, items_path,
                {"ordered_item_key_sha256": digest}, 0,
            )
            self.assertTrue(validation["typed_items"])

            items[0]["correct"] = "true"
            items_path.write_text(json.dumps({"items": {"mmlu": items}}))
            with self.assertRaisesRegex(campaign.CampaignError, "not boolean"):
                campaign.validate_item_details(
                    cfg, "mmlu", result_path, items_path,
                    {"ordered_item_key_sha256": digest}, 0,
                )

    def test_paired_decision_separates_measurement_success_from_adoption(self):
        cfg = fake_config(Path("/tmp/example"))
        item = {"item_key": "a" * 64, "sandbox_timeout": False}

        def verdict(status):
            return {
                "correct": {
                    "noninferiority": {"status": status},
                    "delta": 0.0, "significant": False,
                },
                "truncated": {"delta": 0.0, "significant": False},
            }

        with mock.patch.object(
                campaign.base_harness, "structured_verdicts",
                side_effect=[verdict("PASS"), verdict("PASS"), verdict("PASS")]), \
                mock.patch.object(
                    campaign.eval_harness, "_load_items_file",
                    return_value=({"humaneval": [item]}, {})):
            _verdicts, decision = campaign.paired_results(cfg)

        self.assertEqual(decision["disposition"], "AUTO_ADOPT_B2_1000")
        self.assertTrue(decision["automatic_adoption"])

    def test_campaign_lock_does_not_truncate_before_acquiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            path.write_text("old\n")
            with campaign.campaign_lock(path):
                self.assertEqual(path.read_text(), f"pid={os.getpid()}\n")


if __name__ == "__main__":
    unittest.main()
