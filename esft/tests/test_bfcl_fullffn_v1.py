from __future__ import annotations

import sys
import unittest
from pathlib import Path

ESFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ESFT))

import bfcl_fullffn_v1


class BfclFullFfnV1Tests(unittest.TestCase):
    def test_parse_arm_expands_path_and_reads_dial(self):
        arm = bfcl_fullffn_v1.parse_arm("tail05_a05=~/models/esft/tail05:32:0.5")
        self.assertEqual(arm["name"], "tail05_a05")
        self.assertEqual(arm["model_path"], str(Path("~/models/esft/tail05").expanduser()))
        self.assertEqual(arm["topk"], 32)
        self.assertEqual(arm["alpha"], 0.5)

    def test_parse_arms_rejects_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            bfcl_fullffn_v1.parse_arms(["same=/a:8:0", "same=/b:32:0.5"])

    def test_tail_hook_alpha_zero_matches_top8_renorm(self):
        import torch

        torch.manual_seed(7)
        logits = torch.randn(3, 32)
        scores = torch.softmax(logits, dim=-1)
        idx = torch.arange(32).repeat(3, 1)
        _logits, actual, _idx = bfcl_fullffn_v1.make_router_tail_scale_hook(0.0)(
            None, None, (logits, scores, idx))
        order = scores.argsort(dim=-1, descending=True)
        expected = torch.zeros_like(scores)
        expected.scatter_(-1, order[:, :8], scores.gather(-1, order[:, :8]))
        expected /= expected.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0))

    def test_tail_hook_alpha_one_is_passthrough(self):
        import torch

        torch.manual_seed(11)
        logits = torch.randn(2, 32)
        scores = torch.softmax(logits, dim=-1)
        idx = torch.arange(32).repeat(2, 1)
        _logits, actual, _idx = bfcl_fullffn_v1.make_router_tail_scale_hook(1.0)(
            None, None, (logits, scores, idx))
        self.assertTrue(torch.equal(actual, scores))


if __name__ == "__main__":
    unittest.main()
