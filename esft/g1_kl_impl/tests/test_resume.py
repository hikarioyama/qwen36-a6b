#!/usr/bin/env python3
"""CPU tests for strict delta checkpoint loading and resume preflight."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_esft import load_delta_state_strict, validate_delta_resume_checkpoint


class TinyDeltaModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.register_parameter(
            "delta_gate_up", torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype)))
        self.block.register_parameter(
            "delta_down", torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype)))
        self.other = torch.nn.Parameter(torch.ones(1, dtype=dtype))


def write_delta(path, model, *, drop=None, extra=False, dtype=None, bad_shape=False):
    tensors = {}
    for name, parameter in model.named_parameters():
        if not name.endswith(("delta_gate_up", "delta_down")) or name == drop:
            continue
        shape = (1,) if bad_shape and name.endswith("delta_down") else parameter.shape
        tensors[name] = torch.full(shape, 3, dtype=dtype or parameter.dtype)
    if extra:
        tensors["unexpected.delta_down"] = torch.zeros(1)
    save_file(tensors, str(path), metadata={"format": "esft-qwen-delta-v1"})


class StrictDeltaLoadTests(unittest.TestCase):
    def test_loads_all_deltas_without_touching_other_parameters(self):
        model = TinyDeltaModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "delta_state.safetensors"
            write_delta(path, model)
            info = load_delta_state_strict(model, str(path))
        self.assertEqual(info["num_written"], 2)
        self.assertTrue(torch.equal(model.block.delta_gate_up, torch.full((2, 3), 3.0)))
        self.assertTrue(torch.equal(model.block.delta_down, torch.full((2, 2), 3.0)))
        self.assertTrue(torch.equal(model.other, torch.ones(1)))

    def test_rejects_missing_or_unexpected_keys(self):
        model = TinyDeltaModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.safetensors"
            write_delta(path, model, drop="block.delta_down")
            with self.assertRaises(KeyError):
                load_delta_state_strict(model, str(path))
            path = Path(tmp) / "extra.safetensors"
            write_delta(path, model, extra=True)
            with self.assertRaises(KeyError):
                load_delta_state_strict(model, str(path))

    def test_rejects_shape_and_dtype_conversion(self):
        model = TinyDeltaModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shape.safetensors"
            write_delta(path, model, bad_shape=True)
            with self.assertRaises(ValueError):
                load_delta_state_strict(model, str(path))
            path = Path(tmp) / "dtype.safetensors"
            write_delta(path, model, dtype=torch.float64)
            with self.assertRaises(TypeError):
                load_delta_state_strict(model, str(path))


class ResumePreflightTests(unittest.TestCase):
    def make_checkpoint(self, root, *, global_step=500, world_size=2):
        root = Path(root)
        model = TinyDeltaModel()
        write_delta(root / "delta_state.safetensors", model)
        torch.save({}, root / "optimizer.pt")
        torch.save({}, root / "scheduler.pt")
        (root / "trainer_state.json").write_text(
            json.dumps({"global_step": global_step}))
        for rank in range(world_size):
            torch.save({}, root / f"rng_state_{rank}.pth")

    def test_validates_complete_distributed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            info = validate_delta_resume_checkpoint(tmp, max_steps=1000, world_size=2)
        self.assertEqual(info["global_step"], 500)
        self.assertEqual(info["world_size"], 2)

    def test_rejects_missing_rng_and_nonadvancing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            os.unlink(Path(tmp) / "rng_state_1.pth")
            with self.assertRaises(FileNotFoundError):
                validate_delta_resume_checkpoint(tmp, max_steps=1000, world_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            with self.assertRaises(ValueError):
                validate_delta_resume_checkpoint(tmp, max_steps=500, world_size=2)


if __name__ == "__main__":
    unittest.main()
