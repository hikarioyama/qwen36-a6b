"""CPU unit tests for the opt-in Full-FFN + mobile-router trainer path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


ESFT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ESFT_ROOT / "deploy" / "train_fullffn_dcp.py"
sys.path.insert(0, str(ESFT_ROOT))
spec = importlib.util.spec_from_file_location("fullffn_joint_trainer", TRAINER_PATH)
trainer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(trainer)

class TinyRouter(nn.Module):
    def __init__(self, hidden=4, experts=5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(experts, hidden) * 0.2)

    def forward(self, x):
        return F.linear(x, self.weight)


class TinyExperts(nn.Module):
    def __init__(self, hidden=4, experts=5):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(experts, hidden, hidden))
        self.down_proj = nn.Parameter(torch.randn(experts, hidden, hidden))


class TinySparseMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = TinyRouter()
        self.experts = TinyExperts()
        self.shared_expert = None

    def forward(self, x):
        _ = self.gate(x)
        return x


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinySparseMoeBlock()
        self.attn = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.mlp(self.attn(x))


class TinyFullFfn(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList([TinyLayer()])

    def forward(self, x):
        return self.layers[0](self.embed(x))


class FSDPShardShimRouter(nn.Module):
    """Router with the externally visible FSDP shard empty outside forward.

    Its private full weight is a stand-in for FSDP's temporary unshard.  The
    regression test proves the anchor consumes the forward-emitted logits rather
    than attempting the old, invalid ``F.linear(x, gate.weight)`` afterwards.
    """
    def __init__(self, hidden=4, experts=5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0))
        self._unsharded_weight = nn.Parameter(torch.randn(experts, hidden) * 0.2)

    def forward(self, x):
        logits = F.linear(x.reshape(-1, x.shape[-1]), self._unsharded_weight)
        scores = torch.softmax(logits, dim=-1)[..., :1]
        return logits, scores, torch.zeros_like(scores, dtype=torch.long)


class FSDPShimSparseMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FSDPShardShimRouter()
        self.experts = TinyExperts()
        self.shared_expert = None

    def forward(self, x):
        # Non-reentrant checkpointing matches the trainer's explicit setting.
        # Disable early-stop so this CPU test reliably exercises the hook's
        # backward recomputation invocation as well as the initial forward.
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            return torch.utils.checkpoint.checkpoint(self.gate, x, use_reentrant=False)[0]


class FSDPShimLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FSDPShimSparseMoeBlock()

    def forward(self, x):
        return self.mlp(x)


class FSDPShimFullFfn(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList([FSDPShimLayer()])

    def forward(self, x):
        return self.layers[0](self.embed(x))


def configure_fullffn(model: TinyFullFfn, *, train_router: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    experts = model.layers[0].mlp.experts
    experts.gate_up_proj.requires_grad_(True)
    experts.down_proj.requires_grad_(True)
    model.layers[0].mlp.gate.weight.requires_grad_(train_router)


class FullFfnJointTrainerTests(unittest.TestCase):
    def test_router_group_uses_low_lr_without_overlap(self):
        model = TinyFullFfn()
        configure_fullffn(model, train_router=True)

        groups = trainer.build_fullffn_joint_param_groups(
            model.named_parameters(),
            learning_rate=1e-5,
            weight_decay=0.1,
            train_router=True,
            router_lr_mult=0.08,
        )

        self.assertEqual(len(groups), 2)
        expert_group, router_group = groups
        self.assertEqual(expert_group["lr"], 1e-5)
        self.assertEqual(expert_group["weight_decay"], 0.1)
        self.assertAlmostEqual(router_group["lr"], 8e-7, places=18)
        self.assertEqual(router_group["weight_decay"], 0.0)
        router_ids = {id(p) for p in router_group["params"]}
        expert_ids = {id(p) for p in expert_group["params"]}
        self.assertEqual(router_ids, {id(model.layers[0].mlp.gate.weight)})
        self.assertTrue(router_ids.isdisjoint(expert_ids))

    def test_anchor_kl_uses_pre_topk_output_and_cpu_snapshot(self):
        torch.manual_seed(7)
        model = TinyFullFfn()
        configure_fullffn(model, train_router=True)
        base = trainer.snapshot_router_weights_to_cpu(model)
        self.assertEqual(base[0].device.type, "cpu")
        gate = model.layers[0].mlp.gate
        with torch.no_grad():
            gate.weight.add_(torch.randn_like(gate.weight) * 0.4)
        anchor = trainer.FSDPSafeRouterAnchor(model, base, weight=1.0)
        x = torch.randn(3, 4)

        anchor.begin()
        _ = model(x)
        got = anchor.compute()
        gate_input = model.layers[0].attn(model.embed(x))
        current_logits = F.linear(gate_input, gate.weight)
        base_logits = F.linear(gate_input, base[0])
        expected = F.kl_div(
            F.log_softmax(current_logits.float(), dim=-1),
            F.softmax(base_logits.float(), dim=-1),
            reduction="batchmean",
        )
        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=1e-6))

        got.backward()
        self.assertTrue(trainer.parameter_has_nonzero_grad(gate.weight))
        before = float(got.detach())
        with torch.no_grad():
            gate.weight.add_(gate.weight.grad, alpha=-0.1)
        gate.weight.grad = None
        anchor.begin()
        _ = model(x)
        after = float(anchor.compute().detach())
        self.assertLess(after, before)
        anchor.remove()

    def test_anchor_survives_empty_fsdp_shard_and_ignores_recompute_hook(self):
        torch.manual_seed(11)
        model = FSDPShimFullFfn()
        gate = model.layers[0].mlp.gate
        base = {0: gate._unsharded_weight.detach().cpu().clone()}
        with torch.no_grad():
            gate._unsharded_weight.add_(0.3)
        anchor = trainer.FSDPSafeRouterAnchor(model, base, weight=1.0)

        x = torch.randn(2, 3, 4)
        anchor.begin()
        model_output = model(x)
        kl = anchor.compute()
        self.assertTrue(torch.is_tensor(kl))
        self.assertEqual(gate.weight.numel(), 0)  # the old implementation crashes here
        hook_calls_after_forward = anchor._hook_calls
        (model_output.sum() + kl).backward()
        self.assertGreater(anchor._hook_calls, hook_calls_after_forward)
        self.assertGreater(anchor._ignored_recompute_calls, 0)
        self.assertEqual(len(anchor._records), 0)
        self.assertTrue(trainer.parameter_has_nonzero_grad(gate._unsharded_weight))
        anchor.remove()

    def test_frozen_assertions_cover_frozen_and_joint_modes(self):
        frozen = TinyFullFfn()
        configure_fullffn(frozen, train_router=False)
        frozen.embed.weight.grad = torch.ones_like(frozen.embed.weight)
        self.assertIn(
            "embed.weight",
            trainer.fullffn_frozen_grad_violations(frozen.named_parameters()),
        )
        frozen.embed.weight.grad = None
        self.assertEqual(trainer.fullffn_frozen_grad_violations(frozen.named_parameters()), [])

        joint = TinyFullFfn()
        configure_fullffn(joint, train_router=True)
        joint.layers[0].mlp.gate.weight.grad = torch.ones_like(joint.layers[0].mlp.gate.weight)
        self.assertTrue(trainer.parameter_has_nonzero_grad(joint.layers[0].mlp.gate.weight))
        self.assertEqual(trainer.fullffn_frozen_grad_violations(joint.named_parameters()), [])
        joint.layers[0].attn.weight.grad = torch.ones_like(joint.layers[0].attn.weight)
        self.assertIn(
            "layers.0.attn.weight",
            trainer.fullffn_frozen_grad_violations(joint.named_parameters()),
        )


if __name__ == "__main__":
    unittest.main()
