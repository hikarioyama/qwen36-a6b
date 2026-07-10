"""Unit tests for router-mobile joint training (esft_patch RouterAnchor / gate
unfreeze / low-LR param group).

Synthetic tiny MoE only -- no real model, no GPU. Validates the LOGIC:
  1. grad-gate: router grad>0, selected-expert grad>0, frozen module grad None
  2. anchor is numerically 0 at step0 (current==base) and >0 once the gate moves
  3. build_param_groups yields a router group with lr = mult*base_lr, no overlap
  4. RouterAnchor's KL matches a hand-computed reference
  5. legacy path (train_router off) keeps the gate frozen

Run: python -m pytest tests/test_router_joint.py -q     (or: python tests/test_router_joint.py)
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esft_qwen.esft_patch import (
    to_esft_qwen, build_param_groups,
    enable_router_training, snapshot_router_weights, RouterAnchor,
)


# --------------------------------------------------------------------------- #
# Synthetic tiny MoE mirroring Qwen3_5Moe's class contract:
#   * block class name ends with "SparseMoeBlock" (find_moe_blocks matches this)
#   * module path contains "layers.{i}.mlp"
#   * gate.forward(x) -> (router_logits, router_scores, router_indices), no bias,
#     weight shape (E, H), attribute top_k
#   * experts are PACKED Parameters gate_up_proj (E,2I,H) / down_proj (E,H,I)
# --------------------------------------------------------------------------- #

H, I_, E, TOP_K = 16, 8, 6, 3


class TinyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = E
        self.hidden_dim = H
        self.weight = nn.Parameter(torch.randn(E, H) * 0.1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_val, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        top_val = top_val.to(router_logits.dtype)
        return router_logits, top_val, top_idx


class TinyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I_, H) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(E, H, I_) * 0.1)

    def forward(self, x, top_idx, top_val):
        # (N,H) -> per-token weighted sum over its top_k experts. Differentiable in
        # both packed tensors (only the selected rows contribute via gather).
        N = x.shape[0]
        out = x.new_zeros(N, H)
        for j in range(top_idx.shape[1]):
            e = top_idx[:, j]                       # (N,)
            gu = self.gate_up_proj[e]               # (N, 2I, H)
            dn = self.down_proj[e]                  # (N, H, I)
            h = torch.bmm(gu, x.unsqueeze(-1)).squeeze(-1)   # (N, 2I)
            act = F.silu(h[:, :I_]) * h[:, I_:]              # (N, I)
            y = torch.bmm(dn, act.unsqueeze(-1)).squeeze(-1)  # (N, H)
            out = out + top_val[:, j:j + 1] * y
        return out


class TinySparseMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = TinyRouter()
        self.experts = TinyExperts()
        self.shared_expert = None

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        _, scores, idx = self.gate(x)
        y = self.experts(x, idx, scores)
        return y.view(b, s, h)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinySparseMoeBlock()

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(H, H, bias=False)   # a non-expert / non-router module
        self.layers = nn.ModuleList([TinyLayer() for _ in range(n_layers)])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


def _cfg():
    # select experts [0,2] on layer 0 and [1] on layer 1
    return {"experts": {"0": [0, 2], "1": [1]}}


def _rand_input(b=2, s=4):
    torch.manual_seed(0)
    return torch.randn(b, s, H)


# --------------------------------------------------------------------------- #
# Test 1: grad-gate
# --------------------------------------------------------------------------- #

def test_grad_gate():
    torch.manual_seed(1)
    model = TinyModel()
    handles = to_esft_qwen(model, _cfg(), train_router=True)

    x = _rand_input()
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    # (a) router (gate) params have grad > 0
    router_gn = sum(p.grad.norm().item() for p in handles.router_params if p.grad is not None)
    assert len(handles.router_params) > 0, "no router params recorded"
    assert router_gn > 0, f"router grad norm should be >0, got {router_gn}"
    assert all(p.grad is not None for p in handles.router_params), "a gate param has no grad"

    # (b) selected expert (packed) params have grad > 0
    expert_gn = sum(p.grad.norm().item() for p in handles.expert_params if p.grad is not None)
    assert expert_gn > 0, f"expert grad norm should be >0, got {expert_gn}"

    # (c) a non-unfrozen module (embed) has grad None
    assert model.embed.weight.requires_grad is False
    assert model.embed.weight.grad is None, "frozen embed should have no grad"
    print("test_grad_gate PASS "
          f"(router_gn={router_gn:.4e}, expert_gn={expert_gn:.4e}, embed.grad=None)")


# --------------------------------------------------------------------------- #
# Test 2: anchor is ~0 at step0, >0 after the gate moves
# --------------------------------------------------------------------------- #

def test_anchor_zero_at_step0():
    torch.manual_seed(2)
    model = TinyModel()
    handles = to_esft_qwen(model, _cfg(), train_router=True)
    snap = snapshot_router_weights(model)
    anchor = RouterAnchor(model, snap, weight=0.15)

    x = _rand_input()
    _ = model(x)                       # populates the hook stash
    a0 = anchor.compute()
    a0v = float(a0.detach()) if torch.is_tensor(a0) else a0
    assert a0v < 1e-6, f"anchor at step0 should be ~0, got {a0v}"

    # move the gate weights, recompute
    with torch.no_grad():
        for ref_p in handles.router_params:
            ref_p.add_(torch.randn_like(ref_p) * 0.5)
    _ = model(x)
    a1 = anchor.compute()
    a1v = float(a1.detach()) if torch.is_tensor(a1) else a1
    assert a1v > 1e-4, f"anchor after gate moved should be >0, got {a1v}"
    anchor.remove()
    print(f"test_anchor_zero_at_step0 PASS (a0={a0v:.3e}, a1={a1v:.3e})")


# --------------------------------------------------------------------------- #
# Test 3: param-group LR and no overlap
# --------------------------------------------------------------------------- #

def test_param_group_router_lr():
    torch.manual_seed(3)
    model = TinyModel()
    handles = to_esft_qwen(model, _cfg(), train_router=True)
    base_lr = 1e-5
    mult = 0.08
    groups = build_param_groups(model, handles, weight_decay=0.1,
                                router_lr=mult * base_lr)

    router_groups = [g for g in groups if "lr" in g]
    assert len(router_groups) == 1, f"expected exactly one router group, got {len(router_groups)}"
    rg = router_groups[0]
    assert abs(rg["lr"] - mult * base_lr) < 1e-20, f"router lr wrong: {rg['lr']}"
    assert rg["weight_decay"] == 0.0

    router_ids = {id(p) for p in rg["params"]}
    router_expected = {id(p) for p in handles.router_params}
    assert router_ids == router_expected, "router group params != unfrozen gate params"

    # no overlap with any other group
    for g in groups:
        if g is rg:
            continue
        other = {id(p) for p in g["params"]}
        assert router_ids.isdisjoint(other), "router params leaked into another group"

    # expert params still present and disjoint from router
    expert_ids = {id(p) for p in handles.expert_params}
    assert router_ids.isdisjoint(expert_ids), "router/expert param overlap"
    print(f"test_param_group_router_lr PASS (router_lr={rg['lr']:.3e}, "
          f"n_router={len(router_ids)}, n_groups={len(groups)})")


# --------------------------------------------------------------------------- #
# Test 4: anchor KL matches a hand-computed reference
# --------------------------------------------------------------------------- #

def test_anchor_matches_manual_kl():
    torch.manual_seed(4)
    model = TinyModel(n_layers=1)
    to_esft_qwen(model, {"experts": {"0": [0, 2]}}, train_router=True)
    # snapshot BEFORE moving the gate, then move it so KL != 0
    snap = snapshot_router_weights(model)
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.add_(torch.randn(E, H) * 0.3)
    anchor = RouterAnchor(model, snap, weight=1.0)  # weight=1 for a clean compare

    x = _rand_input()
    _ = model(x)
    got = anchor.compute()
    got_v = float(got.detach())

    # Manual reference: KL(log_softmax(cur) || softmax(base)) batchmean over tokens.
    xf = model.embed(x).reshape(-1, H)   # gate input == embed(x) since layer0 sees embed output
    cur_w = model.layers[0].mlp.gate.weight
    base_w = snap[0]
    cur_logits = F.linear(xf, cur_w)
    base_logits = F.linear(xf, base_w)
    log_cur = F.log_softmax(cur_logits.float(), dim=-1)
    base_prob = F.softmax(base_logits.float(), dim=-1)
    ref = F.kl_div(log_cur, base_prob, reduction="batchmean").item()

    assert abs(got_v - ref) < 1e-5, f"anchor KL mismatch: got {got_v} vs ref {ref}"
    anchor.remove()
    print(f"test_anchor_matches_manual_kl PASS (got={got_v:.6f}, ref={ref:.6f})")


# --------------------------------------------------------------------------- #
# Test 5: legacy path keeps the router frozen
# --------------------------------------------------------------------------- #

def test_legacy_router_frozen():
    torch.manual_seed(5)
    model = TinyModel()
    handles = to_esft_qwen(model, _cfg())   # default: train_router=False
    for ref in model.layers:
        assert ref.mlp.gate.weight.requires_grad is False, "gate must stay frozen by default"
    assert handles.router_params == [], "router_params must be empty in legacy mode"

    # and build_param_groups without router_lr yields no lr-bearing group
    groups = build_param_groups(model, handles, weight_decay=0.1)
    assert all("lr" not in g for g in groups), "no router group should exist by default"
    print("test_legacy_router_frozen PASS (gate frozen, no router group)")


def _run_all():
    tests = [
        test_grad_gate,
        test_anchor_zero_at_step0,
        test_param_group_router_lr,
        test_anchor_matches_manual_kl,
        test_legacy_router_frozen,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"{t.__name__} FAIL: {e}")
    print("=" * 60)
    print(f"{len(tests) - failed}/{len(tests)} PASSED")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
