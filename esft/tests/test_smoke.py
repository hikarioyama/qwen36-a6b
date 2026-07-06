"""CPU-only smoke test for the ESFT-Qwen port. No GPU, no real weights.

Builds a *tiny* stand-in around the real ``Qwen3_5MoeSparseMoeBlock`` (2 layers,
8 experts, hidden 64), random-initialised, and checks the four contracts the
Phase-0 code depends on:

  (a) forward hooks on the router capture (top_idx, top_val) that match the exact
      routing formula in common.compute_router_selection;
  (b) cumulative top-p selection matches hand-computed expectations, and the
      ScoreAccumulator normalisation sums to ~1 per layer;
  (c) to_esft_qwen leaves only the intended packed expert params trainable;
  (d) expert-patch save -> perturb -> load restores the selected experts exactly;
  (e) after a real AdamW step with build_param_groups, non-selected expert rows are
      bit-exact frozen while a selected row moved.

Run:  CUDA_VISIBLE_DEVICES="" <venv>/bin/python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch import nn

# Make the package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from esft_qwen.common import find_moe_blocks, compute_router_selection, infer_moe_dims
from esft_qwen.scoring import (
    ScoreAccumulator,
    select_experts_top_p,
    build_expert_config,
    jaccard,
    domain_overlap_matrix,
)
from esft_qwen.esft_patch import (
    to_esft_qwen,
    save_expert_patch,
    load_expert_patch,
    build_param_groups,
)

HIDDEN = 64
NUM_EXPERTS = 8
TOP_K = 2
MOE_INTER = 16


def tiny_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=MOE_INTER,
        hidden_act="silu",
    )


class _TinyLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = Qwen3_5MoeSparseMoeBlock(cfg)


class _TinyInner(nn.Module):
    def __init__(self, cfg, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(cfg) for _ in range(n_layers)])


class TinyModel(nn.Module):
    """Mimics model.model.language_model.layers.{i}.mlp naming."""

    def __init__(self, cfg, n_layers=2):
        super().__init__()
        self.language_model = _TinyInner(cfg, n_layers)


def build_model(seed=0) -> TinyModel:
    torch.manual_seed(seed)
    cfg = tiny_config()
    model = TinyModel(cfg).to(torch.float32)
    # Random-init all params (empty packed tensors otherwise) so routing varies.
    for p in model.parameters():
        nn.init.normal_(p, mean=0.0, std=0.05)
    return model


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def test_a_hook_capture(model):
    refs = find_moe_blocks(model)
    check("a.find_moe_blocks: 2 layers, idx sorted",
          [r.layer_idx for r in refs] == [0, 1], f"{[r.layer_idx for r in refs]}")

    captured = {}

    def make_hook(pos):
        def hook(_m, _in, out):
            logits, scores, indices = out
            captured[pos] = (logits.detach(), scores.detach(), indices.detach())
        return hook

    hs = [r.gate.register_forward_hook(make_hook(i)) for i, r in enumerate(refs)]
    x = torch.randn(1, 5, HIDDEN)
    with torch.no_grad():
        for r in refs:
            r.block(x)
    for h in hs:
        h.remove()

    ok_shape = all(captured[i][2].shape == (5, TOP_K) for i in captured)
    check("a.hook captures (N, top_k) indices", ok_shape and len(captured) == 2)

    # The captured (idx, val) must match the exact formula recomputed from logits.
    logits, scores, indices = captured[0]
    exp_idx, exp_val = compute_router_selection(logits, TOP_K, norm_topk_prob=True)
    check("a.captured indices == formula", torch.equal(indices, exp_idx))
    check("a.captured weights == formula (renorm on)",
          torch.allclose(scores.float(), exp_val.float(), atol=1e-5))
    # Per-token weights renormalise to 1 -> confirms norm_topk_prob is always on.
    check("a.top_k weights sum to 1",
          torch.allclose(scores.float().sum(-1), torch.ones(5), atol=1e-4))


# --------------------------------------------------------------------------- #
def test_b_selection_and_scoring():
    # Hand case: scores desc 0.4,0.3,0.2,0.1. top_p=0.6 -> take 0.4 (cum 0<0.6),
    # 0.3 (cum .4<.6), stop before 0.2 (cum .7>=.6). => experts {A,B}.
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    sel = select_experts_top_p(scores, 0.6)
    check("b.top-p selects {0,1} for p=0.6", sel == [0, 1], f"{sel}")

    # p=0.0 -> nothing selected (loop breaks immediately: current 0 >= 0).
    check("b.top-p p=0 selects none", select_experts_top_p(scores, 0.0) == [])
    # p=1.0 -> all until cum>=1 (needs all four).
    check("b.top-p p=1.0 selects all", select_experts_top_p(scores, 1.0) == [0, 1, 2, 3])

    # Scoring normalisation: fabricate routing for 10 tokens, top_k=2, 4 experts.
    acc = ScoreAccumulator(num_layers=1, num_experts=4, top_k=2)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, size=(10, 2))
    val = rng.random((10, 2))
    val = val / val.sum(1, keepdims=True)  # per-token weights sum to 1
    acc.update(0, idx, val)
    fin = acc.finalize()
    check("b.n_tokens counted", fin["n_tokens"] == 10, f"{fin['n_tokens']}")
    check("b.token_scores row sums to 1",
          abs(fin["token_scores"][0].sum() - 1.0) < 1e-9, f"{fin['token_scores'][0].sum():.6f}")
    check("b.gate_scores row sums to 1",
          abs(fin["gate_scores"][0].sum() - 1.0) < 1e-9, f"{fin['gate_scores'][0].sum():.6f}")


# --------------------------------------------------------------------------- #
def test_c_to_esft(model):
    # Train experts {0,3} on layer 0 and {5} on layer 1.
    cfg = {"experts": {"0": [0, 3], "1": [5]},
           "shared_experts": False, "non_expert_modules": False}
    handles = to_esft_qwen(model, cfg)

    refs = {r.layer_idx: r for r in find_moe_blocks(model)}
    # Only the two packed expert tensors per configured layer are trainable.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {
        "language_model.layers.0.mlp.experts.gate_up_proj",
        "language_model.layers.0.mlp.experts.down_proj",
        "language_model.layers.1.mlp.experts.gate_up_proj",
        "language_model.layers.1.mlp.experts.down_proj",
    }
    check("c.exactly the packed expert params are trainable", trainable == expected,
          f"extra={trainable - expected} missing={expected - trainable}")
    # Router / shared expert frozen.
    check("c.router frozen",
          not refs[0].gate.weight.requires_grad)
    check("c.shared_expert frozen",
          all(not p.requires_grad for p in refs[0].shared_expert.parameters()))
    # Mask matches config.
    m0 = handles.masks[(0, "gate_up")]
    check("c.mask marks selected experts", m0.nonzero().flatten().tolist() == [0, 3],
          f"{m0.nonzero().flatten().tolist()}")
    handles.remove()


# --------------------------------------------------------------------------- #
def test_d_patch_roundtrip(model):
    cfg = {"experts": {"0": [0, 3], "1": [5]},
           "shared_experts": False, "non_expert_modules": False}
    refs = {r.layer_idx: r for r in find_moe_blocks(model)}
    # Snapshot selected experts before saving.
    before = {
        (0, "gate_up", 0): refs[0].experts.gate_up_proj[0].clone(),
        (0, "gate_up", 3): refs[0].experts.gate_up_proj[3].clone(),
        (1, "down", 5): refs[1].experts.down_proj[5].clone(),
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "patch.safetensors")
        info = save_expert_patch(model, cfg, path)
        # (2 experts L0 + 1 expert L1) * 2 tensors each = 6 slices.
        check("d.patch tensor count == 6", info["num_tensors"] == 6, f"{info['num_tensors']}")
        # Corrupt the model's experts, then restore from patch.
        with torch.no_grad():
            refs[0].experts.gate_up_proj.add_(7.0)
            refs[1].experts.down_proj.add_(7.0)
        load_expert_patch(model, path)
    ok = (
        torch.equal(refs[0].experts.gate_up_proj[0], before[(0, "gate_up", 0)])
        and torch.equal(refs[0].experts.gate_up_proj[3], before[(0, "gate_up", 3)])
        and torch.equal(refs[1].experts.down_proj[5], before[(1, "down", 5)])
    )
    check("d.load restores selected experts exactly", ok)


# --------------------------------------------------------------------------- #
def test_e_freeze_invariance():
    # Fresh model so the earlier tests' state can't interfere.
    model = build_model(seed=1)
    cfg = {"experts": {"0": [2], "1": [4]},
           "shared_experts": False, "non_expert_modules": False}
    handles = to_esft_qwen(model, cfg)
    refs = {r.layer_idx: r for r in find_moe_blocks(model)}

    # Snapshot a non-selected and the selected expert row on layer 0.
    frozen_before = refs[0].experts.gate_up_proj[0].clone()
    trained_before = refs[0].experts.gate_up_proj[2].clone()

    groups = build_param_groups(model, handles, weight_decay=0.1)
    opt = torch.optim.AdamW(groups, lr=1e-2)

    x = torch.randn(2, 8, HIDDEN)
    for _ in range(3):
        opt.zero_grad()
        out = sum(r.block(x).pow(2).mean() for r in refs.values())
        out.backward()
        opt.step()

    frozen_after = refs[0].experts.gate_up_proj[0]
    trained_after = refs[0].experts.gate_up_proj[2]
    check("e.non-selected expert row bit-exact frozen",
          torch.equal(frozen_before, frozen_after))
    check("e.selected expert row moved",
          not torch.equal(trained_before, trained_after))
    # Grad on non-selected rows must be zero (mask working).
    g = refs[0].experts.gate_up_proj.grad
    if g is not None:
        nonsel_grad = g[torch.tensor([0, 1, 3, 4, 5, 6, 7])]
        check("e.non-selected grad rows are zero", torch.count_nonzero(nonsel_grad) == 0)
    handles.remove()


# --------------------------------------------------------------------------- #
def test_f_overlap_matrix():
    cfgs = {
        "math": {"experts": {"0": [0, 1, 2], "1": [3]}},
        "code": {"experts": {"0": [1, 2, 3], "1": [3]}},
    }
    ov = domain_overlap_matrix(cfgs)
    # Layer 0: {0,1,2} vs {1,2,3} -> J = 2/4 = 0.5.
    check("f.jaccard layer0 == 0.5", abs(ov["per_layer"][0][0, 1] - 0.5) < 1e-9,
          f"{ov['per_layer'][0][0, 1]}")
    check("f.jaccard self == 1.0", abs(ov["mean"][0, 0] - 1.0) < 1e-9)


def main():
    print(f"torch {torch.__version__}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}  "
          f"cuda.is_available={torch.cuda.is_available()}")
    model = build_model(seed=0)
    nl, ne, tk = infer_moe_dims(model)
    print(f"tiny model: moe_layers={nl} experts={ne} top_k={tk}")
    test_a_hook_capture(model)
    test_b_selection_and_scoring()
    test_c_to_esft(model)
    test_d_patch_roundtrip(model)
    test_e_freeze_invariance()
    test_f_overlap_matrix()

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"\n{'=' * 60}\nSMOKE: {n_pass}/{n_total} passed")
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
