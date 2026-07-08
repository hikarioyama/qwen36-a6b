#!/usr/bin/env python
"""GRPO training update ([T] phase) for delta-ESFT Qwen3.6-35B-A3B.

This is the training-side half of the iterative GRPO loop described in
``RL_DESIGN.md``. Rollouts + rewards are produced elsewhere ([G] phase, see
``rollout_gen.py`` / ``inc0_gen.py``); this script consumes a rollouts jsonl and
performs one (or a few) on-policy GRPO update(s) on the expert deltas only.

Input record (one line = one prompt group), flexible field names:
    {
      "instance_id": ...,                     # used to join --data for the prompt
      "prompt_messages": [...]  (optional),   # final prompt (post force-instruction)
      "prefill": "<think>\\n",                 # assistant prefill (continue_final)
      "completions": [{"text": prefill+gen, ...}, ...],
      "<reward-key>": [r_0, ..., r_{N-1}],     # per-completion scalar rewards
    }
completions[i]["text"] is the FULL assistant string (prefill re-stitched, exactly
as ``inc0_gen`` writes it), so completion + prefill alignment is byte-exact.

Core design (faithful to RL_DESIGN.md §[T] / §訓練側の詳細):
  * group advantage  adv_i = (r_i - mean(r)) / (std(r) + eps); std==0 group skipped;
  * completion logp under the current policy (delta ON), only completion tokens
    (prompt + prefill masked) contribute to logp_sum;
  * KL reference is the SAME model with the delta temporarily disabled
    (``delta_disabled``) -- no 70GB reference replica;
  * loss = -(adv * logp_sum) + beta * KL(policy || base);
  * v1 is strictly on-policy 1-step (REINFORCE with group baseline, no ratio/clip).
    ``--seq-ratio-mode gspo`` switches on a GSPO sequence-level importance ratio +
    clip for >1-epoch reuse (default off);
  * delta training path (to_esft_delta): router and non-selected experts are frozen
    structurally (requires_grad=False), so no gradient reaches them -- the packed
    tensors are never written. This replaces train_esft's grad-hook masking, which
    is unnecessary here (documented as a design-consistent deviation in the report).

Run (GPU): see ``run_grpo.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------- #
# Group advantage
# --------------------------------------------------------------------------- #

def group_advantages(rewards, eps: float = 1e-6, norm: str = "std"):
    """GRPO group-normalised advantages.

    Returns ``(advantages, skip)``. A group whose rewards are all identical
    (population std == 0) carries no learning signal and is flagged ``skip=True``
    (caller drops it) -- this also absorbs the all-format-fail (-1) group.

    ``norm`` selects the advantage normalisation (both A/B-able; verified relevant
    by the 2026-07-08 RL literature check):
      * ``"std"``  -- ``adv_i = (r_i - mean) / (std + eps)``. Original GRPO
        (DeepSeekMath 2402.03300).
      * ``"mean"`` -- ``adv_i = r_i - mean``. Dr.GRPO (2503.20783) drops the
        std-division, which its authors show introduces a *question-level
        difficulty bias* (easy/hard groups with small reward-std get over-weighted).
        Removing it recovers the unbiased PPO objective. Effect is on token
        efficiency / gradient bias, NOT an unconditional win -- measure same-condition.
    ``std`` is the population std (ddof=0), matching the GRPO reference.
    """
    import torch

    r = torch.as_tensor(rewards, dtype=torch.float32)
    n = r.numel()
    if n == 0:
        return r, True
    mean = r.mean()
    std = r.std(unbiased=False)
    if float(std) == 0.0:
        return torch.zeros_like(r), True
    if norm == "mean":
        adv = r - mean
    else:
        adv = (r - mean) / (std + eps)
    return adv, False


# --------------------------------------------------------------------------- #
# Delta ON/OFF toggle for the KL reference (no reference-model replica)
# --------------------------------------------------------------------------- #

class delta_disabled:
    """Context manager: temporarily route every patched experts module through the
    *class* forward (frozen packed tensors only), i.e. the base model without the
    residual deltas. Restores the delta-wrapped forward on exit.

    ``to_esft_delta`` replaces ``experts.forward`` with a bound
    ``_delta_wrapped_forward``. Rebinding to ``type(experts).forward`` for the
    duration of the block gives the base (delta-off) output, which is exactly the KL
    reference logp -- computed on the same weights in-place, so no 70GB replica.
    """

    def __init__(self, handles):
        self.experts = list(handles.experts_modules.values())
        self._saved = []

    def __enter__(self):
        self._saved = []
        for e in self.experts:
            self._saved.append(e.forward)
            e.forward = types.MethodType(type(e).forward, e)
        return self

    def __exit__(self, *exc):
        for e, f in zip(self.experts, self._saved):
            e.forward = f
        self._saved = []
        return False


# --------------------------------------------------------------------------- #
# Byte-exact tokenisation: prompt (masked) + prefill (masked) + completion (kept)
# --------------------------------------------------------------------------- #

def _render_ids(tokenizer, messages, *, add_generation_prompt=False,
                continue_final_message=False):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _common_prefix_len(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_token_ids(tokenizer, prompt_messages, full_assistant, prefill):
    """Return ``(input_ids, n_prefix)`` for one completion.

    Reconstructs exactly the sequence the server scored: the assistant turn is
    opened with ``continue_final_message=True`` (mirroring ``inc0_gen.gen_one``),
    ``full_assistant = prefill + gen`` is placed as its content. ``n_prefix`` is the
    token count of everything up to and including the prefill (prompt scaffolding +
    ``prefill``); tokens ``input_ids[n_prefix:]`` are the completion, the only ones
    supervised for logp. ``n_prefix`` is taken as the common-prefix length between
    the prefill-boundary render and the full render, so a tokenizer merge across the
    prefill/gen boundary cannot misalign the mask.
    """
    prefix_msgs = list(prompt_messages) + [{"role": "assistant", "content": prefill}]
    full_msgs = list(prompt_messages) + [{"role": "assistant", "content": full_assistant}]
    prefix_ids = _render_ids(tokenizer, prefix_msgs, continue_final_message=True)
    full_ids = _render_ids(tokenizer, full_msgs, continue_final_message=True)
    n_prefix = _common_prefix_len(prefix_ids, full_ids)
    return full_ids, n_prefix


# --------------------------------------------------------------------------- #
# Completion log-probs (memory-frugal: logits only over completion positions)
# --------------------------------------------------------------------------- #

def completion_logps(model, input_ids, n_prefix, device):
    """Per-token log-prob of the completion tokens under the current model state.

    Returns a 1-D tensor of length ``L = len(input_ids) - n_prefix`` (empty if the
    completion has no tokens). Uses ``logits_to_keep = L + 1`` so the vocab-sized
    logits are only materialised for the completion window (+1 for the token that
    predicts the first completion token) rather than the whole sequence -- no full
    (T, V) logits tensor. Falls back to slicing full logits if the model forward
    does not accept ``logits_to_keep``.
    """
    import torch
    import torch.nn.functional as F

    T = len(input_ids)
    L = T - n_prefix
    if L <= 0:
        return torch.zeros(0, device=device)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    keep = L + 1  # positions [n_prefix-1 .. T-1]
    try:
        out = model(input_ids=ids, logits_to_keep=keep, use_cache=False)
        logits = out.logits  # (1, keep, V), positions [T-keep .. T-1] = [n_prefix-1 .. T-1]
        pred = logits[:, :-1, :]  # positions [n_prefix-1 .. T-2] predict [n_prefix .. T-1]
    except TypeError:
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits  # (1, T, V)
        pred = logits[:, n_prefix - 1:T - 1, :]
    logp = F.log_softmax(pred.float(), dim=-1)
    targets = ids[:, n_prefix:T]  # (1, L)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (L,)
    return tok_logp


# --------------------------------------------------------------------------- #
# Per-completion loss
# --------------------------------------------------------------------------- #

def completion_loss(model, handles, input_ids, n_prefix, adv, beta, device,
                    *, old_logps=None, seq_ratio_mode="none", clip_eps=0.2,
                    compute_kl=True):
    """Loss (+ diagnostics) for a single completion.

    v1 (``seq_ratio_mode='none'``): ``-(adv * sum logp_policy) + beta * KL``.
    GSPO  (``seq_ratio_mode='gspo'``): sequence importance ratio
    ``s = exp(mean(logp_new - logp_old))`` with PPO-style clip on ``s*adv``, plus the
    same KL term. ``old_logps`` (list per completion token) comes from the rollout
    record when present; absent, it defaults to the detached current logp (=> s=1 at
    the first on-policy step).

    KL(policy||base) uses the k3 estimator ``exp(ref-logp) - (ref-logp) - 1`` per
    completion token (non-negative, low-variance; TRL/GRPO standard). ``ref`` is the
    delta-disabled logp on the same weights, detached. Returns a dict with ``loss``
    (0-dim tensor, grad-enabled) and scalar diagnostics.
    """
    import torch

    policy_tok = completion_logps(model, input_ids, n_prefix, device)  # (L,) grad
    L = policy_tok.numel()
    if L == 0:
        return None

    ref_sum = torch.tensor(0.0, device=device)
    kl_sum = torch.tensor(0.0, device=device)
    if compute_kl:
        with torch.no_grad(), delta_disabled(handles):
            ref_tok = completion_logps(model, input_ids, n_prefix, device)  # (L,) no grad
        diff = ref_tok - policy_tok  # (L,)
        kl_tok = torch.exp(diff) - diff - 1.0
        kl_sum = kl_tok.sum()
        ref_sum = ref_tok.sum()

    logp_sum = policy_tok.sum()

    if seq_ratio_mode == "gspo":
        if old_logps is not None and len(old_logps) == L:
            old_tok = torch.tensor(old_logps, dtype=torch.float32, device=device)
        else:
            old_tok = policy_tok.detach()
        s = torch.exp((policy_tok - old_tok).mean())  # sequence-level ratio
        unclipped = s * adv
        clipped = torch.clamp(s, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        pg = -torch.minimum(unclipped, clipped)
        loss = pg + beta * kl_sum
        s_val = float(s.detach())
    else:
        loss = -(adv * logp_sum) + beta * kl_sum
        s_val = 1.0

    return {
        "loss": loss,
        "logp_sum": float(logp_sum.detach()),
        "ref_sum": float(ref_sum.detach()),
        "kl_sum": float(kl_sum.detach()),
        "n_tok": int(L),
        "seq_ratio": s_val,
    }


# --------------------------------------------------------------------------- #
# Rollout loading (flexible; optional join to prompts data for the final prompt)
# --------------------------------------------------------------------------- #

def _load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _reward_list(rec, reward_key):
    """Return the per-completion reward list, honouring --reward-key with fallback."""
    for k in ([reward_key] if reward_key else []) + ["rewards", "strict", "lenient"]:
        v = rec.get(k)
        if isinstance(v, list) and v:
            return v
    # Last resort: pull a scalar off each completion dict.
    comps = rec.get("completions", [])
    for k in ([reward_key] if reward_key else []) + ["reward", "strict", "lenient"]:
        vals = [c.get(k) for c in comps if isinstance(c, dict)]
        if vals and all(isinstance(x, (int, float)) for x in vals):
            return vals
    raise KeyError(f"no reward list found in record (tried {reward_key!r}, rewards, strict, lenient)")


def load_groups(rollouts_path, reward_key, prompts_path=None):
    """Yield ``dict(prompt_messages, prefill, texts, rewards)`` per rollout line.

    The final prompt (post force-instruction) is taken from ``rec['prompt_messages']``
    when present, else reconstructed by joining ``instance_id`` to ``prompts_path``
    (grpo_prompts.jsonl) and applying ``inc0_gen.build_prompt`` -- the exact prompt
    the server was fed. This keeps the completion-conditioning on-policy.

    ASSUMPTION: reconstruction uses the prefill/inc0 convention (force-instruction on
    the last user turn + ``<think>\\n`` assistant prefill), which is the GRPO-intended
    rollout path per RL_DESIGN.md §追記2026-07-06. Raw ``rollout_gen.py`` output (no
    force-instruction, no prefill) must NOT be reconstructed this way -- for that,
    embed the exact ``prompt_messages`` in the rollout record instead.
    """
    prompt_index = None
    build_prompt = None
    if prompts_path:
        from inc0_gen import build_prompt as _bp  # rl/ reuse
        build_prompt = _bp
        prompt_index = {r["instance_id"]: r for r in _load_jsonl(prompts_path)}

    for rec in _load_jsonl(rollouts_path):
        prefill = rec.get("prefill", "")
        comps = rec.get("completions", [])
        texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in comps]
        rewards = _reward_list(rec, reward_key)
        if len(rewards) != len(texts):
            n = min(len(rewards), len(texts))
            rewards, texts = rewards[:n], texts[:n]

        pmsgs = rec.get("prompt_messages")
        if pmsgs is None and prompt_index is not None:
            src = prompt_index.get(rec.get("instance_id"))
            if src is None:
                continue
            pmsgs, bp_prefill = build_prompt(src["prompt_messages"])
            if not prefill:
                prefill = bp_prefill
        if pmsgs is None:
            raise KeyError(
                "record lacks 'prompt_messages' and no --prompts-data given to "
                "reconstruct it; cannot tokenise the prompt for on-policy logp.")
        # old per-token logps for GSPO, if the rollout saved them
        old_logps = [c.get("logprobs") if isinstance(c, dict) else None for c in comps]
        old_logps = old_logps[:len(texts)]
        yield {"instance_id": rec.get("instance_id"), "prompt_messages": pmsgs,
               "prefill": prefill, "texts": texts, "rewards": rewards,
               "old_logps": old_logps}


# --------------------------------------------------------------------------- #
# Delta initialisation from the SFT starting policy
# --------------------------------------------------------------------------- #

def init_delta_from_patch(model, handles, patch_path):
    """Initialise deltas so ``base + delta == SFT policy``.

    The SFT patch (``save_expert_patch`` format) stores EFFECTIVE expert slices. The
    delta scheme keeps the packed tensors bit-exact base and trains a residual, so
    the starting policy is encoded as ``delta = effective_SFT - base_slice``. Reads
    each selected slice and writes the residual into the delta Parameters.
    """
    import torch
    from safetensors.torch import load_file as _st_load
    from esft_qwen.esft_patch import _patch_key

    sd = _st_load(patch_path)
    with torch.no_grad():
        for layer_idx, expert_ids in handles.selected.items():
            experts = handles.experts_modules[layer_idx]
            for i, e in enumerate(expert_ids):
                gu = sd[_patch_key(layer_idx, "gate_up", e)].to(
                    device=experts.gate_up_proj.device, dtype=experts.gate_up_proj.dtype)
                dn = sd[_patch_key(layer_idx, "down", e)].to(
                    device=experts.down_proj.device, dtype=experts.down_proj.dtype)
                experts.delta_gate_up[i].copy_(gu - experts.gate_up_proj[e])
                experts.delta_down[i].copy_(dn - experts.down_proj[e])
    return {"num_layers": len(handles.selected)}


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--expert-config", required=True)
    ap.add_argument("--rollouts", required=True, help="rollouts jsonl ([G]-phase output)")
    ap.add_argument("--prompts-data", default=None,
                    help="grpo_prompts.jsonl to reconstruct the final prompt (if rollouts lack prompt_messages)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--init-delta-state", default=None, help="delta_state.safetensors to resume/start from")
    ap.add_argument("--init-patch", default=None,
                    help="SFT expert_patch.safetensors (effective slices) -> init delta = patch - base")
    ap.add_argument("--reward-key", default="lenient")
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1,
                    help="passes over the rollout batch; one synchronized optimizer step each")
    ap.add_argument("--max-seq-len", type=int, default=7168,
                    help="truncate (keep the tail) sequences longer than this; count logged")
    ap.add_argument("--adv-eps", type=float, default=1e-6)
    ap.add_argument("--adv-norm", choices=["std", "mean"], default="std",
                    help="std=GRPO(2402.03300, default) / mean=Dr.GRPO(2503.20783, drops difficulty bias). A/B it.")
    ap.add_argument("--seq-ratio-mode", choices=["none", "gspo"], default="none")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--no-kl", action="store_true", help="disable the KL term (beta ignored)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5934875)
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText,
    )
    from transformers.optimization import Adafactor
    from esft_qwen.delta_patch import to_esft_delta, load_delta_state, save_delta_state

    # ---- distributed ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp = world_size > 1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    expert_config = {k: v for k, v in json.load(open(args.expert_config)).items()
                     if not k.startswith("_")}

    # ---- model + delta ----
    config = AutoConfig.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, config=config, dtype=dtype, low_cpu_mem_usage=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    handles = to_esft_delta(model, expert_config)
    if args.init_delta_state:
        load_delta_state(model, args.init_delta_state)
    elif args.init_patch:
        init_delta_from_patch(model, handles, args.init_patch)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[rank0] trainable delta params: {n_train:,}", flush=True)

    optimizer = Adafactor(handles.expert_params, lr=args.learning_rate,
                          scale_parameter=False, relative_step=False, warmup_init=False)

    # DDP note: we drive collectives manually (one all-reduce per optimizer step)
    # rather than through DDP's autograd hooks, because ranks have uneven shards and
    # a variable number of per-completion backward() calls -- letting DDP all-reduce
    # on every backward would desync the collectives across ranks and hang. Instead
    # every backward runs local-only and we all-reduce the accumulated delta grads
    # once per step, on a schedule (per-epoch) that is identical across ranks.
    ddp_model = model  # forward/backward directly on the module (local grads only)

    def sync_grads():
        if not ddp:
            return
        for p in handles.expert_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad /= world_size

    # ---- data (sharded across ranks by group index) ----
    groups = list(load_groups(args.rollouts, args.reward_key, args.prompts_data))
    my_groups = [g for i, g in enumerate(groups) if i % world_size == rank]
    if rank == 0:
        print(f"[rank0] {len(groups)} groups total; ~{len(my_groups)}/rank", flush=True)

    beta = 0.0 if args.no_kl else args.kl_beta
    compute_kl = not args.no_kl
    stats = {"steps": 0, "skipped_group": 0, "truncated": 0, "empty_comp": 0,
             "loss_sum": 0.0, "kl_sum": 0.0, "logp_sum": 0.0, "n_comp": 0}

    # One synchronized optimizer step per epoch: each rank accumulates local grads
    # over its whole shard (variable #backwards is fine -- no collectives here), then
    # a single all-reduce + step. Every rank runs the same number of epochs, so the
    # collective count matches exactly. This is the strictly-on-policy 1-step GRPO
    # update per rollout batch (epochs>1 reuses the batch; pair with --seq-ratio-mode
    # gspo for off-policy correction).
    n_groups_denom = max(1, len(groups))  # global group count -> stable grad scale
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for gi, g in enumerate(my_groups):
            adv, skip = group_advantages(g["rewards"], args.adv_eps, norm=args.adv_norm)
            if skip:
                if epoch == 0:
                    stats["skipped_group"] += 1
                continue
            n_comp_group = max(1, len(g["texts"]))
            for ci, text in enumerate(g["texts"]):
                if not text:
                    if epoch == 0:
                        stats["empty_comp"] += 1
                    continue
                input_ids, n_prefix = build_token_ids(
                    tok, g["prompt_messages"], text, g["prefill"])
                if len(input_ids) > args.max_seq_len:
                    dropped = len(input_ids) - args.max_seq_len
                    input_ids = input_ids[-args.max_seq_len:]
                    # shift the prompt/prefill boundary by the dropped front tokens;
                    # keep >=0 and leave at least one completion token to supervise.
                    n_prefix = max(0, min(n_prefix - dropped, len(input_ids) - 1))
                    if epoch == 0:
                        stats["truncated"] += 1
                out = completion_loss(
                    ddp_model, handles, input_ids, n_prefix,
                    adv[ci].to(device), beta, device,
                    old_logps=g["old_logps"][ci] if ci < len(g["old_logps"]) else None,
                    seq_ratio_mode=args.seq_ratio_mode, clip_eps=args.clip_eps,
                    compute_kl=compute_kl)
                if out is None:
                    if epoch == 0:
                        stats["empty_comp"] += 1
                    continue
                # normalise a completion's contribution: 1/(group size * #groups) so
                # the accumulated grad is a mean over the batch regardless of shard.
                loss = out["loss"] / (n_comp_group * n_groups_denom)
                loss.backward()  # local accumulation only (no DDP hook)
                stats["loss_sum"] += float(out["loss"].detach())
                stats["kl_sum"] += out["kl_sum"]
                stats["logp_sum"] += out["logp_sum"]
                stats["n_comp"] += 1
            if rank == 0 and (gi + 1) % 40 == 0:
                nc = max(1, stats["n_comp"])
                print(f"[rank0] epoch {epoch} group {gi+1}/{len(my_groups)} "
                      f"loss={stats['loss_sum']/nc:.4f} kl={stats['kl_sum']/nc:.4f} "
                      f"logp={stats['logp_sum']/nc:.2f} skip={stats['skipped_group']} "
                      f"trunc={stats['truncated']}", flush=True)

        sync_grads()  # single collective per epoch, identical across ranks
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(handles.expert_params, args.max_grad_norm)
        optimizer.step()
        stats["steps"] += 1

    if ddp:
        torch.distributed.barrier()
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "delta_state.safetensors")
        info = save_delta_state(model, path)
        with open(os.path.join(args.output_dir, "expert_cfg.json"), "w") as f:
            json.dump(expert_config, f, indent=1)
        tok.save_pretrained(args.output_dir)
        nc = max(1, stats["n_comp"])
        print(f"[rank0] GRPO update complete -> {info['path']}", flush=True)
        print(f"[rank0] steps={stats['steps']} comps={stats['n_comp']} "
              f"skipped_groups={stats['skipped_group']} truncated={stats['truncated']} "
              f"empty={stats['empty_comp']} mean_loss={stats['loss_sum']/nc:.4f} "
              f"mean_kl={stats['kl_sum']/nc:.4f}", flush=True)

    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
