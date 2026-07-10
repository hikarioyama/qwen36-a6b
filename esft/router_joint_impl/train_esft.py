#!/usr/bin/env python
"""Phase-0 (GPU-gated): ESFT training for Qwen3.6-35B-A3B.

Adapts DeepSeek ESFT's train.py to the Qwen3_5Moe architecture. Two methods:

  * ``--method delta`` (default): residual-delta ESFT (esft_qwen/delta_patch.py).
    Packed expert tensors stay frozen; small zero-init delta Parameters
    ([n_sel, ...]) are trained instead, so gradients/optimiser state exist only
    for the selected experts (~2B params for the k32/p0.2 math config, not 32B).
    The whole model then fits on ONE 96GB GPU and the two GPUs run plain DDP:

        # 1) tokenise+pack once (CPU-only, cached):
        <venv>/bin/python train_esft.py ... --prepare-data-only
        # 2) train on both GPUs:
        torchrun --nproc_per_node=2 train_esft.py --model ... \
            --expert-config configs/math_token_k32_p0.2.json \
            --train-data data/train/math.jsonl --output-dir runs/math_esft_k32 \
            --router-top-k 32 --grad-accum 16

  * ``--method maskhook`` (legacy): whole-packed-tensor requires_grad + gradient
    row-masking hooks. Materialises full 64GB bf16 gradients at 35B scale => OOM.
    Kept for the CPU smoke tests and small models only.

``--router-top-k 32`` widens routing at TRAIN time (gate.top_k is read at call
time; see tests/verify_topk_override.py) so rank-9..32 selected experts receive
tokens and therefore gradient -- required for ESFT@k32 checkpoints that will be
SERVED at k=32.

Hyperparameters follow ESFT's configs/base.yaml: LR 1e-5, seq 4096, effective
batch 32 (per_device_batch * grad_accum * n_gpus), constant LR, eval/save on
steps, best checkpoint by eval_loss (selected manually from log history; Trainer
checkpoints hold only the delta tensors, not the 67GB base).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Deltas make allocation sizes irregular (per-layer n_sel varies 12..29);
# expandable segments avoid fragmentation-induced OOM near the 96GB ceiling.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DELTA_STATE_NAME = "delta_state.safetensors"


def install_topk_random_hook(model, gates, topk_set, seed):
    """Register a model-level forward-pre-hook that rewrites gate.top_k every forward.

    * training forward -> k sampled uniformly from ``topk_set`` (all gates share it)
    * eval/save (model.training is False) -> max(topk_set) so eval is always at k=32

    gate.top_k is read at call time by Qwen3_5MoeTopKRouter.forward, so rewriting it
    here before the forward changes the routing width for that step. The RNG is seeded
    per-run (not per-rank) and only advances on training forwards, so both DDP ranks
    draw an identical k sequence. Returns (handle, eval_k).
    """
    topk_set = sorted(set(topk_set))
    if not topk_set:
        raise ValueError("topk_set is empty")
    eval_k = max(topk_set)
    for g in gates:
        g.top_k = eval_k  # default so any pre-train read (infer_moe_dims) sees k=32
    rng = random.Random(seed)

    def _topk_random_pre_hook(module, inputs):
        k = rng.choice(topk_set) if module.training else eval_k
        for g in gates:
            g.top_k = k

    handle = model.register_forward_pre_hook(_topk_random_pre_hook)
    return handle, eval_k


def render_and_tokenize(tokenizer, messages, mask_prompt=True, ignore_id=-100):
    """Tokenise a conversation with the Qwen chat template.

    Builds labels turn-by-turn: assistant turns are supervised, everything else
    (system/user, and the template scaffolding preceding each assistant turn) is
    masked to ``ignore_id`` when ``mask_prompt`` is set. This mirrors ESFT's
    prompt-masking while respecting the model's own chat template.
    """
    input_ids, labels = [], []
    prev_len = 0
    for i, msg in enumerate(messages):
        convo = messages[: i + 1]
        text = tokenizer.apply_chat_template(
            convo, tokenize=False,
            add_generation_prompt=(msg["role"] != "assistant"),
        )
        ids_full = tokenizer(text, add_special_tokens=False)["input_ids"]
        new_ids = ids_full[prev_len:]
        input_ids.extend(new_ids)
        if msg["role"] == "assistant" or not mask_prompt:
            labels.extend(new_ids)
        else:
            labels.extend([ignore_id] * len(new_ids))
        prev_len = len(ids_full)
    return input_ids, labels


def pack_pairs(pairs, tokenizer, seq_length, random_concat_ratio, seed, ignore_id=-100):
    """Concatenate pre-tokenised (input_ids, labels) pairs into fixed-length blocks.

    Follows ESFT's get_examples_from_buffer_pad: greedily fill blocks, occasionally
    dropping the leading token of a concatenated example (random_concat_ratio), pad
    the trailing block.
    """
    rng = random.Random(seed)
    all_in, all_lab = [], []
    cur_in, cur_lab = [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for iid, lab in pairs:
        if len(iid) > seq_length - len(cur_in):
            iid = iid[-(seq_length - len(cur_in)):]
            lab = lab[-(seq_length - len(cur_lab)):]
        if cur_in and rng.random() < random_concat_ratio:
            iid, lab = iid[1:], lab[1:]
        cur_in.extend(iid)
        cur_lab.extend(lab)
        if len(cur_in) >= seq_length:
            all_in.append(cur_in[:seq_length])
            all_lab.append(cur_lab[:seq_length])
            cur_in, cur_lab = [], []
    if cur_in:
        cur_in += [pad_id] * (seq_length - len(cur_in))
        cur_lab += [ignore_id] * (seq_length - len(cur_lab))
        all_in.append(cur_in)
        all_lab.append(cur_lab)
    return all_in, all_lab


def pack_examples(records, tokenizer, seq_length, random_concat_ratio, seed, ignore_id=-100):
    """Original single-process API (kept for the eval harness / tests)."""
    pairs = [render_and_tokenize(tokenizer, rec["messages"], ignore_id=ignore_id)
             for rec in records]
    return pack_pairs(pairs, tokenizer, seq_length, random_concat_ratio, seed, ignore_id)


def pack_pairs_streaming(pairs, tokenizer, seq_length, random_concat_ratio, seed,
                         ignore_id=-100):
    """Memory-efficient pack: identical block layout to pack_pairs, but accumulates
    into int32 numpy blocks and frees each source pair as it is consumed. Returns
    (input_ids, labels) int64 tensors directly. This is what lets 63k long agentic
    trajectories pack at seq 16-24k without the ~90GB Python-list blowup that OOM'd
    the naive path. Consumes `pairs` in place (entries set to None). RNG draw order
    matches pack_pairs exactly so ccr>0 packing is bit-identical."""
    import numpy as np
    rng = random.Random(seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    blk_in, blk_lab = [], []
    cur_in = np.empty(seq_length, dtype=np.int32)
    cur_lab = np.empty(seq_length, dtype=np.int32)
    n = 0
    for i in range(len(pairs)):
        iid, lab = pairs[i]
        pairs[i] = None  # free the source pair immediately
        iid = np.asarray(iid, dtype=np.int32)
        lab = np.asarray(lab, dtype=np.int32)
        if len(iid) > seq_length - n:
            iid = iid[-(seq_length - n):]
            lab = lab[-(seq_length - n):]
        if n > 0 and rng.random() < random_concat_ratio:
            iid, lab = iid[1:], lab[1:]
        m = len(iid)
        cur_in[n:n + m] = iid
        cur_lab[n:n + m] = lab
        n += m
        if n >= seq_length:
            blk_in.append(cur_in)
            blk_lab.append(cur_lab)
            cur_in = np.empty(seq_length, dtype=np.int32)
            cur_lab = np.empty(seq_length, dtype=np.int32)
            n = 0
    if n > 0:
        cur_in[n:] = pad_id
        cur_lab[n:] = ignore_id
        blk_in.append(cur_in)
        blk_lab.append(cur_lab)
    import torch
    input_ids = torch.from_numpy(np.stack(blk_in)).long()
    labels = torch.from_numpy(np.stack(blk_lab)).long()
    return input_ids, labels


# ---- parallel tokenisation (spawn workers; rust tokenizer is per-call serial) ----

_WORKER_TOK = None
_WORKER_CAP = 0


def _trim_tail(iid, lab, cap):
    """Keep only the tail `cap` tokens. pack_pairs discards everything before the
    tail seq_length of any record, so trimming here is byte-identical to the packed
    output but bounds peak RAM to ~cap tokens/record instead of the full trajectory."""
    if cap and len(iid) > cap:
        return iid[-cap:], lab[-cap:]
    return iid, lab


def _tok_worker_init(tok_path, cap):
    global _WORKER_TOK, _WORKER_CAP
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(tok_path)
    _WORKER_CAP = cap


def _tok_worker_one(messages):
    import array
    iid, lab = render_and_tokenize(_WORKER_TOK, messages)
    iid, lab = _trim_tail(iid, lab, _WORKER_CAP)
    # compact 'i' (int32) arrays: 4 B/token vs ~28 B for a Python int in a list.
    # token ids (<250k) and labels (-100..vocab) both fit int32. Cuts the peak
    # tokeniser RAM ~7x, which is what makes 63k long trajectories fit at seq 16-24k.
    return array.array("i", iid), array.array("i", lab)


def tokenize_parallel(records, tok_path, workers, cap=0):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    msgs = [rec["messages"] for rec in records]
    pairs = []
    with ctx.Pool(workers, initializer=_tok_worker_init, initargs=(tok_path, cap)) as pool:
        for i, pair in enumerate(pool.imap(_tok_worker_one, msgs, chunksize=64)):
            pairs.append(pair)
            if (i + 1) % 20000 == 0:
                print(f"[tokenize] {i + 1}/{len(msgs)}", flush=True)
    return pairs


def cache_path_for(args):
    base = os.path.basename(args.train_data)
    return os.path.join(
        args.data_cache_dir,
        f"{base}.seq{args.seq_length}.seed{args.seed}"
        f".ccr{args.random_concat_ratio}.max{args.max_records}.pt")


def build_or_load_packed(args, tokenizer, tok_path, allow_build):
    import torch
    cache = cache_path_for(args)
    if os.path.exists(cache):
        blob = torch.load(cache, weights_only=True)
        print(f"[data] loaded cache {cache}: {blob['input_ids'].shape[0]} blocks")
        return blob["input_ids"], blob["labels"]
    if not allow_build:
        sys.exit(f"[data] cache missing under multi-process launch: {cache}\n"
                 f"run once with --prepare-data-only first (avoids NCCL timeouts "
                 f"while 2 ranks tokenise for an hour)")
    records = []
    with open(args.train_data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if args.max_records and len(records) >= args.max_records:
                    break
    print(f"[data] tokenising {len(records)} records "
          f"({args.tokenize_workers} workers)...", flush=True)
    import time
    t0 = time.time()
    cap = args.tokenize_cap if args.tokenize_cap > 0 else args.seq_length
    if args.tokenize_workers > 1:
        pairs = tokenize_parallel(records, tok_path, args.tokenize_workers, cap=cap)
    else:
        pairs = [_trim_tail(*render_and_tokenize(tokenizer, rec["messages"]), cap)
                 for rec in records]
    print(f"[data] tokenised in {time.time() - t0:.0f}s (tail-cap={cap})")
    input_ids, labels = pack_pairs_streaming(pairs, tokenizer, args.seq_length,
                                             args.random_concat_ratio, args.seed)
    del pairs
    os.makedirs(args.data_cache_dir, exist_ok=True)
    torch.save({"input_ids": input_ids, "labels": labels}, cache)
    print(f"[data] packed {input_ids.shape[0]} blocks of {args.seq_length} -> {cache}")
    return input_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--expert-config", required=True)
    ap.add_argument("--train-data", required=True, help="ESFT-format jsonl ({'messages': [...]})")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["delta", "maskhook"], default="delta")
    ap.add_argument("--router-top-k", type=int, default=0,
                    help="override gate.top_k at train time (0 = keep config value 8); "
                         "use 32 for ESFT@k32 so rank-9..32 experts receive gradient")
    ap.add_argument("--router-topk-random", action="store_true",
                    help="stochastic co-activation (EMoE-min): randomise gate.top_k on "
                         "EVERY training forward, sampled uniformly from --topk-random-set "
                         "(all MoE layers share one k per forward). Eval/save use max(set) "
                         "(=32) so evaluation is always at k=32. Overrides --router-top-k. "
                         "Selection (delta) unchanged; low-k forwards just leave some "
                         "selected experts inactive that step.")
    ap.add_argument("--topk-random-set", default="8,16,24,32",
                    help="comma-separated top_k values sampled uniformly per forward when "
                         "--router-topk-random is set; max() is the eval/save value")
    ap.add_argument("--train-router", action="store_true",
                    help="ROUTER-MOBILE joint training: unfreeze every gate (router) and "
                         "train it at a low LR (--router-lr-mult) with a base-routing anchor "
                         "(--router-anchor-weight). Off by default = legacy frozen router. "
                         "The router logits are computed BEFORE top-k, so the anchor pins the "
                         "k=8 distribution with one forward (k-independent).")
    ap.add_argument("--router-lr-mult", type=float, default=0.08,
                    help="router param-group LR = router_lr_mult * learning_rate")
    ap.add_argument("--router-anchor-weight", type=float, default=0.15,
                    help="lambda for the base-routing anchor KL added to the loss "
                         "(0 disables the anchor; router still trains)")
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="applied to non-expert trainable params only; experts/deltas stay wd=0")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32,
                    help="per_device_batch * grad_accum * n_gpus should be ~32")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--random-concat-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=5934875)
    ap.add_argument("--max-records", type=int, default=0, help="cap training records (0=all)")
    ap.add_argument("--max-val-blocks", type=int, default=64,
                    help="cap eval set size (full 2%% of 23k blocks would make each "
                         "75-step eval take ~20min; 64 blocks = 262k tokens is plenty)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--data-cache-dir", default=None,
                    help="defaults to <dirname(train-data)>/cache")
    ap.add_argument("--tokenize-workers", type=int, default=16)
    ap.add_argument("--prepare-data-only", action="store_true",
                    help="tokenise+pack+cache, then exit (no GPU needed)")
    ap.add_argument("--verify-frozen", action="store_true",
                    help="after training, bitwise-compare every packed expert tensor "
                         "against the base checkpoint on disk (delta method only)")
    ap.add_argument("--tokenize-cap", type=int, default=0,
                    help="cap each record's rendered tokens to its tail N before packing "
                         "(0 = auto: use --seq-length). pack_pairs keeps only the tail "
                         "seq_length of any record (ccr>=0), so tail-cap is byte-identical "
                         "to the packed result while bounding peak tokeniser RAM.")
    ap.add_argument("--fused-ce", action="store_true",
                    help="use Liger fused-linear-cross-entropy: run the text backbone to "
                         "hidden_states then FLCE(lm_head.weight, hidden, labels) without "
                         "materialising full [seq x vocab] logits. Unlocks long seq_length. "
                         "Router aux loss is dropped (router is frozen -> gradient-irrelevant).")
    args = ap.parse_args()
    if args.data_cache_dir is None:
        args.data_cache_dir = os.path.join(os.path.dirname(args.train_data) or ".", "cache")

    import torch
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText,
        TrainingArguments, Trainer,
    )
    from torch.utils.data import TensorDataset

    from esft_qwen.esft_patch import (
        to_esft_qwen, build_param_groups, save_expert_patch,
        enable_router_training, snapshot_router_weights, RouterAnchor,
    )
    from esft_qwen.delta_patch import (
        to_esft_delta, save_delta_state, load_delta_state,
        save_expert_patch_delta, verify_frozen_vs_disk,
    )
    from esft_qwen.common import find_moe_blocks

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_tag = f"[rank{max(local_rank, 0)}]"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    expert_config = {k: v for k, v in json.load(open(args.expert_config)).items()
                     if not k.startswith("_")}

    # ---- data (cached; build only in single-process mode) ----
    input_ids, labels = build_or_load_packed(args, tokenizer, tok_path,
                                             allow_build=(world_size <= 1))
    if args.prepare_data_only:
        print("[data] prepare-data-only: done")
        return
    dataset = TensorDataset(input_ids, labels)
    n_val = min(max(1, int(len(dataset) * 0.02)), args.max_val_blocks)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    print(f"{rank_tag} packed {len(dataset)} blocks of {args.seq_length} tokens "
          f"(train {len(train_ds)} / val {len(val_ds)})")

    # ---- model: full copy per rank (fits one GPU with the delta method) ----
    device_index = local_rank if local_rank >= 0 else 0
    device_map = {"": device_index}
    config = AutoConfig.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, config=config, dtype=dtype, device_map=device_map)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=dtype, device_map=device_map)
    model.config.use_cache = False

    if args.router_topk_random:
        # Stochastic co-activation (EMoE 2509.21892, minimal variant): each TRAIN
        # forward routes at a k sampled uniformly from topk_set so selected experts
        # learn to co-fire across k=8..32 instead of only at the fixed serve-time k.
        # Eval/save fall back to max(set)=32. RNG seeded per-run so both DDP ranks
        # draw the same k each forward (identical routing width, no grad divergence).
        topk_set = sorted({int(x) for x in args.topk_random_set.split(",") if x.strip()})
        if not topk_set:
            sys.exit("--topk-random-set is empty")
        gates = [ref.gate for ref in find_moe_blocks(model)]
        _, eval_k = install_topk_random_hook(model, gates, topk_set, args.seed + 20260708)
        print(f"{rank_tag} router top_k RANDOMISED over {topk_set} per train forward "
              f"on {len(gates)} layers; eval/save k={eval_k}")
    elif args.router_top_k and args.router_top_k > 0:
        refs = find_moe_blocks(model)
        for ref in refs:
            ref.gate.top_k = args.router_top_k
        print(f"{rank_tag} router top_k -> {args.router_top_k} on {len(refs)} layers")

    if args.method == "delta":
        handles = to_esft_delta(model, expert_config)
    else:
        handles = to_esft_qwen(model, expert_config)
    # ---- router-mobile joint training (flag-gated; default off = frozen router) ----
    router_snapshot = None
    if args.train_router:
        rp = enable_router_training(model, handles)
        router_snapshot = snapshot_router_weights(model)  # base anchor target (pre-optimiser)
        print(f"{rank_tag} router UNFROZEN: {len(rp)} gate params trainable, "
              f"LR = {args.router_lr_mult} * {args.learning_rate} = "
              f"{args.router_lr_mult * args.learning_rate:.3e}; "
              f"anchor lambda = {args.router_anchor_weight}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{rank_tag} ESFT trainable params ({args.method}): {n_train:,}")

    if args.fused_ce:
        # Fused linear cross-entropy: skip the [B, seq, vocab] fp32 logits tensor that
        # ForCausalLMLoss materialises (the real long-seq VRAM wall: seq x 248320 x 4B).
        # Run the text backbone -> hidden_states, then FLCE(lm_head.weight, hidden, labels)
        # which fuses the head matmul + CE without ever forming full logits, making loss
        # memory seq-independent. Router aux loss is intentionally dropped: the gate is
        # frozen under ESFT, so its aux term carries no gradient to trainable params.
        import types
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        from transformers.modeling_outputs import CausalLMOutputWithPast
        _backbone = model.model.language_model   # Qwen3_5MoeTextModel -> last_hidden_state
        _lm_head = model.lm_head
        _flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        def _fused_ce_forward(self, input_ids=None, attention_mask=None, labels=None,
                              position_ids=None, **kw):
            out = _backbone(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False)
            hidden = out.last_hidden_state
            loss = None
            if labels is not None:
                H = hidden.size(-1)
                sh = hidden[:, :-1, :].reshape(-1, H)
                sl = labels[:, 1:].reshape(-1).to(sh.device)
                bias = getattr(_lm_head, "bias", None)
                loss = _flce(_lm_head.weight, sh, sl, bias)
            return CausalLMOutputWithPast(loss=loss, logits=None)

        model.forward = types.MethodType(_fused_ce_forward, model)
        print(f"{rank_tag} fused-CE (Liger FLCE) enabled: full [seq x "
              f"{_lm_head.weight.size(0)}] logits skipped; router aux dropped (frozen)")

    # ---- base-routing anchor: wrap model.forward to add lambda*KL(current||base) ----
    # Wrapping the final model.forward (after the optional fused-CE swap) covers both
    # the fused and standard loss paths uniformly. The anchor forward-hooks stash each
    # gate's input during the backbone forward; compute() folds the per-forward KL into
    # the returned loss so it flows through the (trainable) gate weights.
    if args.train_router and args.router_anchor_weight > 0 and router_snapshot is not None:
        anchor = RouterAnchor(model, router_snapshot, weight=args.router_anchor_weight)
        pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                  else tokenizer.eos_token_id)
        _base_forward = model.forward  # bound method (fused or original)

        def _forward_with_anchor(*fargs, **fkw):
            out = _base_forward(*fargs, **fkw)
            if getattr(out, "loss", None) is not None:
                am = fkw.get("attention_mask")
                if am is None:
                    ids = fkw.get("input_ids")
                    if ids is None and fargs:
                        ids = fargs[0]
                    am = (ids != pad_id) if (ids is not None and pad_id is not None) else None
                out.loss = out.loss + anchor.compute(am)
            return out

        model.forward = _forward_with_anchor
        print(f"{rank_tag} router anchor active (lambda={args.router_anchor_weight}, "
              f"{len(anchor.handles)} gate hooks)")

    # Custom optimiser: expert/delta params in a weight_decay=0 group.
    router_lr = (args.router_lr_mult * args.learning_rate) if args.train_router else None
    param_groups = build_param_groups(model, handles, weight_decay=args.weight_decay,
                                      router_lr=router_lr)
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(param_groups, lr=args.learning_rate,
                              scale_parameter=False, relative_step=False,
                              warmup_init=False, beta1=None)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate,
                                      betas=(0.9, 0.95), fused=torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        lr_scheduler_type="constant",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=8,
        load_best_model_at_end=False,  # checkpoints are delta-only; best applied manually below
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(args.dtype == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        report_to=[],
        seed=args.seed,
        ddp_find_unused_parameters=False,   # index_add puts every delta in the graph each microbatch
        ddp_broadcast_buffers=False,
    )

    def data_collator(data):
        return {"input_ids": torch.stack([d[0] for d in data]),
                "labels": torch.stack([d[1] for d in data])}

    class DeltaTrainer(Trainer):
        """Checkpoints hold only the delta tensors (~4GB), not the 67GB base."""

        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            info = save_delta_state(self.model, os.path.join(output_dir, DELTA_STATE_NAME))
            print(f"{rank_tag} [delta-ckpt] {info}")
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    trainer_cls = DeltaTrainer if args.method == "delta" else Trainer

    # grad-gate 診断: GRAD_PROBE=1 のとき、各 optimizer step の直前(=backward 後)に
    # router / 選抜expert の grad norm と、非訓練 param の grad が None かを表示・assert する。
    # 実モデル(40層・gradient checkpointing)越しに router へ勾配が流れるかを見る唯一の実証点。
    _extra_callbacks = []
    if os.environ.get("GRAD_PROBE") == "1":
        from transformers import TrainerCallback
        _rp = list(getattr(handles, "router_params", []) or [])
        _ep = list(getattr(handles, "expert_params", []) or [])
        # 非訓練の見本: embed_tokens.weight(常に凍結のはず)
        _frozen_sample = None
        for _n, _p in model.named_parameters():
            if "embed_tokens" in _n:
                _frozen_sample = (_n, _p); break

        def _gnorm(ps):
            tot, n = 0.0, 0
            for p in ps:
                if p.grad is not None:
                    tot += float(p.grad.detach().float().norm() ** 2); n += 1
            return (tot ** 0.5), n

        class GradProbe(TrainerCallback):
            def on_pre_optimizer_step(self, a, s, c, **kw):
                rn, rc = _gnorm(_rp)
                en, ec = _gnorm(_ep)
                msg = f"{rank_tag} [GRAD_PROBE step={s.global_step}] router_gnorm={rn:.3e}(n={rc}) expert_gnorm={en:.3e}(n={ec})"
                if _frozen_sample is not None:
                    fn, fp = _frozen_sample
                    msg += f" frozen[{fn}].grad={'None' if fp.grad is None else 'NOT-None!!'}"
                print(msg, flush=True)
                if rc and rn == 0.0:
                    print(f"{rank_tag} [GRAD_PROBE] FAIL: router grad is zero (勾配が流れていない)", flush=True)

        _extra_callbacks.append(GradProbe())

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # scheduler built by Trainer (constant)
        callbacks=_extra_callbacks or None,
    )
    if trainer.accelerator.ddp_handler is not None:
        # grads live inside the DDP reduction buckets instead of a second copy
        trainer.accelerator.ddp_handler.gradient_as_bucket_view = True

    trainer.train()

    dev = torch.device(f"cuda:{device_index}")
    print(f"{rank_tag} cuda max_memory_allocated={torch.cuda.max_memory_allocated(dev) / 2**30:.2f}GiB "
          f"max_memory_reserved={torch.cuda.max_memory_reserved(dev) / 2**30:.2f}GiB")

    if not trainer.args.should_save:
        return  # non-main ranks are done

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- pick best checkpoint from eval history and load its deltas ----
    if args.method == "delta":
        evals = [(h["eval_loss"], int(h["step"])) for h in trainer.state.log_history
                 if "eval_loss" in h and "step" in h]
        if evals:
            best_loss, best_step = min(evals)
            ckpt = os.path.join(args.output_dir, f"checkpoint-{best_step}", DELTA_STATE_NAME)
            if os.path.exists(ckpt):
                info = load_delta_state(model, ckpt)
                print(f"[best] eval_loss={best_loss:.5f} @ step {best_step}; "
                      f"loaded {info['num_written']} delta tensors from {ckpt}")
            else:
                print(f"[best] WARNING: checkpoint for best step {best_step} missing "
                      f"({ckpt}); saving final-step deltas instead")
        save_delta_state(model, os.path.join(args.output_dir, DELTA_STATE_NAME))
        patch_path = os.path.join(args.output_dir, "expert_patch.safetensors")
        info = save_expert_patch_delta(model, handles, patch_path)
    else:
        patch_path = os.path.join(args.output_dir, "expert_patch.safetensors")
        info = save_expert_patch(model, expert_config, patch_path)
    with open(os.path.join(args.output_dir, "expert_cfg.json"), "w") as f:
        json.dump(expert_config, f, indent=1)
    tokenizer.save_pretrained(args.output_dir)
    print(f"training complete; patch -> {patch_path} ({info})")

    if args.verify_frozen and args.method == "delta":
        print("[verify-frozen] bitwise compare of all packed expert tensors vs disk...")
        report = verify_frozen_vs_disk(model, args.model, handles)
        summary = {"ok": report["ok"], "n_rows_mismatched": report["n_rows_mismatched"],
                   "delta": report["delta"]}
        print(f"[verify-frozen] {json.dumps(summary)}")
        with open(os.path.join(args.output_dir, "verify_frozen.json"), "w") as f:
            json.dump(report, f, indent=1)
        if not report["ok"]:
            sys.exit("[verify-frozen] FAILED: packed expert tensors deviate from base")


if __name__ == "__main__":
    main()
