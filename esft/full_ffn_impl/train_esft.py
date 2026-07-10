#!/usr/bin/env python
"""Phase-0 (GPU-gated): ESFT training for Qwen3.6-35B-A3B.

Adapts DeepSeek ESFT's train.py to the Qwen3_5Moe architecture. Three methods:

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

  * ``--method full-ffn``: train EVERY routed expert FFN of all 40 layers (32.2B
    trainable) with FSDP FULL_SHARD (ZeRO-3) across 8 GPUs. No delta, no mask, no
    grad hook (all experts are supervised). The custom optimiser path is bypassed;
    the HF Trainer builds the optimiser AFTER accelerate wraps the model in FSDP
    (a pre-wrap optimiser would capture unsharded params). Router/gate/shared/attn/
    embed stay frozen. Launch on all 8 GPUs:

        torchrun --nproc_per_node=8 train_esft.py --model ... --method full-ffn \
            --expert-config configs/fullffn_all.json --train-data data/train/v3.jsonl \
            --output-dir runs/fullffn --router-top-k 32 --seq-length 7168 --fused-ce \
            --optimizer adafactor --grad-accum 4 --max-steps 3000

    (``--expert-config`` is still required by the CLI but its expert selection is
    IGNORED for full-ffn -- all experts train. Pass any valid config, e.g. one with
    an empty ``experts`` map; only its presence is used.)

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
FULLFFN_CHECKPOINT_MARKER = "checkpoint_complete.json"


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
    try:
        iid, lab = render_and_tokenize(_WORKER_TOK, messages)
    except Exception:
        return None  # skip records whose chat template cannot render
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
        _skipped = 0
        for i, pair in enumerate(pool.imap(_tok_worker_one, msgs, chunksize=64)):
            if pair is None:
                _skipped += 1
                continue
            pairs.append(pair)
            if (i + 1) % 20000 == 0:
                print(f"[tokenize] {i + 1}/{len(msgs)}", flush=True)
    if _skipped:
        print(f"[tokenize] skipped {_skipped} record(s) with unrenderable chat template", flush=True)
    return pairs


def cache_path_for(args, data_path=None):
    data_path = data_path or args.train_data
    base = os.path.basename(data_path)
    return os.path.join(
        args.data_cache_dir,
        f"{base}.seq{args.seq_length}.seed{args.seed}"
        f".ccr{args.random_concat_ratio}.max{args.max_records}.pt")


def build_or_load_packed(args, tokenizer, tok_path, allow_build, data_path=None):
    import torch
    data_path = data_path or args.train_data
    cache = cache_path_for(args, data_path)
    if os.path.exists(cache):
        blob = torch.load(cache, weights_only=True)
        print(f"[data] loaded cache {cache}: {blob['input_ids'].shape[0]} blocks")
        return blob["input_ids"], blob["labels"]
    if not allow_build:
        sys.exit(f"[data] cache missing under multi-process launch: {cache}\n"
                 f"run once with --prepare-data-only first (avoids NCCL timeouts "
                 f"while 2 ranks tokenise for an hour)")
    records = []
    with open(data_path) as f:
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


def mix_replay_blocks(main_in, main_lab, replay_in, replay_lab, replay_ratio, seed):
    """FORGETTING GUARD (data stage, default off): interleave a fraction of replay
    blocks so the mixed training set is ``replay_ratio`` replay by block count.

    n_replay = round(replay_ratio/(1-replay_ratio) * n_main); replay blocks are
    sampled (with replacement if the replay pool is smaller) with a fixed seed so
    both/all ranks build the identical mixed set before random_split. Returns the
    concatenated (input_ids, labels)."""
    import torch
    n_main = main_in.shape[0]
    if replay_ratio <= 0.0 or n_main == 0 or replay_in.shape[0] == 0:
        return main_in, main_lab
    n_replay = int(round(replay_ratio / (1.0 - replay_ratio) * n_main))
    if n_replay <= 0:
        return main_in, main_lab
    g = torch.Generator().manual_seed(seed + 777)
    pool = replay_in.shape[0]
    idx = torch.randint(0, pool, (n_replay,), generator=g)
    rep_in = replay_in[idx]
    rep_lab = replay_lab[idx]
    mixed_in = torch.cat([main_in, rep_in], dim=0)
    mixed_lab = torch.cat([main_lab, rep_lab], dim=0)
    print(f"[replay] mixed {n_replay} replay blocks into {n_main} main "
          f"(ratio={n_replay / (n_main + n_replay):.3f})")
    return mixed_in, mixed_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--expert-config", required=True)
    ap.add_argument("--train-data", required=True, help="ESFT-format jsonl ({'messages': [...]})")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["delta", "maskhook", "full-ffn"], default="delta")
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
    ap.add_argument("--router-anchor-stride", type=int, default=1,
                    help="Anchor every Nth MoE layer (>1 cuts retained-activation "
                         "memory; 1 = all layers).")
    ap.add_argument("--router-anchor-weight", type=float, default=0.15,
                    help="lambda for the base-routing anchor KL added to the loss "
                         "(0 disables the anchor; router still trains)")
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="full-ffn expert weight decay (campaign default: 0.0); "
                         "delta experts stay in their dedicated wd=0 group")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32,
                    help="per_device_batch * grad_accum * n_gpus should be ~32")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="[full-ffn] checkpoint-N directory containing a complete DCP "
                         "model+optimizer state, scheduler, trainer state, and per-rank RNG")
    ap.add_argument("--skip-final-hf-export", action="store_true",
                    help="[full-ffn probe only] retain the resumable DCP checkpoint but "
                         "skip the expensive final full-state HF consolidation")
    ap.add_argument("--skip-final-checkpoint", action="store_true",
                    help="[full-ffn diagnostic only] do not force a checkpoint at max_steps")
    ap.add_argument("--deterministic-fullffn", action="store_true",
                    help="[full-ffn diagnostic] require deterministic PyTorch kernels; "
                         "the runner must also pin CUBLAS/NCCL environment settings")
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
    # ---- FULL-FFN forgetting guards (structure only; all default OFF) ----
    ap.add_argument("--replay-data", default=None,
                    help="[full-ffn] ESFT-format jsonl of replay/anchor data mixed into "
                         "training at --replay-ratio (data stage). Cached like --train-data.")
    ap.add_argument("--replay-ratio", type=float, default=0.0,
                    help="[full-ffn] target fraction of training BLOCKS drawn from "
                         "--replay-data (0 = no replay). See mix_replay_blocks.")
    ap.add_argument("--kl-teacher", default=None,
                    help="[full-ffn] path to a FROZEN base checkpoint used as a k=8 "
                         "self-distillation teacher (CE+beta*KL). SCAFFOLD ONLY -- see "
                         "IMPLEMENTATION_NOTES.md 'KL teacher'. Enabling it currently raises "
                         "because a frozen 35B teacher cannot co-reside with the 32B-trainable "
                         "FSDP shards on 8x96GB without a separate sharded-teacher design.")
    ap.add_argument("--kl-beta", type=float, default=0.0,
                    help="[full-ffn] weight of the KL(student_k32 || teacher_k8) term added "
                         "to CE when --kl-teacher is set (forces logits: disables --fused-ce).")
    args = ap.parse_args()
    if args.data_cache_dir is None:
        args.data_cache_dir = os.path.join(os.path.dirname(args.train_data) or ".", "cache")

    is_full_ffn = args.method == "full-ffn"
    if is_full_ffn and args.train_router:
        ap.error("--train-router is forbidden with --method full-ffn; the campaign router is frozen")
    if is_full_ffn and args.router_topk_random:
        ap.error("--router-topk-random is forbidden with --method full-ffn; train at fixed k=32")
    if is_full_ffn and args.optimizer != "adafactor":
        ap.error("--method full-ffn currently requires --optimizer adafactor")
    if args.resume_from_checkpoint and not is_full_ffn:
        ap.error("--resume-from-checkpoint is currently supported only for --method full-ffn")
    if args.skip_final_hf_export and not is_full_ffn:
        ap.error("--skip-final-hf-export is valid only for --method full-ffn")
    if args.skip_final_checkpoint and not is_full_ffn:
        ap.error("--skip-final-checkpoint is valid only for --method full-ffn")
    if args.skip_final_checkpoint and os.environ.get("FULLFFN_PROBE") != "1":
        ap.error("--skip-final-checkpoint requires FULLFFN_PROBE=1")
    if args.deterministic_fullffn and not is_full_ffn:
        ap.error("--deterministic-fullffn is valid only for --method full-ffn")

    # ---- FSDP env MUST be set before from_pretrained so cpu_ram_efficient_loading
    # (rank>0 loads on meta, rank0 materialises real weights) kicks in. transformers'
    # is_fsdp_enabled() checks ACCELERATE_USE_FSDP=="true" AND an initialised process
    # group, so we also init dist below before loading the model. ----
    if is_full_ffn:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "1")
        # Transformers 5.7 drops fsdp_config.state_dict_type while constructing
        # Accelerate's plugin. The environment variable is the authoritative path.
        os.environ["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"

    import torch
    if args.deterministic_fullffn:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError("deterministic Full-FFN requires CUBLAS_WORKSPACE_CONFIG")
        if os.environ.get("NCCL_ALGO") != "Ring" or os.environ.get("NCCL_PROTO") != "Simple":
            raise RuntimeError("deterministic Full-FFN requires NCCL_ALGO=Ring NCCL_PROTO=Simple")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText,
        TrainingArguments, Trainer,
    )
    from torch.utils.data import TensorDataset

    from esft_qwen.esft_patch import (
        to_esft_qwen, to_esft_full, build_param_groups, save_expert_patch,
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

    # full-ffn: init the process group before from_pretrained so meta-loading works.
    if is_full_ffn and world_size > 1 and not torch.distributed.is_initialized():
        torch.cuda.set_device(max(local_rank, 0))
        torch.distributed.init_process_group(backend="nccl")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    expert_config = {k: v for k, v in json.load(open(args.expert_config)).items()
                     if not k.startswith("_")}

    # ---- data (cached; build only in single-process mode) ----
    input_ids, labels = build_or_load_packed(args, tokenizer, tok_path,
                                             allow_build=(world_size <= 1))
    if args.replay_ratio > 0.0 and args.replay_data:
        rep_in, rep_lab = build_or_load_packed(args, tokenizer, tok_path,
                                               allow_build=(world_size <= 1),
                                               data_path=args.replay_data)
        input_ids, labels = mix_replay_blocks(input_ids, labels, rep_in, rep_lab,
                                              args.replay_ratio, args.seed)
    if args.prepare_data_only:
        print("[data] prepare-data-only: done")
        return
    dataset = TensorDataset(input_ids, labels)
    n_val = min(max(1, int(len(dataset) * 0.02)), args.max_val_blocks)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    print(f"{rank_tag} packed {len(dataset)} blocks of {args.seq_length} tokens "
          f"(train {len(train_ds)} / val {len(val_ds)})")

    # ---- model ----
    # delta/maskhook: full copy per rank on ONE GPU (device_map pins it). full-ffn:
    # device_map=None so accelerate/FSDP shards it; with cpu_ram_efficient_loading
    # only rank0 materialises real weights (others meta), then FSDP scatters shards.
    device_index = local_rank if local_rank >= 0 else 0
    device_map = None if is_full_ffn else {"": device_index}
    config = AutoConfig.from_pretrained(args.model)
    if args.deterministic_fullffn:
        print(
            f"{rank_tag} [fullffn-deterministic] enabled=True "
            f"cublas={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} "
            f"nccl_algo={os.environ.get('NCCL_ALGO')} "
            f"nccl_proto={os.environ.get('NCCL_PROTO')} "
            f"torch={torch.are_deterministic_algorithms_enabled()} "
            f"experts_impl={getattr(config, '_experts_implementation', None)}",
            flush=True,
        )
    dtype = getattr(torch, args.dtype)
    load_kw = dict(config=config, dtype=dtype, device_map=device_map)
    if is_full_ffn:
        load_kw["low_cpu_mem_usage"] = True
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
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
    elif args.method == "full-ffn":
        handles = to_esft_full(model)
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
    if is_full_ffn:
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        expected_names = [
            name for name, _ in model.named_parameters()
            if name.endswith(".mlp.experts.gate_up_proj")
            or name.endswith(".mlp.experts.down_proj")
        ]
        unexpected = sorted(set(trainable_names) - set(expected_names))
        missing = sorted(set(expected_names) - set(trainable_names))
        router_trainable = [name for name in trainable_names if ".mlp.gate." in name]
        if (len(trainable_names) != 80 or len(expected_names) != 80 or unexpected
                or missing or router_trainable or n_train != 32_212_254_720):
            raise RuntimeError(
                "Full-FFN freeze boundary mismatch: "
                f"trainable={len(trainable_names)} expected={len(expected_names)} "
                f"params={n_train} router={router_trainable[:3]} "
                f"unexpected={unexpected[:3]} missing={missing[:3]}"
            )
        print(
            f"{rank_tag} [fullffn-freeze-audit] trainable_tensors=80 "
            "trainable_params=32212254720 router_trainable=0 unexpected=0 missing=0",
            flush=True,
        )

    if args.kl_teacher:
        # SCAFFOLD ONLY. The CE+KL loss composition is spec'd in IMPLEMENTATION_NOTES
        # but hosting a frozen 35B k=8 teacher alongside the 32B-trainable FSDP shards
        # needs its own design (a TP/sharded teacher or an external teacher server).
        # Fail loudly rather than silently run something untested/OOM-prone.
        raise NotImplementedError(
            "--kl-teacher is a scaffold; see IMPLEMENTATION_NOTES.md 'KL teacher'. "
            "A frozen 35B teacher does not fit beside the 32B-trainable full-ffn FSDP "
            "shards on 8x96GB without a separate sharded-teacher design.")

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
        anchor = RouterAnchor(model, router_snapshot, weight=args.router_anchor_weight,
                              stride=args.router_anchor_stride)
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

    # ---- optimiser ----
    # delta/maskhook: custom optimiser with a weight_decay=0 group for packed/delta
    # params, passed to the Trainer. full-ffn: DO NOT build the optimiser here -- the
    # Trainer builds it AFTER accelerate wraps the model in FSDP (a pre-wrap optimiser
    # would capture unsharded params and break). Weight decay is applied to the (only
    # trainable) expert params via TrainingArguments.weight_decay + --optim.
    optimizer = None
    if not is_full_ffn:
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

    # ---- FSDP config (full-ffn only) ----
    fsdp = ""
    fsdp_config = None
    optim_name = None
    if is_full_ffn:
        fsdp = "full_shard auto_wrap"
        fsdp_config = {
            # Wrap at the DECODER LAYER, not the experts module: the layer forward
            # all-gathers gate_up_proj AND down_proj together so the expert matmuls
            # see unsharded weights (experts alone cannot be wrapped -- the block
            # forward needs both packed tensors resident in the same unit).
            "transformer_layer_cls_to_wrap": ["Qwen3_5MoeDecoderLayer"],
            # REQUIRED with mixed frozen/trainable params (router/attn frozen,
            # experts trainable): flat-param FSDP forbids mixing requires_grad within
            # one FlatParameter; use_orig_params keeps per-Parameter requires_grad.
            "use_orig_params": True,
            # rank>0 loads on meta, rank0 real -> then FSDP scatters. Halves host-RAM
            # peak (needs the ACCELERATE_USE_FSDP env set before from_pretrained above).
            "sync_module_states": True,
            "cpu_ram_efficient_loading": True,
            # Intermediate checkpoints as sharded DCP (each rank writes its shard);
            # cheap + no 70GB rank0 gather mid-run. Final consolidation is done
            # manually below with FULL_STATE_DICT. (If a HF version rejects this key,
            # drop it -- Trainer default FULL_STATE_DICT still works, just heavier.)
            "state_dict_type": "SHARDED_STATE_DICT",
            "limit_all_gathers": True,
            "backward_prefetch": "backward_pre",
        }
        # Trainer-built optimiser: adafactor keeps state ~= param count. NOTE under
        # FSDP the flat 1D shard makes Adafactor NON-factored (fp32 full second
        # moment ~16GB/GPU), still << AdamW's +258GB. See NOTES memory math.
        optim_name = "adafactor" if args.optimizer == "adafactor" else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay if is_full_ffn else 0.0,
        warmup_steps=0,
        lr_scheduler_type="constant",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="no" if args.skip_final_checkpoint else "steps",
        save_total_limit=None if is_full_ffn else 8,
        load_best_model_at_end=False,  # checkpoints are delta-only; best applied manually below
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(args.dtype == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        report_to=[],
        seed=args.seed,
        ignore_data_skip=False,
        ddp_find_unused_parameters=False,   # index_add puts every delta in the graph each microbatch
        ddp_broadcast_buffers=False,
        # FSDP's default no_sync accumulation retains full, unsharded gradients
        # until the final microbatch. At 32.2B trainable params that is ~64GB/GPU
        # and OOMs GA>1. Synchronising each microbatch keeps gradients sharded while
        # preserving the requested global batch and accumulated update.
        accelerator_config=(
            {"gradient_accumulation_kwargs": {"sync_each_batch": True}}
            if is_full_ffn else None
        ),
        **({"fsdp": fsdp, "fsdp_config": fsdp_config, "optim": optim_name}
           if is_full_ffn else {}),
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

    from transformers.optimization import Adafactor as _TransformersAdafactor

    class CheckpointableAdafactor(_TransformersAdafactor):
        """Adafactor state without the derived per-rank RMS scalar.

        `scale_parameter=False` makes RMS irrelevant to the learning rate, and
        Adafactor recomputes it from the current parameter before every update.
        FSDP1 cannot merge differing scalar RMS values across rank-local shards, so
        omitting only this derived value makes the standard sharded optimizer DCP
        saveable while retaining step and exp_avg_sq exactly.
        """

        def state_dict(self):
            if any(group.get("scale_parameter", False) for group in self.param_groups):
                raise RuntimeError(
                    "CheckpointableAdafactor requires scale_parameter=False"
                )
            state_dict = super().state_dict()
            state_dict["state"] = {
                param_id: {key: value for key, value in state.items() if key != "RMS"}
                for param_id, state in state_dict["state"].items()
            }
            return state_dict

    class FullFfnStandardTrainer(Trainer):
        """HF/Accelerate sharded FSDP checkpointing with an atomic completeness marker."""

        @staticmethod
        def _state_component_signatures(model, optimizer=None):
            """Digest local model and optimizer components independently.

            Probe checkpoints hash every local byte. Normal campaign checkpoints
            sample large tensors, retaining a cheap load-integrity audit. Optimizer
            scalar values are canonicalised separately so an Adafactor step stored
            as Python ``int`` and restored as a scalar tensor compare semantically.
            """
            import hashlib
            import numbers

            full_digest = os.environ.get("FULLFFN_PROBE") == "1"

            def tensor_bytes(digest, value):
                tensor = value.detach().contiguous().view(torch.uint8).reshape(-1)
                if not full_digest and tensor.numel() > 4096:
                    indices = torch.linspace(
                        0, tensor.numel() - 1, 4096,
                        device=tensor.device, dtype=torch.long,
                    )
                    tensor = tensor[indices]
                chunk = 64 * 1024 * 1024
                for start in range(0, tensor.numel(), chunk):
                    digest.update(
                        tensor[start:start + chunk].cpu().contiguous().numpy().tobytes()
                    )

            model_digest = hashlib.sha256()
            named_parameters = sorted(model.named_parameters(), key=lambda item: item[0])
            name_by_param = {}
            for name, parameter in named_parameters:
                name_by_param[id(parameter)] = name
                if not parameter.requires_grad or parameter.numel() == 0:
                    continue
                model_digest.update(
                    f"model:{name}:{parameter.dtype}:{tuple(parameter.shape)}".encode()
                )
                tensor_bytes(model_digest, parameter)

            result = {
                "mode": "full" if full_digest else "sampled",
                "model": model_digest.hexdigest(),
            }
            if optimizer is None:
                return result

            tensor_digest = hashlib.sha256()
            scalar_digest = hashlib.sha256()
            optimizer_states = list(optimizer.state.items())
            unnamed = [parameter for parameter, _ in optimizer_states
                       if id(parameter) not in name_by_param]
            if unnamed:
                raise RuntimeError(
                    f"optimizer contains {len(unnamed)} parameters absent from model names"
                )
            optimizer_states.sort(key=lambda item: name_by_param[id(item[0])])
            for state_index, (parameter, state) in enumerate(optimizer_states):
                name = name_by_param[id(parameter)]
                for key in sorted(state):
                    if key == "RMS":
                        continue
                    value = state[key]
                    prefix = f"optim:{name}:{key}"
                    if torch.is_tensor(value) and value.numel() != 1:
                        tensor_digest.update(
                            f"{prefix}:{value.dtype}:{tuple(value.shape)}".encode()
                        )
                        tensor_bytes(tensor_digest, value)
                    elif torch.is_tensor(value):
                        scalar = value.detach().cpu().item()
                        if key == "step":
                            scalar = int(scalar)
                        scalar_digest.update(f"{prefix}:{scalar!r}".encode())
                    elif isinstance(value, numbers.Number):
                        scalar = int(value) if key == "step" else value
                        scalar_digest.update(f"{prefix}:{scalar!r}".encode())
                    else:
                        scalar_digest.update(f"{prefix}:{value!r}".encode())

            for group_index, group in enumerate(optimizer.param_groups):
                for key in sorted(group):
                    if key == "params":
                        continue
                    scalar_digest.update(
                        f"group:{group_index}:{key}:{group[key]!r}".encode()
                    )
            result["optimizer_tensors"] = tensor_digest.hexdigest()
            result["optimizer_scalars"] = scalar_digest.hexdigest()
            result["optimizer_state_entries"] = len(optimizer.state)
            return result

        @staticmethod
        def _rng_signature():
            import hashlib
            import numpy as np

            digest = hashlib.sha256()
            digest.update(repr(random.getstate()).encode())
            np_state = np.random.get_state()
            digest.update(repr((np_state[0], np_state[2:])).encode())
            digest.update(np_state[1].tobytes())
            digest.update(torch.random.get_rng_state().numpy().tobytes())
            for state in torch.cuda.random.get_rng_state_all():
                digest.update(state.cpu().numpy().tobytes())
            return digest.hexdigest()

        @staticmethod
        def _gradient_signature(model):
            import hashlib

            digest = hashlib.sha256()
            none_count = 0
            tensor_count = 0
            for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
                if not parameter.requires_grad or parameter.numel() == 0:
                    continue
                grad = parameter.grad
                if grad is None:
                    none_count += 1
                    digest.update(f"{name}:None".encode())
                    continue
                tensor_count += 1
                digest.update(
                    f"{name}:{grad.dtype}:{tuple(grad.shape)}".encode()
                )
                raw = grad.detach().contiguous().view(torch.uint8).reshape(-1)
                chunk = 64 * 1024 * 1024
                for start in range(0, raw.numel(), chunk):
                    digest.update(
                        raw[start:start + chunk].cpu().contiguous().numpy().tobytes()
                    )
            return {
                "digest": digest.hexdigest(),
                "tensor_count": tensor_count,
                "none_count": none_count,
            }

        @staticmethod
        def _checkpoint_requirements(checkpoint, world_size):
            required = [
                os.path.join(checkpoint, "pytorch_model_fsdp_0", ".metadata"),
                os.path.join(checkpoint, "optimizer_0", ".metadata"),
                os.path.join(checkpoint, "scheduler.pt"),
                os.path.join(checkpoint, "trainer_state.json"),
            ]
            required.extend(
                os.path.join(checkpoint, f"rng_state_{rank}.pth")
                for rank in range(world_size)
            )
            return required

        def _save_checkpoint(self, model, trial):
            import datetime as dt
            import torch.distributed as dist
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint = os.path.join(
                self._get_output_dir(trial=trial),
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )
            marker_path = os.path.join(checkpoint, FULLFFN_CHECKPOINT_MARKER)
            if self.args.should_save and os.path.exists(marker_path):
                os.unlink(marker_path)
            if dist.is_initialized():
                dist.barrier()
            raw_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            local_components = self._state_component_signatures(model, raw_optimizer)
            super()._save_checkpoint(model, trial)
            if dist.is_initialized():
                dist.barrier()
            local_components_post = self._state_component_signatures(model, raw_optimizer)
            local_save_stable = local_components_post == local_components
            print(
                f"{rank_tag} [fullffn-save-live-audit] step={self.state.global_step} "
                f"match={local_save_stable} pre={local_components} "
                f"post={local_components_post}",
                flush=True,
            )

            optimizer_state_entries = [len(raw_optimizer.state)]
            state_components = [local_components]
            state_components_post = [local_components_post]
            if dist.is_initialized():
                optimizer_state_entries = [None] * int(self.args.world_size)
                dist.all_gather_object(optimizer_state_entries, len(raw_optimizer.state))
                state_components = [None] * int(self.args.world_size)
                dist.all_gather_object(state_components, local_components)
                state_components_post = [None] * int(self.args.world_size)
                dist.all_gather_object(state_components_post, local_components_post)
            if state_components_post != state_components:
                raise RuntimeError(
                    "Full-FFN checkpoint save mutated live model/optimizer state"
                )
            if self.args.should_save:
                required = self._checkpoint_requirements(checkpoint, int(self.args.world_size))
                missing = [path for path in required if not os.path.isfile(path)]
                if missing:
                    raise RuntimeError(
                        "Full-FFN checkpoint is incomplete; missing " + ", ".join(missing[:8])
                    )
                marker = {
                    "schema_version": 2,
                    "format": "hf_accelerate_fsdp_sharded_dcp",
                    "complete": True,
                    "global_step": int(self.state.global_step),
                    "world_size": int(self.args.world_size),
                    "optimizer": raw_optimizer.__class__.__name__,
                    "optimizer_state_entries_by_rank": optimizer_state_entries,
                    "state_components_by_rank": state_components,
                    "state_components_post_save_by_rank": state_components_post,
                    "adafactor_rms_checkpointed": False,
                    "adafactor_rms_policy": "recompute_each_step_scale_parameter_false",
                    "scheduler_last_epoch": int(self.lr_scheduler.state_dict().get("last_epoch", -1)),
                    "weight_decay": float(self.args.weight_decay),
                    "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                temp_path = marker_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(marker, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, marker_path)
                print(f"{rank_tag} [fullffn-checkpoint-save] {marker}", flush=True)
            if dist.is_initialized():
                dist.barrier()

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            marker_path = os.path.join(resume_from_checkpoint, FULLFFN_CHECKPOINT_MARKER)
            if not os.path.isfile(marker_path):
                raise RuntimeError(
                    f"refusing incomplete Full-FFN checkpoint (marker missing): {resume_from_checkpoint}"
                )
            with open(marker_path) as f:
                marker = json.load(f)
            if not marker.get("complete"):
                raise RuntimeError(f"incomplete Full-FFN checkpoint: {resume_from_checkpoint}")
            if int(marker.get("world_size", -1)) != int(self.args.world_size):
                raise RuntimeError(
                    f"exact resume requires world_size={marker.get('world_size')}, "
                    f"got {self.args.world_size}"
                )
            if float(marker.get("weight_decay", -1.0)) != float(self.args.weight_decay):
                raise RuntimeError(
                    f"weight_decay mismatch: checkpoint={marker.get('weight_decay')} "
                    f"run={self.args.weight_decay}"
                )
            trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
            with open(trainer_state_path) as f:
                saved_trainer_state = json.load(f)
            if int(saved_trainer_state.get("global_step", -1)) != int(marker["global_step"]):
                raise RuntimeError("trainer_state global_step does not match checkpoint marker")
            missing = [
                path for path in self._checkpoint_requirements(
                    resume_from_checkpoint, int(self.args.world_size)
                )
                if not os.path.isfile(path)
            ]
            if missing:
                raise RuntimeError(
                    "Full-FFN resume checkpoint is incomplete; missing " + ", ".join(missing[:8])
                )
            result = super()._load_from_checkpoint(resume_from_checkpoint, model)
            target_model = model if model is not None else self.model
            loaded_model = self._state_component_signatures(target_model)["model"]
            expected_components = marker.get("state_components_by_rank", [])
            if (os.environ.get("FULLFFN_PROBE") == "1"
                    and len(expected_components) != int(self.args.world_size)):
                raise RuntimeError("probe checkpoint lacks per-rank component signatures")
            expected_model = (
                expected_components[int(self.args.process_index)].get("model")
                if len(expected_components) == int(self.args.world_size) else None
            )
            model_match = expected_model is None or loaded_model == expected_model
            self._fullffn_load_audit = {
                "model_expected": expected_model,
                "model_loaded": loaded_model,
                "model_match": model_match,
            }
            print(
                f"{rank_tag} [fullffn-model-load] step={marker['global_step']} "
                f"model_match={model_match} expected={expected_model} loaded={loaded_model}",
                flush=True,
            )
            return result

        def _load_optimizer_and_scheduler(self, checkpoint):
            if not checkpoint or not os.path.isfile(
                os.path.join(checkpoint, FULLFFN_CHECKPOINT_MARKER)
            ):
                return super()._load_optimizer_and_scheduler(checkpoint)

            # Accelerate 1.14 constructs its SHARDED DCP load destination from the
            # fresh optimizer.state_dict(). Adafactor initializes state lazily, so
            # that destination has no state entries and DCP silently ignores the
            # saved exp_avg_sq tensors. Build the canonical optimizer state from
            # checkpoint metadata instead, following PyTorch's FSDP1 recipe.
            import torch.distributed.checkpoint as dist_cp
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            raw_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            plugin = self.accelerator.state.fsdp_plugin
            if plugin.fsdp_version != 1:
                raise RuntimeError(f"Full-FFN resume requires FSDP1, got {plugin.fsdp_version}")
            if plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
                raise RuntimeError(
                    "Full-FFN resume requires SHARDED_STATE_DICT, got "
                    f"{plugin.state_dict_type}"
                )
            optimizer_dir = os.path.join(checkpoint, "optimizer_0")
            scheduler_path = os.path.join(checkpoint, "scheduler.pt")
            self.accelerator.wait_for_everyone()
            with FSDP.state_dict_type(
                self.model,
                plugin.state_dict_type,
                plugin.state_dict_config,
                plugin.optim_state_dict_config,
            ):
                model_state = self.model.state_dict()
                canonical = dist_cp.load_sharded_optimizer_state_dict(
                    model_state_dict=model_state,
                    optimizer_key="optimizer",
                    storage_reader=dist_cp.FileSystemReader(optimizer_dir),
                )["optimizer"]
                local_state = FSDP.optim_state_dict_to_load(
                    model=self.model,
                    optim=raw_optimizer,
                    optim_state_dict=canonical,
                )
                raw_optimizer.load_state_dict(local_state)
            del model_state, canonical, local_state
            self.lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=True)
            )
            self.accelerator.wait_for_everyone()

            if checkpoint:
                with open(os.path.join(checkpoint, FULLFFN_CHECKPOINT_MARKER)) as f:
                    marker = json.load(f)
                loaded_entries = len(raw_optimizer.state)
                expected_by_rank = marker.get("optimizer_state_entries_by_rank", [])
                expected_entries = (
                    int(expected_by_rank[int(self.args.process_index)])
                    if len(expected_by_rank) == int(self.args.world_size) else -1
                )
                if expected_entries >= 0 and loaded_entries != expected_entries:
                    raise RuntimeError(
                        f"optimizer state restore mismatch: loaded={loaded_entries} "
                        f"expected={expected_entries}"
                    )
                loaded_components = self._state_component_signatures(
                    self.model, raw_optimizer
                )
                expected_components = marker.get("state_components_by_rank", [])
                if (os.environ.get("FULLFFN_PROBE") == "1"
                        and len(expected_components) != int(self.args.world_size)):
                    raise RuntimeError("probe checkpoint lacks optimizer component signatures")
                expected = (
                    expected_components[int(self.args.process_index)]
                    if len(expected_components) == int(self.args.world_size) else {}
                )
                tensor_match = (
                    not expected
                    or loaded_components.get("optimizer_tensors")
                    == expected.get("optimizer_tensors")
                )
                scalar_match = (
                    not expected
                    or loaded_components.get("optimizer_scalars")
                    == expected.get("optimizer_scalars")
                )
                audit = getattr(self, "_fullffn_load_audit", {})
                audit.update({
                    "optimizer_tensors_match": tensor_match,
                    "optimizer_scalars_match": scalar_match,
                })
                print(
                    f"{rank_tag} [fullffn-optimizer-load] states={loaded_entries} "
                    f"scheduler_last_epoch={self.lr_scheduler.state_dict().get('last_epoch')} "
                    f"model_match={audit.get('model_match')} "
                    f"optimizer_tensors_match={tensor_match} "
                    f"optimizer_scalars_match={scalar_match}",
                    flush=True,
                )
                if expected and not all((
                    audit.get("model_match"), tensor_match, scalar_match,
                )):
                    raise RuntimeError(
                        "Full-FFN checkpoint load integrity mismatch: "
                        f"{audit}"
                    )
            return None

        def training_step(self, model, inputs, num_items_in_batch=None):
            probe_record = None
            if os.environ.get("FULLFFN_PROBE") == "1":
                import hashlib

                micro = int(getattr(self, "_fullffn_probe_micro", 0))
                digest = hashlib.sha256()
                for key in ("input_ids", "labels"):
                    value = inputs.get(key)
                    if value is not None:
                        raw = value.detach().contiguous().view(torch.uint8).cpu()
                        digest.update(key.encode())
                        digest.update(raw.numpy().tobytes())
                probe_record = (
                    int(self.state.global_step),
                    micro % int(self.args.gradient_accumulation_steps),
                    digest.hexdigest(),
                )
                if probe_record[0] == 5 and probe_record[1] == 0:
                    print(
                        f"{rank_tag} [fullffn-phase-a] stage=RNG_BEFORE_FORWARD "
                        f"step=5 rng_sha256={self._rng_signature()} "
                        f"model_training={model.training}",
                        flush=True,
                    )
                self._fullffn_probe_micro = micro + 1

            loss = super().training_step(model, inputs, num_items_in_batch)
            if probe_record is not None:
                step, micro, batch_digest = probe_record
                loss_hex = float(loss.detach().float().cpu().item()).hex()
                print(
                    f"{rank_tag} [fullffn-step-input] step={step} micro={micro} "
                    f"batch_sha256={batch_digest} loss_hex={loss_hex}",
                    flush=True,
                )
            return loss

    trainer_cls = (DeltaTrainer if args.method == "delta" else
                   FullFfnStandardTrainer if is_full_ffn else Trainer)

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

    # FULLFFN_PROBE=1: per-layer grad-norm coverage for ALL 40 expert shards +
    # frozen-router/attn/embed assertions. Under FSDP use_orig_params the flat shards
    # still expose per-Parameter .grad, so we can measure each layer. Used by
    # run_fullffn_probe.sh (asserts 2/3/4A). Assertion 1 (peak mem) is printed at the
    # end of training; assertion 4B (router byte-equality) is checked offline on the
    # saved checkpoint by the probe script.
    if os.environ.get("FULLFFN_PROBE") == "1":
        from transformers import TrainerCallback
        _ep = list(getattr(handles, "expert_params", []) or [])

        class FullFfnProbe(TrainerCallback):
            def on_pre_optimizer_step(self, a, s, c, **kw):
                # assertion 3: every trainable expert Parameter has a non-zero grad.
                zero, none, total = 0, 0, 0
                for p in _ep:
                    total += 1
                    if p.grad is None:
                        none += 1
                    elif float(p.grad.detach().float().norm()) == 0.0:
                        zero += 1
                covered = total - none - zero
                # Under FSDP use_orig_params each rank's contiguous flat-shard slice
                # intersects only SOME orig params (a rank may own gate_up of a layer
                # but not down) -> local .grad None is EXPECTED. The correct assertion
                # is the UNION across ranks: every expert param must be covered by at
                # least one rank.
                cov = torch.tensor([0 if (p.grad is None or
                                          float(p.grad.detach().float().norm()) == 0.0)
                                    else 1 for p in _ep],
                                   dtype=torch.int32, device="cuda")
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(cov, op=torch.distributed.ReduceOp.MAX)
                union = int(cov.sum())
                print(f"{rank_tag} [FULLFFN_PROBE step={s.global_step}] "
                      f"expert_params={total} local_covered={covered} "
                      f"union_covered={union} grad_none(local)={none} grad_zero(local)={zero}",
                      flush=True)
                if union < total:
                    print(f"{rank_tag} [FULLFFN_PROBE] FAIL: union coverage {union}/{total} "
                          f"(some expert param got no gradient on ANY rank)", flush=True)
                elif s.global_step <= 1:
                    print(f"{rank_tag} [FULLFFN_PROBE] OK: union coverage {union}/{total}",
                          flush=True)

                if s.global_step == 5:
                    gradient = FullFfnStandardTrainer._gradient_signature(kw["model"])
                    print(
                        f"{rank_tag} [fullffn-phase-a] stage=GRAD6_AFTER_CLIP "
                        f"step=5 gradient={gradient}",
                        flush=True,
                    )

            def on_optimizer_step(self, a, s, c, **kw):
                if s.global_step != 5:
                    return
                optimizer = kw["optimizer"]
                raw_optimizer = getattr(optimizer, "optimizer", optimizer)
                components = FullFfnStandardTrainer._state_component_signatures(
                    kw["model"], raw_optimizer
                )
                step_types = {}
                for state in raw_optimizer.state.values():
                    value = state.get("step")
                    if torch.is_tensor(value):
                        key = f"tensor:{value.dtype}:{value.device.type}"
                    else:
                        key = type(value).__name__
                    step_types[key] = step_types.get(key, 0) + 1
                print(
                    f"{rank_tag} [fullffn-phase-a] stage=STEP6_POST_OPT "
                    f"step=5 components={components} step_types={step_types}",
                    flush=True,
                )

        _extra_callbacks.append(FullFfnProbe())

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=_extra_callbacks or None,
    )
    if optimizer is not None:
        trainer_kwargs["optimizers"] = (optimizer, None)  # scheduler built by Trainer
    if is_full_ffn:
        trainer_kwargs["optimizer_cls_and_kwargs"] = (
            CheckpointableAdafactor,
            {
                "lr": args.learning_rate,
                "scale_parameter": False,
                "relative_step": False,
                "warmup_init": False,
                "beta1": None,
            },
        )
    trainer = trainer_cls(**trainer_kwargs)
    if is_full_ffn:
        actual_state_dict_type = str(trainer.accelerator.state.fsdp_plugin.state_dict_type)
        if "SHARDED_STATE_DICT" not in actual_state_dict_type:
            raise RuntimeError(
                "Full-FFN requires FSDP SHARDED_STATE_DICT, got "
                f"{actual_state_dict_type}"
            )
        print(f"{rank_tag} FSDP state_dict_type={actual_state_dict_type}", flush=True)
    if trainer.accelerator.ddp_handler is not None:
        # grads live inside the DDP reduction buckets instead of a second copy
        trainer.accelerator.ddp_handler.gradient_as_bucket_view = True

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Always leave a resumable checkpoint at the requested target step, including
    # targets such as 300 that are not multiples of the regular 200-step interval.
    if is_full_ffn and not args.skip_final_checkpoint:
        import torch.distributed as dist
        final_ckpt = os.path.join(args.output_dir, f"checkpoint-{trainer.state.global_step}")
        needs_save = not os.path.isfile(os.path.join(final_ckpt, FULLFFN_CHECKPOINT_MARKER))
        if dist.is_initialized():
            decision = [needs_save if trainer.args.should_save else None]
            dist.broadcast_object_list(decision, src=0)
            needs_save = bool(decision[0])
        if needs_save:
            trainer._save_checkpoint(trainer.model, trial=None)

    dev = torch.device(f"cuda:{device_index}")
    print(f"{rank_tag} cuda max_memory_allocated={torch.cuda.max_memory_allocated(dev) / 2**30:.2f}GiB "
          f"max_memory_reserved={torch.cuda.max_memory_reserved(dev) / 2**30:.2f}GiB")

    # ---- FULLFFN_PROBE assertion 4 (part A): no frozen param received a gradient ----
    # Router/attn/embed have requires_grad=False so their .grad must stay None. The
    # stronger BYTE-EQUALITY check of the saved router weights vs base is done offline
    # on the consolidated checkpoint by run_fullffn_probe.sh step 2 (comparing after
    # the FSDP FULL_STATE_DICT gather is cleaner than summoning shards here).
    if is_full_ffn and os.environ.get("FULLFFN_PROBE") == "1":
        bad_grad = [n for n, p in model.named_parameters()
                    if not p.requires_grad and p.grad is not None]
        if bad_grad:
            print(f"{rank_tag} [FULLFFN_PROBE] FAIL: {len(bad_grad)} frozen params got a "
                  f"gradient, e.g. {bad_grad[:3]}", flush=True)
        else:
            print(f"{rank_tag} [FULLFFN_PROBE] OK: no frozen param received a gradient "
                  f"(router/attn/embed .grad all None)", flush=True)

    # ---- full-ffn final save: MUST run on ALL ranks (FULL_STATE_DICT gather is a
    # collective all-gather). Doing the early should_save return first would deadlock
    # rank0 waiting for the other ranks inside state_dict(). ----
    if args.method == "full-ffn":
        if args.skip_final_hf_export:
            if trainer.args.should_save:
                print(
                    f"training complete; resumable Full-FFN DCP checkpoint at "
                    f"{args.output_dir}/checkpoint-{trainer.state.global_step} "
                    "(final HF export skipped)",
                    flush=True,
                )
            return
        # Gather the FULL_STATE_DICT on rank0 (offload_to_cpu so the 70GB gather lands
        # in host RAM, not VRAM) and save_pretrained a normal HF model directory. eval
        # then runs unchanged via --model base --model-path <dir> --topk 32. gpu-host has
        # ~1.5TB free RAM so the gather fits; if a host cannot hold it, keep
        # SHARDED_STATE_DICT and consolidate offline with torch.distributed.checkpoint
        # (DCP) -> see IMPLEMENTATION_NOTES 'Save'.
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig,
        )
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = trainer.model.state_dict()   # collective: every rank participates
        if trainer.args.should_save:
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.save_pretrained(args.output_dir, state_dict=cpu_state,
                                      safe_serialization=True)
            with open(os.path.join(args.output_dir, "expert_cfg.json"), "w") as f:
                json.dump(expert_config, f, indent=1)
            tokenizer.save_pretrained(args.output_dir)
            print(f"training complete; full-ffn model -> {args.output_dir}")
        return

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
