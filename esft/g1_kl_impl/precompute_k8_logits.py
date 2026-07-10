#!/usr/bin/env python
"""Precompute the k=8 (base) teacher's top-64 next-token logits for a packed cache.

G1 offline-teacher KL distillation. The base Qwen3.6-35B-A3B (routed at its native
top_k=8) is the teacher whose knowledge routing is intact; we distil its output
distribution into the k=32 student (same weights + selected deltas) so widening the
router does not degrade knowledge. Storing the teacher's logits ONCE (no live teacher
beside the student) sidesteps the co-residence VRAM problem entirely.

For every block of the packed cache we run the frozen base model
(backbone -> hidden -> lm_head) and keep the TOP-64 logits + their vocab indices at
each of the seq_length positions. The student later forms its own logits on exactly
those 64 indices via a partial lm_head matmul (see train_esft.py compute_kl_term), so
the KL is taken over a shared truncated support -- no full [seq x vocab] tensor is ever
persisted or reloaded.

8-GPU split: launch one process per GPU with CUDA_VISIBLE_DEVICES set (NOT torchrun);
each process is handed a contiguous block range via --rank/--world-size and writes its
own shards, so there is no collective and a crashed rank resumes independently.

Output layout (per --out-dir):
    manifest.json                      {chunk_size, num_blocks, top_k, seq_length, dtype}
    shard_{start:08d}.safetensors      {"logits": (n,S,64) bf16, "indices": (n,S,64) int32}
where start is the GLOBAL block index of the shard's first block. train_esft's
TeacherLogitStore maps block i -> shard (i//chunk_size)*chunk_size, row i%chunk_size.
Shards already on disk are skipped (resume-safe).

Size estimate (v3 cache, S=7168, top_k=64, ~21.8k blocks):
    per block  = S*64*(2 bytes bf16 + 4 bytes int32) = 7168*64*6  = 2.75 MB
    total      ~= 21800 * 2.75 MB                                  ~= 60 GB
(logits ~20 GB bf16 + indices ~40 GB int32). Keep on the docker-raid, not /tmp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model dir (frozen teacher)")
    ap.add_argument("--cache-file", required=True,
                    help="packed cache .pt (tensor dict {'input_ids','labels': (N,S)}) -- "
                         "the SAME file the student trains on, so block i lines up")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=0, help="this process's shard rank")
    ap.add_argument("--world-size", type=int, default=1, help="number of GPU processes")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="blocks per shard file (finer = more resume granularity)")
    ap.add_argument("--teacher-top-k", type=int, default=8,
                    help="gate.top_k the teacher routes at (base k=8 is the intact path)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pos-chunk", type=int, default=1024,
                    help="seq positions per lm_head+topk chunk; caps the transient "
                         "[pos_chunk x vocab] logits tensor (7168x248320 bf16 ~= 3.5GB at full)")
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
    )
    from safetensors.torch import save_file as st_save
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esft_qwen.common import find_moe_blocks

    dtype = getattr(torch, args.dtype)
    os.makedirs(args.out_dir, exist_ok=True)

    blob = torch.load(args.cache_file, weights_only=True)
    input_ids_all = blob["input_ids"]           # (N,S) long
    N, S = input_ids_all.shape
    print(f"[rank{args.rank}] cache {args.cache_file}: {N} blocks x {S}", flush=True)

    # rank0 writes the manifest (idempotent) so TeacherLogitStore can self-describe.
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if args.rank == 0 and not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            json.dump({"chunk_size": args.chunk_size, "num_blocks": int(N),
                       "top_k": args.top_k, "seq_length": int(S),
                       "dtype": args.dtype, "teacher_top_k": args.teacher_top_k}, f, indent=1)
        print(f"[rank{args.rank}] wrote manifest {manifest_path}", flush=True)

    # contiguous block range for this rank
    per = (N + args.world_size - 1) // args.world_size
    lo = args.rank * per
    hi = min(N, lo + per)
    if lo >= N:
        print(f"[rank{args.rank}] no blocks assigned (lo={lo} >= N={N})", flush=True)
        return

    # align the rank's own work to chunk boundaries so each shard is written by exactly
    # one rank even if `per` is not a multiple of chunk_size: a rank owns a shard iff the
    # shard's start falls in its [lo,hi) range.
    config = AutoConfig.from_pretrained(args.model)
    load_kw = dict(config=config, dtype=dtype, device_map={"": 0})
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.eval()
    model.config.use_cache = False

    refs = find_moe_blocks(model)
    for ref in refs:
        ref.gate.top_k = args.teacher_top_k
    print(f"[rank{args.rank}] teacher top_k -> {args.teacher_top_k} on {len(refs)} layers",
          flush=True)

    backbone = model.model.language_model
    lm_head = model.lm_head
    device = torch.device("cuda:0")

    # iterate shard-aligned starts that begin inside this rank's range
    first_shard = (lo // args.chunk_size) * args.chunk_size
    n_written_shards = 0
    for start in range(first_shard, hi, args.chunk_size):
        if start < lo:
            continue  # belongs to the previous rank
        end = min(start + args.chunk_size, N)
        shard_path = os.path.join(args.out_dir, f"shard_{start:08d}.safetensors")
        if os.path.exists(shard_path):
            print(f"[rank{args.rank}] skip existing {os.path.basename(shard_path)}", flush=True)
            continue
        n = end - start
        logits_out = torch.empty((n, S, args.top_k), dtype=dtype)   # cpu
        indices_out = torch.empty((n, S, args.top_k), dtype=torch.int32)
        for j in range(n):
            b = start + j
            ids = input_ids_all[b:b + 1].to(device)                 # (1,S)
            with torch.no_grad():
                hidden = backbone(input_ids=ids, use_cache=False).last_hidden_state  # (1,S,H)
                hidden = hidden[0]                                  # (S,H)
                # position-chunk the lm_head matmul so the [pos x vocab] logits never
                # exceed pos_chunk rows at once.
                for ps in range(0, S, args.pos_chunk):
                    pe = min(ps + args.pos_chunk, S)
                    lg = torch.nn.functional.linear(hidden[ps:pe], lm_head.weight)  # (c,V)
                    tv, ti = torch.topk(lg, args.top_k, dim=-1)     # (c,64)
                    logits_out[j, ps:pe] = tv.to(dtype).cpu()
                    indices_out[j, ps:pe] = ti.to(torch.int32).cpu()
        st_save({"logits": logits_out.contiguous(), "indices": indices_out.contiguous()},
                shard_path,
                metadata={"start": str(start), "end": str(end), "top_k": str(args.top_k)})
        n_written_shards += 1
        print(f"[rank{args.rank}] wrote {os.path.basename(shard_path)} "
              f"blocks[{start}:{end}]", flush=True)
    print(f"[rank{args.rank}] done: {n_written_shards} shard(s) in [{lo},{hi})", flush=True)


if __name__ == "__main__":
    main()
