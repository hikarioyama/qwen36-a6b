#!/usr/bin/env python
"""Phase-0 (GPU-gated): ESFT training for Qwen3.6-35B-A3B.

Adapts DeepSeek ESFT's train.py to the Qwen3_5Moe architecture:

  * freezing/masking via to_esft_qwen (packed-expert gradient masking), not the
    upstream buffer/param trick;
  * a custom optimiser param-group split (build_param_groups) that keeps the packed
    expert tensors in a weight_decay=0 group, so non-selected experts stay bit-exact
    frozen (upstream relies on them being buffers);
  * only the selected experts' slices are saved (save_expert_patch), giving a tiny
    ESFT checkpoint instead of the full 35B model.

Hyperparameters follow ESFT's configs/base.yaml: LR 1e-5, seq 4096, effective batch
32 (per-device batch * grad-accum * #devices), <=500 steps, eval/save every 100,
load-best-at-end on eval loss.

Run (GPU phase):
    <venv>/bin/python train_esft.py \
        --model Qwen/Qwen3.6-35B-A3B \
        --expert-config configs/math_token_p0.2.json \
        --train-data data/train/math.jsonl \
        --output-dir runs/math_esft
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def render_and_tokenize(tokenizer, messages, mask_prompt=True, ignore_id=-100):
    """Tokenise a conversation with the Qwen chat template.

    Builds labels turn-by-turn: assistant turns are supervised, everything else
    (system/user, and the template scaffolding preceding each assistant turn) is
    masked to ``ignore_id`` when ``mask_prompt`` is set. This mirrors ESFT's
    prompt-masking while respecting the model's own chat template.
    """
    input_ids, labels = [], []
    prev_len = 0
    prev_text = ""
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
        prev_text = text
    return input_ids, labels


def pack_examples(records, tokenizer, seq_length, random_concat_ratio, seed, ignore_id=-100):
    """Concatenate tokenised conversations into fixed-length (seq_length) blocks.

    Follows ESFT's get_examples_from_buffer_pad: greedily fill blocks, occasionally
    dropping the leading token of a concatenated example (random_concat_ratio), pad
    the trailing block.
    """
    rng = random.Random(seed)
    all_in, all_lab = [], []
    cur_in, cur_lab = [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for rec in records:
        iid, lab = render_and_tokenize(tokenizer, rec["messages"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--expert-config", required=True)
    ap.add_argument("--train-data", required=True, help="ESFT-format jsonl ({'messages': [...]})")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="applied to non-expert trainable params only; experts stay wd=0")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32,
                    help="per_device_batch * grad_accum * n_gpus should be ~32")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--random-concat-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=5934875)
    ap.add_argument("--max-records", type=int, default=0, help="cap training records (0=all)")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText,
        TrainingArguments, Trainer,
    )
    from torch.utils.data import TensorDataset

    from esft_qwen.esft_patch import to_esft_qwen, build_param_groups, save_expert_patch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    expert_config = {k: v for k, v in json.load(open(args.expert_config)).items()
                     if not k.startswith("_")}

    # ---- data ----
    records = []
    with open(args.train_data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if args.max_records and len(records) >= args.max_records:
                    break
    all_in, all_lab = pack_examples(records, tokenizer, args.seq_length,
                                    args.random_concat_ratio, args.seed)
    input_ids = torch.tensor(all_in, dtype=torch.long)
    labels = torch.tensor(all_lab, dtype=torch.long)
    dataset = TensorDataset(input_ids, labels)
    n_val = max(1, int(len(dataset) * 0.02))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    print(f"packed {len(dataset)} blocks of {args.seq_length} tokens "
          f"(train {len(train_ds)} / val {len(val_ds)})")

    # ---- model ----
    config = AutoConfig.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, config=config, dtype=dtype, device_map="auto")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=dtype, device_map="auto")
    model.config.use_cache = False

    handles = to_esft_qwen(model, expert_config)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ESFT trainable params: {n_train:,}")

    # Custom optimiser: packed expert params in a weight_decay=0 group.
    param_groups = build_param_groups(model, handles, weight_decay=args.weight_decay)
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
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(args.dtype == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        report_to=[],
    )

    def data_collator(data):
        return {"input_ids": torch.stack([d[0] for d in data]),
                "labels": torch.stack([d[1] for d in data])}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # scheduler built by Trainer (constant)
    )
    trainer.train()

    # Save the tiny ESFT patch (selected experts only) + config + tokenizer.
    os.makedirs(args.output_dir, exist_ok=True)
    patch_path = os.path.join(args.output_dir, "expert_patch.safetensors")
    save_expert_patch(model, expert_config, patch_path)
    with open(os.path.join(args.output_dir, "expert_cfg.json"), "w") as f:
        json.dump(expert_config, f, indent=1)
    tokenizer.save_pretrained(args.output_dir)
    print(f"training complete; patch -> {patch_path}")


if __name__ == "__main__":
    main()
