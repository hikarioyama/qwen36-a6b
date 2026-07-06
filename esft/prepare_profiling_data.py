#!/usr/bin/env python
"""Phase-0 (CPU): build profiling + training data for ESFT on Qwen3.6-35B-A3B.

For each domain (math / coding / japanese):

  * data/profiling/{domain}.jsonl  -- N packed blocks of exactly SEQ_LENGTH tokens
        (chat-template applied, tokenised). Used by collect_router_stats.py to
        profile expert routing. {"input_ids": [...], "n_tokens": SEQ_LENGTH}.

  * data/train/{domain}.jsonl      -- ESFT-format training records, one per source
        sample: {"messages": [{"role","content"}, ...]}. Raw text (tokenised later
        by train_esft.py), so writing the full corpus is cheap.

Tokeniser: loaded from a local snapshot (default: the Qwen3.6-35B-A3B-FP8 cache,
which ships the full tokeniser + chat_template.jinja). No GPU, no model weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

# CPU only.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DATASETS_ROOT = "/mnt/data/datasets/esft"

# domain -> list of (source_file, loader_name)
DOMAIN_SOURCES = {
    "math": [(f"{DATASETS_ROOT}/meta-math_MetaMathQA/MetaMathQA-395K.json", "metamath")],
    "coding": [(f"{DATASETS_ROOT}/theblackcat102_evol-codealpaca-v1/train.jsonl", "instruction_output")],
    "japanese": [
        (f"{DATASETS_ROOT}/llm-jp_magpie-sft-v1.0/magpie-sft-v1.0.jsonl", "conversations"),
        (f"{DATASETS_ROOT}/llm-jp_oasst2-33k-ja/oasst2-33k-ja.jsonl", "conversations"),
    ],
    "toolcall": [
        (f"{DATASETS_ROOT}/xlam-fc-60k/xlam-function-calling-60k.parquet", "xlam_toolcall"),
    ],
}


def default_tokenizer_path() -> str:
    for pat in (
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/*"),
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*"),
    ):
        hits = glob.glob(pat)
        for h in hits:
            if os.path.exists(os.path.join(h, "tokenizer_config.json")):
                return h
    return "Qwen/Qwen3.6-35B-A3B"


# --------------------------------------------------------------------------- #
# Loaders: each yields either a `messages` list [{"role","content"}, ...] per
# sample, or a `(messages, tools)` tuple when the domain carries a per-sample
# tool schema (tool-calling). `iter_domain_messages` normalises both shapes to
# `(messages, tools)`, with tools=None for the plain-chat domains.
# --------------------------------------------------------------------------- #
def _iter_json_or_jsonl(path):
    if path.endswith(".json"):
        with open(path) as f:
            for obj in json.load(f):
                yield obj
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_metamath(path):
    for obj in _iter_json_or_jsonl(path):
        q, r = obj.get("query"), obj.get("response")
        if q and r:
            yield [{"role": "user", "content": str(q)},
                   {"role": "assistant", "content": str(r)}]


def load_instruction_output(path):
    for obj in _iter_json_or_jsonl(path):
        ins, out = obj.get("instruction"), obj.get("output")
        if ins and out:
            yield [{"role": "user", "content": str(ins)},
                   {"role": "assistant", "content": str(out)}]


def load_conversations(path):
    for obj in _iter_json_or_jsonl(path):
        conv = obj.get("conversations") or obj.get("messages")
        if not conv:
            continue
        msgs = [{"role": m["role"], "content": str(m["content"])}
                for m in conv if m.get("role") in ("user", "assistant") and m.get("content")]
        if len(msgs) >= 2:
            yield msgs


def _normalize_tool_call_msgs(raw):
    """Coerce xLAM-style messages into what the Qwen chat template expects.

    Two fixes: assistant ``content`` may be null (template renders none as ''),
    and tool_call ``arguments`` ships as a JSON *string* while the template does
    ``arguments|items`` (needs a mapping). Parse the string to a dict.
    """
    out = []
    for m in raw:
        role = m["role"]
        content = m.get("content")
        nm = {"role": role, "content": content if content is not None else ""}
        tcs = m.get("tool_calls")
        if tcs:
            norm = []
            for tc in tcs:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (ValueError, TypeError):
                        args = {}
                norm.append({"type": "function",
                             "function": {"name": fn["name"], "arguments": args}})
            nm["tool_calls"] = norm
        out.append(nm)
    return out


def load_xlam_toolcall(path):
    """Yield (messages, tools) from the xLAM function-calling parquet.

    ``tools`` is the OpenAI-schema list of available functions (JSON string in
    the source), passed to the chat template so the routing context includes the
    tool definitions -- not just the tool_call text.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["messages", "tools"])
    msgs_col = table.column("messages").to_pylist()
    tools_col = table.column("tools").to_pylist()
    for raw_msgs, raw_tools in zip(msgs_col, tools_col):
        if not raw_msgs or not raw_tools:
            continue
        try:
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        except (ValueError, TypeError):
            continue
        if not tools:
            continue
        msgs = _normalize_tool_call_msgs(raw_msgs)
        if len(msgs) >= 2:
            yield (msgs, tools)


LOADERS = {
    "metamath": load_metamath,
    "instruction_output": load_instruction_output,
    "conversations": load_conversations,
    "xlam_toolcall": load_xlam_toolcall,
}


def iter_domain_messages(domain):
    """Yield ``(messages, tools)`` per sample; tools is None for plain-chat domains."""
    for path, loader_name in DOMAIN_SOURCES[domain]:
        if not os.path.exists(path):
            print(f"  WARN: missing source {path}, skipping")
            continue
        for item in LOADERS[loader_name](path):
            if isinstance(item, tuple):
                yield item
            else:
                yield (item, None)


def tokenize_chat(tokenizer, msgs, tools=None) -> list[int]:
    """Render a conversation with the chat template and tokenise to a flat id list.

    The Qwen3.6 multimodal processor's ``apply_chat_template(tokenize=True)`` returns
    a non-standard BatchEncoding wrapping a fast-tokenizer ``Encoding`` object, so we
    render to text first (tokenize=False) and tokenise the string. The template
    already emits the special tokens, hence ``add_special_tokens=False``.

    ``tools`` (list of OpenAI function schemas) is passed through to the template so
    the tool-calling domain keeps the available-function definitions in context.
    Plain-chat domains pass tools=None and render exactly as before.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools is not None:
        kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(msgs, **kwargs)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# --------------------------------------------------------------------------- #
def build_profiling(domain, tokenizer, seq_length, n_blocks, seed, max_scan):
    """Pack chat-formatted token streams into `n_blocks` blocks of `seq_length`."""
    rng = random.Random(seed)
    # Reservoir-ish: scan up to max_scan samples, shuffle, then pack in order.
    samples = []
    for i, item in enumerate(iter_domain_messages(domain)):
        samples.append(item)
        if len(samples) >= max_scan:
            break
    rng.shuffle(samples)

    blocks = []
    buf: list[int] = []
    for msgs, tools in samples:
        ids = tokenize_chat(tokenizer, msgs, tools)
        buf.extend(ids)
        while len(buf) >= seq_length:
            blocks.append(buf[:seq_length])
            buf = buf[seq_length:]
            if len(blocks) >= n_blocks:
                return blocks
    return blocks  # may be < n_blocks if corpus tiny


def write_train_jsonl(domain, out_path, max_samples):
    n = 0
    with open(out_path, "w") as f:
        for msgs, tools in iter_domain_messages(domain):
            rec = {"messages": msgs}
            if tools is not None:
                # keep the tool schemas so train_esft.py can render them back into
                # context; without them the tool-calling domain loses its meaning.
                rec["tools"] = tools
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if max_samples and n >= max_samples:
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=default_tokenizer_path())
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--domains", nargs="+", default=list(DOMAIN_SOURCES))
    ap.add_argument("--n-profiling-blocks", type=int, default=32)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=5934875)
    ap.add_argument("--max-scan", type=int, default=20000,
                    help="max source samples to scan when packing profiling blocks")
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="cap training jsonl per domain (0 = full corpus)")
    ap.add_argument("--skip-train", action="store_true", help="only build profiling data")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"tokenizer <- {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    prof_dir = os.path.join(args.out_dir, "profiling")
    train_dir = os.path.join(args.out_dir, "train")
    os.makedirs(prof_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    stats = {}
    for domain in args.domains:
        print(f"\n=== domain: {domain} ===")
        blocks = build_profiling(domain, tok, args.seq_length, args.n_profiling_blocks,
                                 args.seed, args.max_scan)
        pp = os.path.join(prof_dir, f"{domain}.jsonl")
        with open(pp, "w") as f:
            for b in blocks:
                f.write(json.dumps({"input_ids": b, "n_tokens": len(b)}) + "\n")
        prof_tokens = sum(len(b) for b in blocks)
        print(f"  profiling: {len(blocks)} blocks x {args.seq_length} = {prof_tokens} tokens -> {pp}")

        n_train = 0
        if not args.skip_train:
            tp = os.path.join(train_dir, f"{domain}.jsonl")
            n_train = write_train_jsonl(domain, tp, args.max_train_samples)
            print(f"  train: {n_train} records -> {tp}")

        stats[domain] = {"profiling_blocks": len(blocks), "profiling_tokens": prof_tokens,
                         "train_records": n_train}

    with open(os.path.join(args.out_dir, "prepare_stats.json"), "w") as f:
        json.dump({"tokenizer": args.tokenizer, "seq_length": args.seq_length, "domains": stats}, f, indent=2)
    print("\nSUMMARY:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
