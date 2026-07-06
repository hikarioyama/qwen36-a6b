#!/usr/bin/env python
"""BFCL-style tool-call evaluation for the ESFT toolcall domain (Qwen3.6-35B-A3B).

Independent from ``eval_harness.py`` (it may be *imported* for the proven model
load / 2-GPU machinery, but is never edited). Measures four things the toolcall
consistency target cares about:

  * format-validity : did the model emit a well-formed tool call whose function
                      name actually exists in the provided ``tools``?
  * name accuracy   : did it pick the right function(s)? (multiset over calls)
  * argument AST    : do the arguments match gold, structurally + type-coerced,
                      order-independent (BFCL "AST" style)?
  * over-trigger    : on questions that need NO tool (tools present but irrelevant),
                      how often does it call one anyway? (lower is better)

GROUND TRUTH (verified against the shipped chat template, not assumed): Qwen3.6
serializes a call NOT as JSON but as a nested XML block ::

    <tool_call>
    <function=FUNC_NAME>
    <parameter=PARAM_NAME>
    VALUE
    </parameter>
    </function>
    </tool_call>

and the eval prompt is ``apply_chat_template(user_msgs, tools=tools,
add_generation_prompt=True)`` which injects a ``# Tools`` system block + the format
rules. The reasoning model opens ``<think>`` in the generation prompt, so answers
are parsed from the post-``</think>`` segment. ``<tool_call>``/``</think>`` are
added tokens that SURVIVE ``skip_special_tokens=True`` (verified), so decoding
matches eval_harness.

Held-out hygiene: positives are a fixed seeded sample of ``toolcall.jsonl`` indices
(``heldout_positive_indices``); negatives are the HEAD slice of the general-domain
train files. The balanced-training builder excludes exactly these, so eval never
scores memorised rows. See ``run_toolcall.sh`` / the balanced-data builder.

Run (real eval, both GPUs; the orchestrator launches this):
    ~/esft-work/venv/bin/python ~/esft/eval_toolcall.py \
        --model patched --patch runs/toolcall_esft_k32/expert_patch.safetensors \
        --topk 8 --n-pos 500 --n-neg 300 --tag toolcall_patched

CPU check only (no GPU, no model):
    ~/esft-work/venv/bin/python ~/esft/eval_toolcall.py --self-test
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import random
import argparse
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# eval_harness's module-level imports are stdlib-only (torch is lazy inside its
# functions), so importing it here is CPU-safe. We reuse its proven model spec /
# loader / EOS ids / CI helper -- never edit it.
from eval_harness import (  # noqa: E402
    resolve_model_spec, load_subject_model, EOS_IDS, ci95_normal,
    MODEL_35B, REPORT_DIR,
)

TOOLCALL_DATA = os.path.join(PROJECT_ROOT, "data", "train", "toolcall.jsonl")
GENERAL_SOURCES = [
    ("japanese", os.path.join(PROJECT_ROOT, "data", "train", "japanese.jsonl")),
    ("coding", os.path.join(PROJECT_ROOT, "data", "train", "coding.jsonl")),
    ("math", os.path.join(PROJECT_ROOT, "data", "train", "math.jsonl")),
]

# --- held-out contract (shared with the balanced-data builder) --------------
HELDOUT_SEED = 20260703
N_HELDOUT_POS = 600            # positives reserved from toolcall.jsonl for eval
EVAL_NEG_HEAD_PER_SOURCE = 400  # negatives drawn from the HEAD of each general file
# the builder must skip at least this many head lines per source (disjointness):
BUILDER_NEG_SKIP_HEAD = 1000


# =========================== tool-call parsing ==============================
# Qwen3.6 XML tool-call format (see module docstring). NOT json.
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def post_think(text: str) -> str:
    """The answer segment: everything after the last ``</think>`` (whole text if
    the model never closed the think block)."""
    return text.split("</think>")[-1] if "</think>" in text else text


def parse_tool_calls(text: str) -> list[dict]:
    """Parse every ``<tool_call>`` block into ``{"name", "arguments": {p: str}}``.

    Values are raw strings (the XML format is untyped); type coercion happens at
    comparison time in :func:`_deep_equal`. Malformed blocks (no ``<function=...>``)
    are skipped rather than raising, so a truncated tail simply drops out.
    """
    calls = []
    for block in _TOOLCALL_RE.findall(text):
        fm = _FUNC_RE.search(block)
        if not fm:
            continue
        name = fm.group(1).strip()
        args = {}
        for pname, pval in _PARAM_RE.findall(fm.group(2)):
            args[pname.strip()] = pval.strip()
        calls.append({"name": name, "arguments": args})
    return calls


# ---------------------- argument AST comparison -----------------------------
def _scalar_equal(gold, pred) -> bool:
    """Compare a typed gold scalar against a (usually string) predicted value with
    BFCL-style type coercion."""
    if isinstance(gold, bool):
        if isinstance(pred, bool):
            return gold == pred
        return str(pred).strip().lower() == str(gold).lower()
    if isinstance(gold, (int, float)):
        try:
            return abs(float(gold) - float(str(pred).strip())) < 1e-9
        except (ValueError, TypeError):
            return False
    if gold is None:
        return str(pred).strip().lower() in ("none", "null", "")
    return str(gold).strip() == str(pred).strip()


def _deep_equal(gold, pred) -> bool:
    """Order-independent, type-coercing structural equality. ``pred`` leaves that
    should be containers may arrive as JSON strings (from the XML format)."""
    if isinstance(gold, dict):
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except (ValueError, TypeError):
                return False
        if not isinstance(pred, dict) or set(gold) != set(pred):
            return False
        return all(_deep_equal(gold[k], pred[k]) for k in gold)
    if isinstance(gold, list):
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except (ValueError, TypeError):
                return False
        if not isinstance(pred, list) or len(gold) != len(pred):
            return False
        return all(_deep_equal(g, p) for g, p in zip(gold, pred))
    return _scalar_equal(gold, pred)


def match_calls(gold_calls: list[dict], pred_calls: list[dict]):
    """(name_match, arg_match). name_match = same multiset of function names.
    arg_match = a perfect bijection where names AND arguments both match (handles
    repeated same-name calls with different args)."""
    name_match = sorted(g["name"] for g in gold_calls) == sorted(p["name"] for p in pred_calls)
    if len(gold_calls) != len(pred_calls):
        return name_match, False
    used = [False] * len(pred_calls)

    def backtrack(i):
        if i == len(gold_calls):
            return True
        for j, p in enumerate(pred_calls):
            if (not used[j] and p["name"] == gold_calls[i]["name"]
                    and _deep_equal(gold_calls[i]["arguments"], p["arguments"])):
                used[j] = True
                if backtrack(i + 1):
                    return True
                used[j] = False
        return False

    return name_match, backtrack(0)


def format_valid(pred_calls: list[dict], tool_names) -> bool:
    """Well-formed: at least one parsed call, and every call names a function that
    actually exists in the provided tools (no hallucinated names)."""
    if not pred_calls:
        return False
    names = set(tool_names)
    return all(c["name"] in names for c in pred_calls)


def score_item(item: dict, gen_text: str) -> dict:
    """Score one generation. Positives -> format/name/arg flags; negatives ->
    over_trigger flag (any tool-call attempt in the answer segment)."""
    seg = post_think(gen_text)
    calls = parse_tool_calls(seg)
    if item["type"] == "pos":
        nm, am = match_calls(item["gold_calls"], calls)
        return {"format_valid": format_valid(calls, item["tool_names"]),
                "name_match": nm, "arg_match": am}
    over = ("<tool_call>" in seg) or bool(calls)
    return {"over_trigger": over}


# ============================ eval-set building =============================
def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def heldout_positive_indices(total: int, n: int = N_HELDOUT_POS) -> set:
    """Deterministic seeded held-out index set into toolcall.jsonl. The balanced
    builder excludes exactly these so eval positives are never trained on."""
    n = min(n, total)
    return set(random.Random(HELDOUT_SEED).sample(range(total), n))


def _split_positive(rec):
    """(prompt_messages, gold_calls) for a positive record: the turns up to (and
    excluding) the first assistant turn that carries tool_calls, and its calls."""
    msgs = rec["messages"]
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            gold = [{"name": tc["function"]["name"],
                     "arguments": tc["function"]["arguments"]} for tc in m["tool_calls"]]
            return msgs[:i], gold
    return None, None


def load_tool_pool(max_pool: int = 4000) -> list:
    """Unique tool schemas seen across toolcall.jsonl (dedup by serialized function),
    used to give negatives a realistic, in-distribution tools catalog."""
    pool, seen = [], set()
    for rec in _read_jsonl(TOOLCALL_DATA):
        for t in rec.get("tools", []):
            key = json.dumps(t.get("function", t), sort_keys=True)
            if key not in seen:
                seen.add(key)
                pool.append(t)
        if len(pool) >= max_pool:
            break
    return pool


def sample_tools(rng: random.Random, pool: list, k_min=1, k_max=4) -> list:
    k = rng.randint(k_min, min(k_max, len(pool)))
    return rng.sample(pool, k)


def iter_general_records(skip_head: int, per_source: int,
                         max_user_chars=1500, max_asst_chars=4000):
    """Yield ``(domain, [user, assistant])`` from the general-domain train files:
    the negative source (a real question answered directly, no tool). ``skip_head``
    reserves the eval/builder split; length caps keep negatives from ballooning."""
    for domain, path in GENERAL_SOURCES:
        if not os.path.exists(path):
            continue
        taken = 0
        for i, rec in enumerate(_read_jsonl(path)):
            if i < skip_head:
                continue
            msgs = rec.get("messages", [])
            if len(msgs) < 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
                continue
            u, a = msgs[0].get("content", ""), msgs[1].get("content", "")
            if not u or not a or "<tool_call>" in a:
                continue
            if len(u) > max_user_chars or len(a) > max_asst_chars:
                continue
            yield domain, [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            taken += 1
            if per_source and taken >= per_source:
                break


def build_eval_items(n_pos: int, n_neg: int) -> list:
    """Assemble held-out eval items. Positives from the seeded toolcall slice;
    negatives from the HEAD of the general files with sampled (irrelevant) tools."""
    items = []

    # positives -----------------------------------------------------------
    all_pos = list(_read_jsonl(TOOLCALL_DATA))
    held = heldout_positive_indices(len(all_pos))
    picked = 0
    for idx in sorted(held):
        if picked >= n_pos:
            break
        rec = all_pos[idx]
        pm, gold = _split_positive(rec)
        if pm is None or not gold:
            continue
        tool_names = [t["function"]["name"] for t in rec.get("tools", [])]
        items.append({"type": "pos", "prompt_messages": pm, "tools": rec.get("tools", []),
                      "gold_calls": gold, "tool_names": tool_names})
        picked += 1

    # negatives -----------------------------------------------------------
    rng = random.Random(HELDOUT_SEED + 1)
    pool = load_tool_pool()
    by_domain = {}
    for domain, msgs in iter_general_records(skip_head=0, per_source=EVAL_NEG_HEAD_PER_SOURCE):
        by_domain.setdefault(domain, []).append(msgs)
    # round-robin across domains so the negative mix is balanced, not all-japanese
    ordered, idx = [], 0
    while len(ordered) < n_neg and any(idx < len(v) for v in by_domain.values()):
        for domain in by_domain:
            if idx < len(by_domain[domain]) and len(ordered) < n_neg:
                ordered.append((domain, by_domain[domain][idx]))
        idx += 1
    for domain, msgs in ordered:
        tools = sample_tools(rng, pool)
        items.append({"type": "neg", "prompt_messages": [msgs[0]], "tools": tools,
                      "tool_names": [t["function"]["name"] for t in tools], "domain": domain})
    return items


def format_prompt(item: dict, tok) -> str:
    """Render an item to a generation-ready prompt: the user turn(s) + the tools
    catalog, with the assistant generation prompt appended."""
    return tok.apply_chat_template(
        item["prompt_messages"], tools=item["tools"],
        add_generation_prompt=True, tokenize=False)


# ============================ generation / scoring ==========================
def _count_gen_tokens(row_ids):
    hit = [row_ids.index(e) for e in EOS_IDS if e in row_ids]
    return (min(hit) + 1) if hit else len(row_ids)


def run_items(gpu_id, tok, model, items, batch_size, max_new, keep_samples=4):
    import torch
    dev = f"cuda:{gpu_id}"
    rendered = [format_prompt(it, tok) for it in items]
    agg = {"n_pos": 0, "format_valid": 0, "name_match": 0, "arg_match": 0,
           "n_neg": 0, "over_trigger": 0, "gen_tokens": 0}
    samples = []
    t0 = time.time()
    done = 0
    for i in range(0, len(rendered), batch_size):
        chunk = rendered[i:i + batch_size]
        ichunk = items[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False,
                eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)
        gen = out[:, in_len:]
        for row, item in zip(gen, ichunk):
            gt = row.tolist()
            agg["gen_tokens"] += _count_gen_tokens(gt)
            text = tok.decode(row, skip_special_tokens=True)
            sc = score_item(item, text)
            if item["type"] == "pos":
                agg["n_pos"] += 1
                agg["format_valid"] += int(sc["format_valid"])
                agg["name_match"] += int(sc["name_match"])
                agg["arg_match"] += int(sc["arg_match"])
            else:
                agg["n_neg"] += 1
                agg["over_trigger"] += int(sc["over_trigger"])
            done += 1
            if len(samples) < keep_samples:
                samples.append({"type": item["type"], "score": sc,
                                "answer": post_think(text)[:400]})
        print(f"[gpu{gpu_id} toolcall] {done}/{len(rendered)} "
              f"fmt={agg['format_valid']}/{agg['n_pos']} "
              f"over={agg['over_trigger']}/{agg['n_neg']}", flush=True)
    agg["gen_time"] = time.time() - t0
    agg["gpu"] = gpu_id
    agg["samples"] = samples
    return agg


def worker(gpu_id, spec, items, batch_size, max_new, q):
    import torch
    torch.cuda.set_device(gpu_id)
    tok, model, _ = load_subject_model(spec, gpu_id)
    q.put(run_items(gpu_id, tok, model, items, batch_size, max_new))


# ============================ aggregation / flush ===========================
def _rate(num, den):
    return round(num / den, 4) if den else None


def aggregate(parts, spec):
    n_pos = sum(p["n_pos"] for p in parts)
    n_neg = sum(p["n_neg"] for p in parts)
    fv = sum(p["format_valid"] for p in parts)
    nm = sum(p["name_match"] for p in parts)
    am = sum(p["arg_match"] for p in parts)
    ov = sum(p["over_trigger"] for p in parts)
    gtok = sum(p["gen_tokens"] for p in parts)
    tsum = sum(p["gen_time"] for p in parts)
    tmax = max((p["gen_time"] for p in parts), default=0.0)
    fvr, nmr, amr = _rate(fv, n_pos), _rate(nm, n_pos), _rate(am, n_pos)
    ovr = _rate(ov, n_neg)
    return {
        "model": spec["kind"], "topk": spec.get("topk"),
        "n_pos": n_pos, "n_neg": n_neg,
        "format_validity": fvr, "format_validity_ci95": ci95_normal(fvr, n_pos) if fvr is not None else None,
        "name_acc": nmr, "name_acc_ci95": ci95_normal(nmr, n_pos) if nmr is not None else None,
        "arg_ast_acc": amr, "arg_ast_acc_ci95": ci95_normal(amr, n_pos) if amr is not None else None,
        "over_trigger_rate": ovr, "over_trigger_ci95": ci95_normal(ovr, n_neg) if ovr is not None else None,
        "counts": {"format_valid": fv, "name_match": nm, "arg_match": am, "over_trigger": ov},
        "gen_tokens": gtok,
        "tok_s": round(gtok / tsum, 2) if tsum else None,
        "tok_s_parallel": round(gtok / tmax, 2) if tmax else None,
    }


def flush(out_path, result, meta, samples):
    data = {"_meta": meta, "results": result, "samples": samples}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


def print_table(r):
    print("\n" + "=" * 66)
    print(f"model={r['model']}  topk={r['topk']}   (pos={r['n_pos']} neg={r['n_neg']})")
    print("-" * 66)
    print(f"  format-validity : {r['format_validity']}  ±{r['format_validity_ci95']}")
    print(f"  name accuracy   : {r['name_acc']}  ±{r['name_acc_ci95']}")
    print(f"  arg AST match   : {r['arg_ast_acc']}  ±{r['arg_ast_acc_ci95']}")
    print(f"  over-trigger    : {r['over_trigger_rate']}  ±{r['over_trigger_ci95']}  (lower better)")
    print(f"  tok/s {r['tok_s']}  (2-gpu wall {r['tok_s_parallel']})")
    print("=" * 66)


# =============================== self-test ==================================
def self_test() -> int:
    """CPU-only: drive the scorers with hand-labelled cases (no model)."""
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "search"}},
    ]
    tool_names = ["get_weather", "search"]

    correct_single = (
        "<think>\ncall weather\n</think>\n\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>")
    multi_typed = (
        "reasoning</think>\n"
        "<tool_call>\n<function=search>\n<parameter=query>\nbest pizza\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n<parameter=verified>\ntrue\n</parameter>\n</function>\n</tool_call>")
    two_calls = (
        "</think>\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n</tool_call>")
    plain = "</think>\n\nThe capital of France is Paris. Happy to help further!"
    hallucinated = ("</think>\n<tool_call>\n<function=delete_everything>\n"
                    "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>")
    wrong_arg = ("</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n</tool_call>")
    think_leak = (  # a call appears only INSIDE think; the answer is plain text
        "<think>\nmaybe <tool_call>\n<function=get_weather>\n<parameter=city>\nX\n</parameter>\n</function>\n</tool_call>\n</think>\n\nParis.")

    cases = []

    def pos(text, gold, exp_fv, exp_nm, exp_am, label):
        item = {"type": "pos", "gold_calls": gold, "tool_names": tool_names}
        sc = score_item(item, text)
        cases.append((label, sc == {"format_valid": exp_fv, "name_match": exp_nm, "arg_match": exp_am}, sc,
                      {"format_valid": exp_fv, "name_match": exp_nm, "arg_match": exp_am}))

    def neg(text, exp_over, label):
        sc = score_item({"type": "neg"}, text)
        cases.append((label, sc == {"over_trigger": exp_over}, sc, {"over_trigger": exp_over}))

    pos(correct_single, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        True, True, True, "pos: single correct call")
    pos(multi_typed, [{"name": "search", "arguments": {"query": "best pizza", "limit": 5, "verified": True}}],
        True, True, True, "pos: int/bool coercion")
    pos(two_calls, [{"name": "get_weather", "arguments": {"city": "Tokyo"}},
                    {"name": "get_weather", "arguments": {"city": "Osaka"}}],
        True, True, True, "pos: two same-name calls (bijection)")
    pos(wrong_arg, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        True, True, False, "pos: right name, wrong arg")
    pos(hallucinated, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        False, False, False, "pos: hallucinated function name -> invalid")
    pos(plain, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        False, False, False, "pos: no call emitted -> invalid")

    neg(plain, False, "neg: plain answer -> no over-trigger")
    neg(correct_single, True, "neg: emitted a call -> over-trigger")
    neg(think_leak, False, "neg: call only inside <think> -> not over-trigger")

    width = max(len(c[0]) for c in cases)
    ok = True
    print("=" * (width + 26))
    print("eval_toolcall self-test (CPU, no model)")
    print("=" * (width + 26))
    for label, passed, got, exp in cases:
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {label:<{width}}")
        if not passed:
            ok = False
            print(f"         got={got} exp={exp}")
    # a couple of direct parser assertions
    p = parse_tool_calls(post_think(multi_typed))
    parser_ok = (len(p) == 1 and p[0]["name"] == "search"
                 and p[0]["arguments"] == {"query": "best pizza", "limit": "5", "verified": "true"})
    print(f"  [{'PASS' if parser_ok else 'FAIL'}] parser: names+params extracted verbatim")
    ok = ok and parser_ok
    print("=" * (width + 26))
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


# ================================== main ====================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only scorer check (no GPU/model), then exit")
    ap.add_argument("--model", choices=["base", "patched", "nvfp4", "dense"])
    ap.add_argument("--patch", default=None)
    ap.add_argument("--nvfp4-model-path", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--n-pos", type=int, default=500)
    ap.add_argument("--n-neg", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--report-dir", default=REPORT_DIR)
    ap.add_argument("--dry-build", action="store_true",
                    help="CPU: build eval items and print a few, then exit (no model)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if args.dry_build:
        items = build_eval_items(args.n_pos, args.n_neg)
        npos = sum(1 for it in items if it["type"] == "pos")
        print(f"built {len(items)} items: pos={npos} neg={len(items) - npos}")
        for it in items[:2] + items[-2:]:
            print(json.dumps({k: it[k] for k in it if k != "tools"}, ensure_ascii=False)[:500])
        return

    if not args.model:
        ap.error("--model is required (or use --self-test)")

    try:
        spec = resolve_model_spec(args.model, model_path=args.model_path, patch=args.patch,
                                  nvfp4_model_path=args.nvfp4_model_path, topk=args.topk)
    except ValueError as e:
        ap.error(str(e))

    gpus = [int(x) for x in args.gpus.split(",")]
    assert len(gpus) == 2, "wired for exactly 2 GPUs"

    items = build_eval_items(args.n_pos, args.n_neg)
    npos = sum(1 for it in items if it["type"] == "pos")
    print(f"built {len(items)} eval items (pos={npos} neg={len(items) - npos})", flush=True)
    subsets = {gpus[0]: items[0::2], gpus[1]: items[1::2]}

    tag = args.tag or f"toolcall_{spec['kind']}_k{spec.get('topk')}"
    out_path = os.path.join(args.report_dir, f"{tag}.json")
    meta = {"model": spec["kind"], "model_path": spec["model_path"], "patch": spec.get("patch"),
            "topk": spec.get("topk"), "n_pos": npos, "n_neg": len(items) - npos,
            "batch_size": args.batch_size, "max_new": args.max_new, "gpus": gpus,
            "heldout_seed": HELDOUT_SEED, "data": TOOLCALL_DATA,
            "note": "reasoning model; answers parsed post-</think>; Qwen3.6 XML tool-call format"}

    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(g, spec, subsets[g], args.batch_size, args.max_new, q))
             for g in gpus]
    for p in procs:
        p.start()
    parts = [q.get() for _ in gpus]
    for p in procs:
        p.join()

    result = aggregate(parts, spec)
    samples = [s for p in parts for s in p["samples"]]
    flush(out_path, result, meta, samples)
    print_table(result)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
