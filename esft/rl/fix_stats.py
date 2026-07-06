#!/usr/bin/env python
"""Recompute prompt_tokens (correctly) and clean strategy labels over the
already-built grpo_prompts.jsonl, then rewrite jsonl + stats."""
import json, time
from collections import Counter
from transformers import AutoTokenizer

JSONL = "~/esft/data/rl/grpo_prompts.jsonl"
STATS = "~/esft/data/rl/grpo_prompts_stats.json"
TOK = "~/esft-work/models/Qwen3.6-35B-A3B"

def clean_strategy(iid):
    # {owner__repo}.{commit8}.{strategy}__{hash}
    parts = iid.split(".", 2)
    if len(parts) < 3:
        return "other"
    tail = parts[2]                      # strategy__hash  OR  numeric
    return tail.rsplit("__", 1)[0] if "__" in tail else "".join(c for c in tail if not c.isdigit()) or "numeric"

def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    recs = [json.loads(l) for l in open(JSONL)]
    print("records:", len(recs), flush=True)
    tok_lens = []
    by_strat = Counter()
    for i, r in enumerate(recs):
        pm = r["prompt_messages"]
        s = tok.apply_chat_template(pm, add_generation_prompt=True, tokenize=False)
        tlen = len(tok(s, add_special_tokens=False).input_ids)
        r["prompt_tokens"] = tlen
        r["strategy"] = clean_strategy(r["instance_id"])
        tok_lens.append(tlen)
        by_strat[r["strategy"]] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)
    # rewrite jsonl
    with open(JSONL, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tok_lens.sort()
    def pct(p): return tok_lens[min(len(tok_lens) - 1, int(p * len(tok_lens)))]
    d = json.load(open(STATS))
    d["stats"]["kept_by_strategy"] = dict(by_strat.most_common())
    d["stats"].pop("dropped_by_strategy", None)
    d["stats"]["prompt_token_len"] = {
        "n": len(tok_lens), "min": tok_lens[0], "p50": pct(.50), "p90": pct(.90),
        "p95": pct(.95), "p99": pct(.99), "max": tok_lens[-1],
        "mean": round(sum(tok_lens) / len(tok_lens), 1),
        "gt_32768": sum(1 for x in tok_lens if x > 32768),
        "gt_16384": sum(1 for x in tok_lens if x > 16384),
        "gt_8192": sum(1 for x in tok_lens if x > 8192),
    }
    # refresh samples' token counts
    for s in d.get("samples", []):
        for r in recs:
            if r["instance_id"] == s["instance_id"]:
                s["prompt_tokens"] = r["prompt_tokens"]; break
    d["stats"]["token_fix_elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(d, open(STATS, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({"kept_by_strategy": dict(by_strat.most_common()),
                      "prompt_token_len": d["stats"]["prompt_token_len"]}, indent=2))

if __name__ == "__main__":
    main()
